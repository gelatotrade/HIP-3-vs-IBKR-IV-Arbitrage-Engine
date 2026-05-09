use anyhow::{Context, Result};
use clap::Parser;
use execution::{ExecutionEngine, ExecutionState};
use hyperliquid_client::{
    HyperliquidRest, HyperliquidWs, Wallet, WsEvent, MAINNET_REST, MAINNET_WS, TESTNET_REST,
    TESTNET_WS,
};
use orderbook::Orderbook;
use persistence::TradeLog;
use risk::{RiskLimits, RiskManager};
use signal_bridge::SignalReader;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "trade-bot", about = "HIP-3 vs IBKR live execution engine")]
struct Cli {
    /// Use Hyperliquid testnet endpoints.
    #[arg(long, env = "TRADE_BOT_TESTNET")]
    testnet: bool,

    /// Path to the shared-memory ring buffer file (Python writes signals here).
    #[arg(
        long,
        env = "TRADE_BOT_RINGBUFFER",
        default_value = "/tmp/hip3_signals.ring"
    )]
    ringbuffer: PathBuf,

    /// SQLite trade log path.
    #[arg(long, env = "TRADE_BOT_DB", default_value = "/tmp/hip3_tradelog.db")]
    db: PathBuf,

    /// Hex-encoded private key (do NOT pass via CLI in production; use env).
    #[arg(long, env = "HYPERLIQUID_PRIVATE_KEY")]
    private_key: Option<String>,

    /// Comma-separated list of symbols to subscribe to (e.g. "AAPL,NVDA,TSLA").
    #[arg(long, env = "TRADE_BOT_SYMBOLS", default_value = "AAPL,NVDA,TSLA")]
    symbols: String,

    /// Prometheus metrics bind address.
    #[arg(long, env = "TRADE_BOT_METRICS", default_value = "127.0.0.1:9091")]
    metrics_addr: String,

    /// Dry-run: don't actually place orders, just log them.
    #[arg(long, env = "TRADE_BOT_DRY_RUN")]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    info!(?cli.testnet, ?cli.dry_run, "starting trade-bot");

    let symbols: Vec<String> = cli
        .symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if symbols.is_empty() {
        anyhow::bail!("no symbols configured");
    }

    let (rest_url, ws_url) = if cli.testnet {
        (TESTNET_REST, TESTNET_WS)
    } else {
        (MAINNET_REST, MAINNET_WS)
    };

    // Build REST client + optional wallet.
    let mut rest = HyperliquidRest::new(rest_url, !cli.testnet)?;
    if let Some(key) = cli.private_key.as_deref() {
        if !cli.dry_run {
            let wallet = Wallet::from_hex(key).context("invalid private key")?;
            info!(addr = %wallet.address_hex(), "wallet loaded");
            rest = rest.with_wallet(wallet);
        }
    } else if !cli.dry_run {
        anyhow::bail!("private key required for live trading; pass --dry-run otherwise");
    }
    let rest = Arc::new(rest);

    // Resolve asset IDs.
    let universe = rest.meta().await.context("failed to load universe")?;
    let mut asset_ids: HashMap<String, u32> = HashMap::new();
    for (idx, asset) in universe.universe.iter().enumerate() {
        asset_ids.insert(asset.name.clone(), idx as u32);
    }
    info!(n = asset_ids.len(), "loaded universe");

    // Initialise per-symbol orderbooks.
    let books: HashMap<String, Arc<Orderbook>> = symbols
        .iter()
        .map(|s| (s.clone(), Arc::new(Orderbook::new())))
        .collect();

    // Trade log + state + risk.
    let log = TradeLog::open(&cli.db).context("failed to open trade log")?;
    let state = Arc::new(ExecutionState::new());
    let risk = Arc::new(RiskManager::new(RiskLimits::default()));

    let engine = Arc::new(ExecutionEngine::new(
        rest.clone(),
        risk.clone(),
        state.clone(),
        log.clone(),
        asset_ids.clone(),
        books.clone(),
    ));

    // Spawn metrics endpoint.
    let metrics_addr = cli.metrics_addr.parse()?;
    tokio::spawn(async move {
        if let Err(e) = metrics::serve(metrics_addr).await {
            error!(error = %e, "metrics server failed");
        }
    });

    // Spawn WebSocket consumer for orderbook updates.
    let mut ws = HyperliquidWs::new(ws_url);
    for s in &symbols {
        ws.subscribe_l2_book(s);
    }
    let mut ws_rx = ws.spawn();
    let books_for_ws = books.clone();
    tokio::spawn(async move {
        while let Some(event) = ws_rx.recv().await {
            match event {
                WsEvent::L2Book {
                    coin,
                    bids,
                    asks,
                    time,
                } => {
                    if let Some(book) = books_for_ws.get(&coin) {
                        book.apply_snapshot(&bids, &asks, time);
                    }
                }
                WsEvent::Connected => info!("ws connected"),
                WsEvent::Disconnected => warn!("ws disconnected"),
                _ => {}
            }
        }
    });

    // Open the shared-memory ring buffer.
    let mut reader =
        SignalReader::open(&cli.ringbuffer).context("failed to open signal ring buffer")?;
    info!(path = %cli.ringbuffer.display(), "signal reader opened");

    // Hot loop: poll the ring buffer with a tiny adaptive sleep.
    let mut idle_us = 1u64;
    loop {
        match reader.try_pop() {
            Ok(Some(signal)) => {
                idle_us = 1;
                if cli.dry_run {
                    info!(?signal, "DRY RUN — would place order");
                    continue;
                }
                let engine = engine.clone();
                tokio::spawn(async move {
                    match engine.handle_signal(&signal).await {
                        Ok(outcome) => info!(?outcome, "signal handled"),
                        Err(e) => error!(error = %e, "signal handler failed"),
                    }
                });
            }
            Ok(None) => {
                // Adaptive backoff: 1µs initial, doubling up to 1ms when idle.
                if idle_us < 1_000 {
                    idle_us = (idle_us * 2).min(1_000);
                }
                sleep(Duration::from_micros(idle_us)).await;
            }
            Err(e) => {
                error!(error = %e, "ring buffer read error");
                sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,trade_bot=debug"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .compact()
        .init();
}
