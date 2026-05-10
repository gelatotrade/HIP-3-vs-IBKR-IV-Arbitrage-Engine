use crate::state::{ExecutionState, OpenOrder};
use anyhow::Result;
use chrono::Utc;
use common::{Order, OrderType, Side, Signal};
use hyperliquid_client::HyperliquidRest;
use orderbook::Orderbook;
use persistence::TradeLog;
use risk::{RiskManager, RiskRejection};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

pub struct ExecutionEngine {
    rest: Arc<HyperliquidRest>,
    risk: Arc<RiskManager>,
    state: Arc<ExecutionState>,
    log: TradeLog,
    asset_ids: HashMap<String, u32>,
    books: HashMap<String, Arc<Orderbook>>,
    next_client_id: AtomicU64,
}

impl ExecutionEngine {
    pub fn new(
        rest: Arc<HyperliquidRest>,
        risk: Arc<RiskManager>,
        state: Arc<ExecutionState>,
        log: TradeLog,
        asset_ids: HashMap<String, u32>,
        books: HashMap<String, Arc<Orderbook>>,
    ) -> Self {
        Self {
            rest,
            risk,
            state,
            log,
            asset_ids,
            books,
            next_client_id: AtomicU64::new(now_nanos()),
        }
    }

    /// Process one signal: validate, risk-check, place order.
    pub async fn handle_signal(&self, signal: &Signal) -> Result<HandleOutcome> {
        let started = Instant::now();
        let _ = self.log.record_signal(signal);
        metrics::metrics()
            .signals_received
            .with_label_values(&[signal_kind_label(signal)])
            .inc();

        let book = match self.books.get(&signal.symbol) {
            Some(b) => b.load(),
            None => {
                warn!(symbol = %signal.symbol, "no book for signal");
                return Ok(HandleOutcome::NoBook);
            }
        };

        let mark_price = match book.mid() {
            Some(m) => m,
            None => return Ok(HandleOutcome::EmptyBook),
        };

        let limit_price = signal
            .limit_price
            .unwrap_or_else(|| improve_price(mark_price, signal.side));
        let client_id = self.next_client_id.fetch_add(1, Ordering::Relaxed);
        let order = Order {
            client_id,
            symbol: signal.symbol.clone(),
            side: signal.side,
            order_type: OrderType::Limit,
            price: Some(limit_price),
            size: signal.size,
            reduce_only: false,
            created_at: Utc::now(),
        };

        let now_ms = chrono::Utc::now().timestamp_millis() as u64;
        let positions = self.state.positions();
        if let Err(rejection) = self
            .risk
            .check_order(&order, mark_price, &positions, now_ms)
        {
            warn!(?rejection, "risk rejected order");
            metrics::metrics()
                .orders_rejected
                .with_label_values(&[risk_label(&rejection)])
                .inc();
            return Ok(HandleOutcome::RiskRejected(rejection));
        }

        self.log.record_order(&order)?;
        self.state.register_open(OpenOrder {
            client_id,
            exchange_oid: None,
            symbol: signal.symbol.clone(),
            side: signal.side,
            remaining_size: signal.size,
        });

        let asset_id = match self.asset_ids.get(&signal.symbol) {
            Some(&id) => id,
            None => {
                warn!(symbol = %signal.symbol, "unknown asset id");
                self.state.close_open(client_id);
                return Ok(HandleOutcome::UnknownSymbol);
            }
        };

        let api_t = Instant::now();
        let result = self
            .rest
            .place_order(
                asset_id,
                signal.side == Side::Buy,
                &limit_price.to_string(),
                &signal.size.to_string(),
                false,
                "Gtc",
            )
            .await;
        metrics::metrics()
            .api_latency_ms
            .with_label_values(&["place_order"])
            .observe(api_t.elapsed().as_secs_f64() * 1000.0);

        match result {
            Ok(resp) => {
                metrics::metrics()
                    .orders_placed
                    .with_label_values(&[&signal.symbol, side_label(signal.side)])
                    .inc();
                let _ = self.log.update_order_status(client_id, "submitted", None);
                let elapsed_us = started.elapsed().as_micros() as f64;
                metrics::metrics()
                    .signal_to_order_latency_us
                    .with_label_values(&[signal_kind_label(signal)])
                    .observe(elapsed_us);
                debug!(client_id, status = %resp.status, "order submitted");
                Ok(HandleOutcome::Placed { client_id })
            }
            Err(e) => {
                self.state.close_open(client_id);
                let _ = self
                    .log
                    .update_order_status(client_id, "exchange_error", None);
                metrics::metrics()
                    .orders_rejected
                    .with_label_values(&["exchange_error"])
                    .inc();
                warn!(error = %e, "exchange rejected order");
                Ok(HandleOutcome::ExchangeError(e.to_string()))
            }
        }
    }

    pub fn state(&self) -> &Arc<ExecutionState> {
        &self.state
    }
}

#[derive(Debug)]
pub enum HandleOutcome {
    Placed { client_id: u64 },
    NoBook,
    EmptyBook,
    UnknownSymbol,
    RiskRejected(RiskRejection),
    ExchangeError(String),
}

fn improve_price(mid: Decimal, side: Side) -> Decimal {
    let bps = Decimal::new(2, 4); // 2 bps inside mid
    if side.is_buy() {
        mid * (Decimal::ONE - bps)
    } else {
        mid * (Decimal::ONE + bps)
    }
}

fn signal_kind_label(s: &Signal) -> &'static str {
    use common::SignalKind::*;
    match s.kind {
        VolSpread => "vol_spread",
        Calendar => "calendar",
        RiskReversal => "risk_reversal",
        GammaScalp => "gamma_scalp",
    }
}

fn side_label(s: Side) -> &'static str {
    if s.is_buy() {
        "buy"
    } else {
        "sell"
    }
}

fn risk_label(r: &RiskRejection) -> &'static str {
    match r {
        RiskRejection::KillSwitch => "kill_switch",
        RiskRejection::DailyLoss { .. } => "daily_loss",
        RiskRejection::SizeBelowMin { .. } => "size_below_min",
        RiskRejection::OrderTooLarge { .. } => "order_too_large",
        RiskRejection::SymbolNotional { .. } => "symbol_notional",
        RiskRejection::TotalNotional { .. } => "total_notional",
        RiskRejection::RateLimited { .. } => "rate_limited",
    }
}

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
