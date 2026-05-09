//! Prometheus metrics + HTTP scrape endpoint.

use anyhow::Result;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec, CounterVec, Encoder,
    GaugeVec, HistogramVec, TextEncoder,
};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::OnceLock;
use tracing::{error, info};

pub struct Metrics {
    pub orders_placed: CounterVec,
    pub orders_rejected: CounterVec,
    pub fills: CounterVec,
    pub fill_notional: CounterVec,
    pub signals_received: CounterVec,
    pub realized_pnl: GaugeVec,
    pub unrealized_pnl: GaugeVec,
    pub position_size: GaugeVec,
    pub orderbook_age_ms: GaugeVec,
    pub signal_to_order_latency_us: HistogramVec,
    pub api_latency_ms: HistogramVec,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

pub fn metrics() -> &'static Metrics {
    METRICS.get_or_init(|| Metrics {
        orders_placed: register_counter_vec!(
            "trade_orders_placed_total",
            "Orders placed",
            &["symbol", "side"]
        )
        .unwrap(),
        orders_rejected: register_counter_vec!(
            "trade_orders_rejected_total",
            "Orders rejected by risk or exchange",
            &["reason"]
        )
        .unwrap(),
        fills: register_counter_vec!("trade_fills_total", "Fills received", &["symbol", "side"])
            .unwrap(),
        fill_notional: register_counter_vec!(
            "trade_fill_notional_usd",
            "Fill notional in USD",
            &["symbol"]
        )
        .unwrap(),
        signals_received: register_counter_vec!(
            "trade_signals_received_total",
            "Signals consumed from bridge",
            &["kind"]
        )
        .unwrap(),
        realized_pnl: register_gauge_vec!(
            "trade_realized_pnl_usd",
            "Realised PnL in USD",
            &["symbol"]
        )
        .unwrap(),
        unrealized_pnl: register_gauge_vec!(
            "trade_unrealized_pnl_usd",
            "Unrealised PnL in USD",
            &["symbol"]
        )
        .unwrap(),
        position_size: register_gauge_vec!(
            "trade_position_size",
            "Current position size",
            &["symbol"]
        )
        .unwrap(),
        orderbook_age_ms: register_gauge_vec!(
            "trade_orderbook_age_ms",
            "Age of last orderbook snapshot",
            &["symbol"]
        )
        .unwrap(),
        signal_to_order_latency_us: register_histogram_vec!(
            "trade_signal_to_order_latency_us",
            "Microseconds from signal received to order sent",
            &["kind"]
        )
        .unwrap(),
        api_latency_ms: register_histogram_vec!(
            "trade_api_latency_ms",
            "Hyperliquid API call latency",
            &["endpoint"]
        )
        .unwrap(),
    })
}

async fn handle(_req: Request<Body>) -> Result<Response<Body>, Infallible> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        error!(error = %e, "failed to encode metrics");
        return Ok(Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from("encode error"))
            .unwrap());
    }
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", encoder.format_type())
        .body(Body::from(buffer))
        .unwrap())
}

pub async fn serve(addr: SocketAddr) -> Result<()> {
    info!(%addr, "metrics endpoint listening");
    let make_svc = make_service_fn(|_| async { Ok::<_, Infallible>(service_fn(handle)) });
    Server::bind(&addr).serve(make_svc).await?;
    Ok(())
}
