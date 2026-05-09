use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    pub fn is_buy(self) -> bool {
        matches!(self, Side::Buy)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    Limit,
    Market,
    PostOnly,
    Ioc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub client_id: u64,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub price: Option<Decimal>,
    pub size: Decimal,
    pub reduce_only: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub order_id: u64,
    pub client_id: u64,
    pub symbol: String,
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
    pub fee: Decimal,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Position {
    pub symbol: String,
    pub size: Decimal,
    pub entry_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub last_update: DateTime<Utc>,
}

impl Position {
    pub fn is_flat(&self) -> bool {
        self.size.is_zero()
    }

    pub fn notional(&self, mark_price: Decimal) -> Decimal {
        self.size.abs() * mark_price
    }
}

/// Trade signal flowing from Python research stack into Rust execution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKind {
    /// Vol spread arb: HIP-3 IV vs IBKR IV
    VolSpread,
    /// Term structure calendar spread
    Calendar,
    /// Risk reversal (skew arb)
    RiskReversal,
    /// Gamma scalp setup
    GammaScalp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub seq: u64,
    pub kind: SignalKind,
    pub symbol: String,
    pub side: Side,
    pub size: Decimal,
    pub limit_price: Option<Decimal>,
    pub edge_bps: i32,
    pub confidence: f32,
    pub ttl_ms: u32,
    pub timestamp_ns: u64,
}
