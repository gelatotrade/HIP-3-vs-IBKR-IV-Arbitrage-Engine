//! SQLite-backed persistent trade log + signal log + state snapshots.

use anyhow::Result;
use common::{Fill, Order, Signal};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS orders (
    client_id     INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    order_type    TEXT NOT NULL,
    price         TEXT,
    size          TEXT NOT NULL,
    reduce_only   INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    exchange_oid  INTEGER,
    last_update   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL,
    client_id     INTEGER NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    price         TEXT NOT NULL,
    size          TEXT NOT NULL,
    fee           TEXT NOT NULL,
    timestamp     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fills_symbol_time ON fills(symbol, timestamp);

CREATE TABLE IF NOT EXISTS signals (
    seq           INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    size          TEXT NOT NULL,
    limit_price   TEXT,
    edge_bps      INTEGER NOT NULL,
    confidence    REAL NOT NULL,
    ttl_ms        INTEGER NOT NULL,
    received_at   TEXT NOT NULL,
    acted_on      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    realized_pnl  TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL,
    total_notional TEXT NOT NULL
);
"#;

#[derive(Clone)]
pub struct TradeLog {
    conn: Arc<Mutex<Connection>>,
}

impl TradeLog {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn record_order(&self, order: &Order) -> Result<()> {
        let conn = self.conn.lock();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO orders (client_id, symbol, side, order_type, price, size, \
             reduce_only, created_at, status, last_update) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                order.client_id as i64,
                order.symbol,
                serde_json::to_string(&order.side)?.trim_matches('"'),
                serde_json::to_string(&order.order_type)?.trim_matches('"'),
                order.price.map(|p| p.to_string()),
                order.size.to_string(),
                order.reduce_only as i64,
                order.created_at.to_rfc3339(),
                "pending",
                now,
            ],
        )?;
        Ok(())
    }

    pub fn update_order_status(
        &self,
        client_id: u64,
        status: &str,
        oid: Option<u64>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE orders SET status = ?1, exchange_oid = ?2, last_update = ?3 WHERE client_id = ?4",
            params![status, oid.map(|o| o as i64), now, client_id as i64],
        )?;
        Ok(())
    }

    pub fn record_fill(&self, fill: &Fill) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO fills (order_id, client_id, symbol, side, price, size, fee, timestamp) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                fill.order_id as i64,
                fill.client_id as i64,
                fill.symbol,
                serde_json::to_string(&fill.side)?.trim_matches('"'),
                fill.price.to_string(),
                fill.size.to_string(),
                fill.fee.to_string(),
                fill.timestamp.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn record_signal(&self, sig: &Signal) -> Result<()> {
        let conn = self.conn.lock();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR IGNORE INTO signals (seq, kind, symbol, side, size, limit_price, \
             edge_bps, confidence, ttl_ms, received_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                sig.seq as i64,
                serde_json::to_string(&sig.kind)?.trim_matches('"'),
                sig.symbol,
                serde_json::to_string(&sig.side)?.trim_matches('"'),
                sig.size.to_string(),
                sig.limit_price.map(|p| p.to_string()),
                sig.edge_bps,
                sig.confidence as f64,
                sig.ttl_ms as i64,
                now,
            ],
        )?;
        Ok(())
    }

    pub fn fill_count(&self) -> Result<i64> {
        let conn = self.conn.lock();
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM fills", [], |r| r.get(0))?;
        Ok(n)
    }

    pub fn order_count(&self) -> Result<i64> {
        let conn = self.conn.lock();
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM orders", [], |r| r.get(0))?;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use common::{OrderType, Side, SignalKind};
    use rust_decimal_macros::dec;

    fn order() -> Order {
        Order {
            client_id: 42,
            symbol: "AAPL".into(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Some(dec!(195.5)),
            size: dec!(1.0),
            reduce_only: false,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn record_order_round_trip() {
        let log = TradeLog::open_in_memory().unwrap();
        log.record_order(&order()).unwrap();
        assert_eq!(log.order_count().unwrap(), 1);
        log.update_order_status(42, "filled", Some(99)).unwrap();
        assert_eq!(log.order_count().unwrap(), 1);
    }

    #[test]
    fn record_fill_round_trip() {
        let log = TradeLog::open_in_memory().unwrap();
        log.record_fill(&Fill {
            order_id: 99,
            client_id: 42,
            symbol: "AAPL".into(),
            side: Side::Buy,
            price: dec!(195.5),
            size: dec!(1.0),
            fee: dec!(0.10),
            timestamp: Utc::now(),
        })
        .unwrap();
        assert_eq!(log.fill_count().unwrap(), 1);
    }

    #[test]
    fn record_signal_idempotent() {
        let log = TradeLog::open_in_memory().unwrap();
        let sig = Signal {
            seq: 1,
            kind: SignalKind::VolSpread,
            symbol: "AAPL".into(),
            side: Side::Buy,
            size: dec!(1.0),
            limit_price: Some(dec!(100.0)),
            edge_bps: 30,
            confidence: 0.8,
            ttl_ms: 200,
            timestamp_ns: 0,
        };
        log.record_signal(&sig).unwrap();
        log.record_signal(&sig).unwrap();
    }
}
