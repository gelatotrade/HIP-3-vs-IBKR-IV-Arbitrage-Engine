//! Local L2 orderbook state, kept in sync from WebSocket snapshots/diffs.

use arc_swap::ArcSwap;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct BookSnapshot {
    pub bids: Vec<(Decimal, Decimal)>,
    pub asks: Vec<(Decimal, Decimal)>,
    pub last_update_ms: u64,
}

impl BookSnapshot {
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|(p, _)| *p)
    }

    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|(p, _)| *p)
    }

    pub fn mid(&self) -> Option<Decimal> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        Some((b + a) / Decimal::from(2))
    }

    pub fn spread_bps(&self) -> Option<Decimal> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        let mid = (b + a) / Decimal::from(2);
        if mid.is_zero() {
            return None;
        }
        Some((a - b) / mid * Decimal::from(10_000))
    }
}

/// Lock-free orderbook updated via `ArcSwap`. Readers never block writers.
pub struct Orderbook {
    snapshot: ArcSwap<BookSnapshot>,
}

impl Orderbook {
    pub fn new() -> Self {
        Self {
            snapshot: ArcSwap::from_pointee(BookSnapshot::default()),
        }
    }

    pub fn load(&self) -> Arc<BookSnapshot> {
        self.snapshot.load_full()
    }

    pub fn apply_snapshot(
        &self,
        bids: &[(String, String)],
        asks: &[(String, String)],
        time_ms: u64,
    ) {
        let bids = parse_levels(bids, true);
        let asks = parse_levels(asks, false);
        self.snapshot.store(Arc::new(BookSnapshot {
            bids,
            asks,
            last_update_ms: time_ms,
        }));
    }
}

impl Default for Orderbook {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_levels(raw: &[(String, String)], descending: bool) -> Vec<(Decimal, Decimal)> {
    let mut map = BTreeMap::new();
    for (px, sz) in raw {
        let p = match Decimal::from_str(px) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let s = Decimal::from_str(sz).unwrap_or_default();
        if !s.is_zero() {
            map.insert(p, s);
        }
    }
    let mut out: Vec<_> = map.into_iter().collect();
    if descending {
        out.reverse();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_apply_and_query() {
        let book = Orderbook::new();
        book.apply_snapshot(
            &[("100.0".into(), "5".into()), ("99.5".into(), "10".into())],
            &[("100.5".into(), "3".into()), ("101.0".into(), "8".into())],
            12345,
        );
        let snap = book.load();
        assert_eq!(
            snap.best_bid().unwrap(),
            Decimal::from_str("100.0").unwrap()
        );
        assert_eq!(
            snap.best_ask().unwrap(),
            Decimal::from_str("100.5").unwrap()
        );
        assert_eq!(snap.last_update_ms, 12345);
    }

    #[test]
    fn book_spread_bps() {
        let book = Orderbook::new();
        book.apply_snapshot(
            &[("100.0".into(), "1".into())],
            &[("100.1".into(), "1".into())],
            0,
        );
        let snap = book.load();
        let spread = snap.spread_bps().unwrap();
        assert!(spread > Decimal::from(9) && spread < Decimal::from(11));
    }
}
