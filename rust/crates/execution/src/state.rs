use common::{Fill, Position, Side};
use parking_lot::RwLock;
use rust_decimal::prelude::Signed;
use rust_decimal::Decimal;
use std::collections::HashMap;

/// In-memory execution state — positions, realised PnL, open orders.
#[derive(Default)]
pub struct ExecutionState {
    positions: RwLock<HashMap<String, Position>>,
    realized_pnl: RwLock<Decimal>,
    open_orders: RwLock<HashMap<u64, OpenOrder>>,
}

#[derive(Clone, Debug)]
pub struct OpenOrder {
    pub client_id: u64,
    pub exchange_oid: Option<u64>,
    pub symbol: String,
    pub side: Side,
    pub remaining_size: Decimal,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn positions(&self) -> HashMap<String, Position> {
        self.positions.read().clone()
    }

    pub fn position(&self, symbol: &str) -> Option<Position> {
        self.positions.read().get(symbol).cloned()
    }

    pub fn realized_pnl(&self) -> Decimal {
        *self.realized_pnl.read()
    }

    pub fn apply_fill(&self, fill: &Fill) -> Decimal {
        let mut positions = self.positions.write();
        let pos = positions
            .entry(fill.symbol.clone())
            .or_insert_with(|| Position {
                symbol: fill.symbol.clone(),
                ..Default::default()
            });

        let signed = if fill.side == Side::Buy {
            fill.size
        } else {
            -fill.size
        };
        let prev_size = pos.size;
        let new_size = prev_size + signed;

        // Position reducing/flipping → realise PnL on the closed portion.
        let realised_delta = if !prev_size.is_zero() && prev_size.signum() != signed.signum() {
            let closed = signed.abs().min(prev_size.abs());
            let pnl_per_unit = if prev_size.is_sign_positive() {
                fill.price - pos.entry_price
            } else {
                pos.entry_price - fill.price
            };
            pnl_per_unit * closed - fill.fee
        } else {
            -fill.fee
        };

        // Update entry price (volume-weighted) if adding to position.
        if prev_size.signum() == signed.signum() || prev_size.is_zero() {
            let total_abs = prev_size.abs() + signed.abs();
            if !total_abs.is_zero() {
                pos.entry_price =
                    (pos.entry_price * prev_size.abs() + fill.price * signed.abs()) / total_abs;
            }
        } else if new_size.signum() != prev_size.signum() && !new_size.is_zero() {
            pos.entry_price = fill.price;
        }

        pos.size = new_size;
        pos.realized_pnl += realised_delta;
        pos.last_update = fill.timestamp;

        if pos.size.is_zero() {
            pos.entry_price = Decimal::ZERO;
        }

        *self.realized_pnl.write() += realised_delta;
        realised_delta
    }

    pub fn register_open(&self, order: OpenOrder) {
        self.open_orders.write().insert(order.client_id, order);
    }

    pub fn close_open(&self, client_id: u64) {
        self.open_orders.write().remove(&client_id);
    }

    pub fn open_orders(&self) -> Vec<OpenOrder> {
        self.open_orders.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn fill(side: Side, price: Decimal, size: Decimal) -> Fill {
        Fill {
            order_id: 1,
            client_id: 1,
            symbol: "AAPL".into(),
            side,
            price,
            size,
            fee: dec!(0),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn opening_long_sets_entry() {
        let st = ExecutionState::new();
        st.apply_fill(&fill(Side::Buy, dec!(100), dec!(2)));
        let pos = st.position("AAPL").unwrap();
        assert_eq!(pos.size, dec!(2));
        assert_eq!(pos.entry_price, dec!(100));
    }

    #[test]
    fn closing_long_realises_pnl() {
        let st = ExecutionState::new();
        st.apply_fill(&fill(Side::Buy, dec!(100), dec!(2)));
        st.apply_fill(&fill(Side::Sell, dec!(110), dec!(2)));
        assert_eq!(st.realized_pnl(), dec!(20));
        let pos = st.position("AAPL").unwrap();
        assert_eq!(pos.size, dec!(0));
    }

    #[test]
    fn averaging_in_long() {
        let st = ExecutionState::new();
        st.apply_fill(&fill(Side::Buy, dec!(100), dec!(1)));
        st.apply_fill(&fill(Side::Buy, dec!(110), dec!(1)));
        let pos = st.position("AAPL").unwrap();
        assert_eq!(pos.size, dec!(2));
        assert_eq!(pos.entry_price, dec!(105));
    }

    #[test]
    fn flipping_position() {
        let st = ExecutionState::new();
        st.apply_fill(&fill(Side::Buy, dec!(100), dec!(1)));
        st.apply_fill(&fill(Side::Sell, dec!(110), dec!(3)));
        let pos = st.position("AAPL").unwrap();
        assert_eq!(pos.size, dec!(-2));
        assert_eq!(pos.entry_price, dec!(110));
        assert_eq!(st.realized_pnl(), dec!(10));
    }
}
