//! Pre-trade risk checks and kill-switch.

use common::{Order, Position, Side};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;
use tracing::{error, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Max absolute notional position per symbol, in USD.
    pub max_notional_per_symbol: Decimal,
    /// Max aggregate notional across all positions.
    pub max_total_notional: Decimal,
    /// Max single-order notional in USD.
    pub max_order_notional: Decimal,
    /// Max daily realised loss before kill-switch trips.
    pub max_daily_loss: Decimal,
    /// Reject orders with size <= this threshold.
    pub min_order_size: Decimal,
    /// Max orders per second (sliding window).
    pub max_orders_per_sec: u32,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_notional_per_symbol: dec!(50_000),
            max_total_notional: dec!(200_000),
            max_order_notional: dec!(25_000),
            max_daily_loss: dec!(5_000),
            min_order_size: dec!(0.001),
            max_orders_per_sec: 10,
        }
    }
}

#[derive(Debug, Error)]
pub enum RiskRejection {
    #[error("kill switch is engaged")]
    KillSwitch,
    #[error("daily loss limit hit: pnl={pnl}, limit={limit}")]
    DailyLoss { pnl: Decimal, limit: Decimal },
    #[error("order size {size} below minimum {min}")]
    SizeBelowMin { size: Decimal, min: Decimal },
    #[error("order notional {notional} exceeds {limit}")]
    OrderTooLarge { notional: Decimal, limit: Decimal },
    #[error("symbol notional would be {projected}, exceeds {limit}")]
    SymbolNotional { projected: Decimal, limit: Decimal },
    #[error("total notional would be {projected}, exceeds {limit}")]
    TotalNotional { projected: Decimal, limit: Decimal },
    #[error("rate limit exceeded: {orders} orders in last second")]
    RateLimited { orders: u32 },
}

pub struct RiskManager {
    limits: RwLock<RiskLimits>,
    kill_switch: AtomicBool,
    realised_pnl: RwLock<Decimal>,
    recent_order_ts: RwLock<Vec<u64>>,
}

impl RiskManager {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits: RwLock::new(limits),
            kill_switch: AtomicBool::new(false),
            realised_pnl: RwLock::new(Decimal::ZERO),
            recent_order_ts: RwLock::new(Vec::with_capacity(64)),
        }
    }

    pub fn engage_kill_switch(&self) {
        error!("kill switch engaged");
        self.kill_switch.store(true, Ordering::SeqCst);
    }

    pub fn release_kill_switch(&self) {
        warn!("kill switch released");
        self.kill_switch.store(false, Ordering::SeqCst);
    }

    pub fn is_killed(&self) -> bool {
        self.kill_switch.load(Ordering::Acquire)
    }

    pub fn record_pnl(&self, delta: Decimal) {
        let mut pnl = self.realised_pnl.write();
        *pnl += delta;
        let limit = self.limits.read().max_daily_loss;
        if *pnl <= -limit {
            drop(pnl);
            self.engage_kill_switch();
        }
    }

    pub fn check_order(
        &self,
        order: &Order,
        mark_price: Decimal,
        positions: &HashMap<String, Position>,
        now_ms: u64,
    ) -> Result<(), RiskRejection> {
        if self.is_killed() {
            return Err(RiskRejection::KillSwitch);
        }

        let limits = self.limits.read().clone();

        if order.size < limits.min_order_size {
            return Err(RiskRejection::SizeBelowMin {
                size: order.size,
                min: limits.min_order_size,
            });
        }

        let order_px = order.price.unwrap_or(mark_price);
        let order_notional = order.size * order_px;
        if order_notional > limits.max_order_notional {
            return Err(RiskRejection::OrderTooLarge {
                notional: order_notional,
                limit: limits.max_order_notional,
            });
        }

        let current = positions
            .get(&order.symbol)
            .map(|p| p.size)
            .unwrap_or(Decimal::ZERO);
        let signed_delta = if order.side == Side::Buy {
            order.size
        } else {
            -order.size
        };
        let projected_size = (current + signed_delta).abs();
        let projected_notional = projected_size * mark_price;
        if projected_notional > limits.max_notional_per_symbol {
            return Err(RiskRejection::SymbolNotional {
                projected: projected_notional,
                limit: limits.max_notional_per_symbol,
            });
        }

        let mut total: Decimal = positions
            .iter()
            .filter(|(s, _)| *s != &order.symbol)
            .map(|(_, p)| p.size.abs() * mark_price)
            .sum();
        total += projected_notional;
        if total > limits.max_total_notional {
            return Err(RiskRejection::TotalNotional {
                projected: total,
                limit: limits.max_total_notional,
            });
        }

        let mut ts = self.recent_order_ts.write();
        ts.retain(|&t| now_ms.saturating_sub(t) < 1000);
        if ts.len() as u32 >= limits.max_orders_per_sec {
            return Err(RiskRejection::RateLimited {
                orders: ts.len() as u32,
            });
        }
        ts.push(now_ms);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use common::{OrderType, Side};

    fn make_order(size: Decimal, price: Decimal, side: Side) -> Order {
        Order {
            client_id: 1,
            symbol: "AAPL".to_string(),
            side,
            order_type: OrderType::Limit,
            price: Some(price),
            size,
            reduce_only: false,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn rejects_when_killed() {
        let rm = RiskManager::new(RiskLimits::default());
        rm.engage_kill_switch();
        let positions = HashMap::new();
        let order = make_order(dec!(1), dec!(100), Side::Buy);
        assert!(matches!(
            rm.check_order(&order, dec!(100), &positions, 0),
            Err(RiskRejection::KillSwitch)
        ));
    }

    #[test]
    fn rejects_oversized_order() {
        let rm = RiskManager::new(RiskLimits::default());
        let positions = HashMap::new();
        let order = make_order(dec!(1000), dec!(100), Side::Buy);
        assert!(matches!(
            rm.check_order(&order, dec!(100), &positions, 0),
            Err(RiskRejection::OrderTooLarge { .. })
        ));
    }

    #[test]
    fn accepts_normal_order() {
        let rm = RiskManager::new(RiskLimits::default());
        let positions = HashMap::new();
        let order = make_order(dec!(1), dec!(100), Side::Buy);
        assert!(rm.check_order(&order, dec!(100), &positions, 0).is_ok());
    }

    #[test]
    fn pnl_loss_engages_kill() {
        let rm = RiskManager::new(RiskLimits::default());
        rm.record_pnl(dec!(-6_000));
        assert!(rm.is_killed());
    }
}
