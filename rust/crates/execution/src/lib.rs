//! Execution engine: consumes signals, applies risk checks, places orders,
//! and tracks positions/PnL.

pub mod engine;
pub mod state;

pub use engine::ExecutionEngine;
pub use state::ExecutionState;
