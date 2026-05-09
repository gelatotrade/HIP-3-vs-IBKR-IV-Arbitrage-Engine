//! Hyperliquid HIP-3 perp markets client.
//!
//! Provides REST + WebSocket access plus EIP-712 typed-data signing
//! for authenticated order placement and cancellation.

pub mod rest;
pub mod signing;
pub mod types;
pub mod ws;

pub use rest::HyperliquidRest;
pub use signing::Wallet;
pub use ws::{HyperliquidWs, WsEvent};

pub const MAINNET_REST: &str = "https://api.hyperliquid.xyz";
pub const MAINNET_WS: &str = "wss://api.hyperliquid.xyz/ws";
pub const TESTNET_REST: &str = "https://api.hyperliquid-testnet.xyz";
pub const TESTNET_WS: &str = "wss://api.hyperliquid-testnet.xyz/ws";
