//! Hyperliquid WebSocket client with auto-reconnect.

use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub enum WsEvent {
    L2Book {
        coin: String,
        bids: Vec<(String, String)>,
        asks: Vec<(String, String)>,
        time: u64,
    },
    Trades {
        coin: String,
        trades: Vec<Value>,
    },
    UserFills(Value),
    UserEvents(Value),
    Raw(Value),
    Connected,
    Disconnected,
}

pub struct HyperliquidWs {
    url: String,
    subscriptions: Vec<Value>,
    backoff_ms: u64,
}

impl HyperliquidWs {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            subscriptions: Vec::new(),
            backoff_ms: 100,
        }
    }

    pub fn subscribe_l2_book(&mut self, coin: &str) {
        self.subscriptions.push(json!({
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": coin}
        }));
    }

    pub fn subscribe_trades(&mut self, coin: &str) {
        self.subscriptions.push(json!({
            "method": "subscribe",
            "subscription": {"type": "trades", "coin": coin}
        }));
    }

    pub fn subscribe_user_fills(&mut self, user: &str) {
        self.subscriptions.push(json!({
            "method": "subscribe",
            "subscription": {"type": "userFills", "user": user}
        }));
    }

    /// Spawn a background task that maintains the connection and emits events.
    /// The returned receiver yields `WsEvent`s; closing the sender stops the task.
    pub fn spawn(self) -> mpsc::Receiver<WsEvent> {
        let (tx, rx) = mpsc::channel(1024);
        tokio::spawn(self.run(tx));
        rx
    }

    async fn run(self, tx: mpsc::Sender<WsEvent>) {
        let mut backoff_ms = self.backoff_ms;
        loop {
            match self.connect_once(&tx).await {
                Ok(()) => {
                    info!("ws stream ended cleanly, reconnecting");
                    backoff_ms = self.backoff_ms;
                }
                Err(e) => {
                    warn!(error = %e, backoff_ms, "ws disconnect, will retry");
                    if tx.send(WsEvent::Disconnected).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(30_000);
                }
            }
        }
    }

    async fn connect_once(&self, tx: &mpsc::Sender<WsEvent>) -> Result<()> {
        let (ws, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws.split();

        for sub in &self.subscriptions {
            write.send(Message::Text(sub.to_string())).await?;
        }

        if tx.send(WsEvent::Connected).await.is_err() {
            return Ok(());
        }

        while let Some(msg) = read.next().await {
            let msg = msg?;
            match msg {
                Message::Text(text) => {
                    let parsed: Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!(error = %e, "failed to parse ws message");
                            continue;
                        }
                    };
                    let event = parse_event(parsed);
                    if tx.send(event).await.is_err() {
                        return Ok(());
                    }
                }
                Message::Ping(p) => {
                    write.send(Message::Pong(p)).await?;
                }
                Message::Close(_) => {
                    return Err(anyhow!("server closed connection"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn parse_event(v: Value) -> WsEvent {
    let channel = v
        .get("channel")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let data = v.get("data").cloned().unwrap_or(Value::Null);

    match channel.as_str() {
        "l2Book" => {
            let coin = data
                .get("coin")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let levels = data.get("levels").cloned().unwrap_or(json!([[], []]));
            let bids = parse_levels(&levels[0]);
            let asks = parse_levels(&levels[1]);
            let time = data.get("time").and_then(|t| t.as_u64()).unwrap_or(0);
            WsEvent::L2Book {
                coin,
                bids,
                asks,
                time,
            }
        }
        "trades" => {
            let coin = v
                .get("data")
                .and_then(|d| d.get(0))
                .and_then(|t| t.get("coin"))
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let trades = data.as_array().cloned().unwrap_or_default();
            WsEvent::Trades { coin, trades }
        }
        "userFills" => WsEvent::UserFills(data),
        "userEvents" => WsEvent::UserEvents(data),
        _ => {
            if let Some(err) = v.get("error") {
                error!(?err, "ws error");
            }
            WsEvent::Raw(v)
        }
    }
}

fn parse_levels(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|lvl| {
                    let px = lvl.get("px").and_then(|s| s.as_str())?.to_string();
                    let sz = lvl.get("sz").and_then(|s| s.as_str())?.to_string();
                    Some((px, sz))
                })
                .collect()
        })
        .unwrap_or_default()
}
