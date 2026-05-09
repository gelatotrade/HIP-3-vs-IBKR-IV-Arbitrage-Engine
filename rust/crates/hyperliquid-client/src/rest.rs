use crate::signing::{action_connection_id, Wallet};
use crate::types::{L2Book, OrderResponse, Universe, UserState};
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

#[derive(Clone)]
pub struct HyperliquidRest {
    base: String,
    http: Client,
    wallet: Option<Arc<Wallet>>,
    is_mainnet: bool,
}

impl HyperliquidRest {
    pub fn new(base: impl Into<String>, is_mainnet: bool) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(10))
            .pool_idle_timeout(Duration::from_secs(30))
            .tcp_nodelay(true)
            .build()?;
        Ok(Self {
            base: base.into(),
            http,
            wallet: None,
            is_mainnet,
        })
    }

    pub fn with_wallet(mut self, wallet: Wallet) -> Self {
        self.wallet = Some(Arc::new(wallet));
        self
    }

    async fn info(&self, payload: Value) -> Result<Value> {
        let url = format!("{}/info", self.base);
        let resp = self.http.post(url).json(&payload).send().await?;
        let status = resp.status();
        let body: Value = resp.json().await?;
        if !status.is_success() {
            return Err(anyhow!("info {} failed: {}", status, body));
        }
        Ok(body)
    }

    pub async fn meta(&self) -> Result<Universe> {
        let body = self.info(json!({"type": "meta"})).await?;
        Ok(serde_json::from_value(body)?)
    }

    pub async fn meta_and_asset_ctxs(&self) -> Result<Value> {
        self.info(json!({"type": "metaAndAssetCtxs"})).await
    }

    pub async fn l2_book(&self, coin: &str) -> Result<L2Book> {
        let body = self.info(json!({"type": "l2Book", "coin": coin})).await?;
        Ok(serde_json::from_value(body)?)
    }

    pub async fn user_state(&self, user: &str) -> Result<UserState> {
        let body = self
            .info(json!({"type": "clearinghouseState", "user": user}))
            .await?;
        Ok(serde_json::from_value(body)?)
    }

    /// Send a signed action to the exchange endpoint.
    pub async fn exchange(&self, action: Value) -> Result<OrderResponse> {
        let wallet = self
            .wallet
            .as_ref()
            .ok_or_else(|| anyhow!("wallet required for exchange action"))?;

        let nonce = current_nonce_ms();
        let vault = [0u8; 20];
        let connection_id = action_connection_id(&action, &vault, nonce)?;
        let signature = wallet.sign_l1_action(&connection_id, self.is_mainnet)?;

        let payload = json!({
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": null,
        });

        debug!("exchange payload nonce={nonce}");

        let url = format!("{}/exchange", self.base);
        let resp = self
            .http
            .post(url)
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;

        if let Some(status) = resp.get("status").and_then(|v| v.as_str()) {
            if status == "err" {
                warn!(?resp, "exchange returned error");
                return Err(anyhow!("exchange error: {resp}"));
            }
        }

        Ok(OrderResponse {
            status: resp
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            response: resp.get("response").cloned(),
        })
    }

    pub async fn place_order(
        &self,
        asset_id: u32,
        is_buy: bool,
        price: &str,
        size: &str,
        reduce_only: bool,
        tif: &str,
    ) -> Result<OrderResponse> {
        let action = json!({
            "type": "order",
            "orders": [{
                "a": asset_id,
                "b": is_buy,
                "p": price,
                "s": size,
                "r": reduce_only,
                "t": {"limit": {"tif": tif}},
            }],
            "grouping": "na",
        });
        self.exchange(action).await
    }

    pub async fn cancel_order(&self, asset_id: u32, oid: u64) -> Result<OrderResponse> {
        let action = json!({
            "type": "cancel",
            "cancels": [{"a": asset_id, "o": oid}],
        });
        self.exchange(action).await
    }
}

fn current_nonce_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
