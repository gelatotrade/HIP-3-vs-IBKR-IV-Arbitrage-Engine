use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Universe {
    pub universe: Vec<AssetMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMeta {
    pub name: String,
    #[serde(rename = "szDecimals")]
    pub sz_decimals: u32,
    #[serde(default)]
    #[serde(rename = "maxLeverage")]
    pub max_leverage: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCtx {
    #[serde(rename = "markPx")]
    pub mark_px: String,
    #[serde(rename = "midPx")]
    pub mid_px: Option<String>,
    pub funding: String,
    #[serde(rename = "openInterest")]
    pub open_interest: String,
    #[serde(rename = "dayNtlVlm")]
    pub day_ntl_vlm: String,
    #[serde(rename = "premium")]
    pub premium: Option<String>,
    #[serde(rename = "oraclePx")]
    pub oracle_px: String,
    #[serde(rename = "impactPxs")]
    pub impact_pxs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Book {
    pub coin: String,
    pub levels: [Vec<L2Level>; 2],
    pub time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Level {
    pub px: String,
    pub sz: String,
    pub n: u32,
}

impl L2Book {
    pub fn best_bid(&self) -> Option<Decimal> {
        self.levels[0].first().and_then(|l| l.px.parse().ok())
    }

    pub fn best_ask(&self) -> Option<Decimal> {
        self.levels[1].first().and_then(|l| l.px.parse().ok())
    }

    pub fn mid(&self) -> Option<Decimal> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        Some((b + a) / Decimal::from(2))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub a: u32,
    pub b: bool,
    pub p: String,
    pub s: String,
    pub r: bool,
    pub t: TifOrTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TifOrTrigger {
    Limit { limit: LimitTif },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitTif {
    pub tif: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderResponse {
    pub status: String,
    pub response: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    #[serde(rename = "marginSummary")]
    pub margin_summary: MarginSummary,
    #[serde(rename = "assetPositions")]
    pub asset_positions: Vec<AssetPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginSummary {
    #[serde(rename = "accountValue")]
    pub account_value: String,
    #[serde(rename = "totalNtlPos")]
    pub total_ntl_pos: String,
    #[serde(rename = "totalRawUsd")]
    pub total_raw_usd: String,
    #[serde(rename = "totalMarginUsed")]
    pub total_margin_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetPosition {
    pub position: PositionData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionData {
    pub coin: String,
    pub szi: String,
    #[serde(rename = "entryPx")]
    pub entry_px: Option<String>,
    #[serde(rename = "positionValue")]
    pub position_value: String,
    #[serde(rename = "unrealizedPnl")]
    pub unrealized_pnl: String,
}
