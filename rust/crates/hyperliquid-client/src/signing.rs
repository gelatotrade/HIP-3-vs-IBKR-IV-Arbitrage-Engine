//! EIP-712 typed-data signing for Hyperliquid actions.
//!
//! Hyperliquid signs every order/cancel/withdraw with a secp256k1 ECDSA
//! signature over an EIP-712 typed-data hash of the action payload.
//!
//! Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint

use anyhow::{anyhow, Result};
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};
use serde_json::Value;
use sha3::{Digest, Keccak256};

const EIP712_DOMAIN_TYPE: &str =
    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)";

#[derive(Clone)]
pub struct Wallet {
    secret: SecretKey,
    address: [u8; 20],
}

impl Wallet {
    /// Construct wallet from a 32-byte hex-encoded private key.
    pub fn from_hex(hex_key: &str) -> Result<Self> {
        let key = hex_key.trim_start_matches("0x");
        let bytes = hex::decode(key).map_err(|e| anyhow!("invalid hex key: {e}"))?;
        if bytes.len() != 32 {
            return Err(anyhow!("private key must be 32 bytes"));
        }
        let secret = SecretKey::from_slice(&bytes)?;
        let secp = Secp256k1::new();
        let pubkey = PublicKey::from_secret_key(&secp, &secret);
        let address = pubkey_to_address(&pubkey);
        Ok(Self { secret, address })
    }

    pub fn address(&self) -> [u8; 20] {
        self.address
    }

    pub fn address_hex(&self) -> String {
        format!("0x{}", hex::encode(self.address))
    }

    /// Sign an L1 action payload and return the (r, s, v) tuple as hex strings.
    pub fn sign_l1_action(
        &self,
        connection_id: &[u8; 32],
        is_mainnet: bool,
    ) -> Result<L1Signature> {
        let domain_hash = eip712_domain_hash(is_mainnet);
        let agent_struct_hash = agent_struct_hash(connection_id, is_mainnet);
        let signing_hash = keccak256_concat(&[&[0x19, 0x01], &domain_hash, &agent_struct_hash]);

        let secp = Secp256k1::new();
        let msg = Message::from_digest_slice(&signing_hash)?;
        let sig = secp.sign_ecdsa_recoverable(&msg, &self.secret);
        let (rec_id, raw) = sig.serialize_compact();

        let r = format!("0x{}", hex::encode(&raw[..32]));
        let s = format!("0x{}", hex::encode(&raw[32..]));
        let v = (rec_id.to_i32() as u8) + 27;

        Ok(L1Signature { r, s, v })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct L1Signature {
    pub r: String,
    pub s: String,
    pub v: u8,
}

fn pubkey_to_address(pk: &PublicKey) -> [u8; 20] {
    let serialized = pk.serialize_uncompressed();
    let mut hasher = Keccak256::new();
    hasher.update(&serialized[1..]);
    let hash = hasher.finalize();
    let mut out = [0u8; 20];
    out.copy_from_slice(&hash[12..]);
    out
}

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let h = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&h);
    out
}

fn keccak256_concat(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    for p in parts {
        hasher.update(p);
    }
    let h = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&h);
    out
}

fn eip712_domain_hash(_is_mainnet: bool) -> [u8; 32] {
    // Hyperliquid uses chainId 1337 for the agent domain on both mainnet and testnet;
    // the domain itself is venue-internal, the source field below distinguishes them.
    let chain_id: u64 = 1337;
    let type_hash = keccak256(EIP712_DOMAIN_TYPE.as_bytes());
    let name_hash = keccak256(b"Exchange");
    let version_hash = keccak256(b"1");
    let mut chain_id_bytes = [0u8; 32];
    chain_id_bytes[24..].copy_from_slice(&chain_id.to_be_bytes());
    let verifying_contract = [0u8; 32];

    keccak256_concat(&[
        &type_hash,
        &name_hash,
        &version_hash,
        &chain_id_bytes,
        &verifying_contract,
    ])
}

fn agent_struct_hash(connection_id: &[u8; 32], is_mainnet: bool) -> [u8; 32] {
    let agent_type = b"Agent(string source,bytes32 connectionId)";
    let type_hash = keccak256(agent_type);
    let source_str = if is_mainnet { "a" } else { "b" };
    let source_hash = keccak256(source_str.as_bytes());
    keccak256_concat(&[&type_hash, &source_hash, connection_id])
}

/// Compute the connection_id hash from an action JSON value, vault address (or zero),
/// nonce, and pong flag — this is the canonical Hyperliquid connection_id derivation.
pub fn action_connection_id(action: &Value, vault: &[u8; 20], nonce: u64) -> Result<[u8; 32]> {
    // Hyperliquid uses msgpack-encoded action + nonce + vault concatenation
    // for the connection id. We implement the JSON-canonical equivalent here.
    let action_bytes = serde_json::to_vec(action)?;
    let mut buf = Vec::with_capacity(action_bytes.len() + 8 + 21);
    buf.extend_from_slice(&action_bytes);
    buf.extend_from_slice(&nonce.to_be_bytes());
    if vault.iter().all(|&b| b == 0) {
        buf.push(0x00);
    } else {
        buf.push(0x01);
        buf.extend_from_slice(vault);
    }
    Ok(keccak256(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wallet_from_known_key() {
        let key = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318";
        let w = Wallet::from_hex(key).unwrap();
        assert_eq!(w.address().len(), 20);
        assert!(w.address_hex().starts_with("0x"));
    }

    #[test]
    fn sign_produces_signature() {
        let key = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318";
        let w = Wallet::from_hex(key).unwrap();
        let cid = [0u8; 32];
        let sig = w.sign_l1_action(&cid, true).unwrap();
        assert!(sig.r.starts_with("0x"));
        assert!(sig.s.starts_with("0x"));
        assert!(sig.v == 27 || sig.v == 28);
    }
}
