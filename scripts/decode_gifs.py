#!/usr/bin/env python3
"""Decode base64-encoded GIF files."""
import base64
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'

for name in ['hip3_arbitrage_strategies_3d', 'hip3_greeks_strategies_3d']:
    b64_file = IMG_DIR / f'{name}.gif.b64'
    gif_file = IMG_DIR / f'{name}.gif'
    if b64_file.exists():
        data = base64.b64decode(b64_file.read_text().strip())
        gif_file.write_bytes(data)
        print(f'Decoded {gif_file.name}: {len(data):,} bytes')
        b64_file.unlink()
    elif gif_file.exists():
        print(f'{gif_file.name} already exists ({gif_file.stat().st_size:,} bytes)')
    else:
        print(f'WARNING: {b64_file.name} not found')
