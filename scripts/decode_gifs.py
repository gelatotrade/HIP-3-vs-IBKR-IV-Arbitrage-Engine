#!/usr/bin/env python3
"""Decode base64-encoded GIF files (supports chunked .b64.partN files)."""
import base64, glob
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'

for name in ['hip3_arbitrage_strategies_3d', 'hip3_greeks_strategies_3d']:
    gif_file = IMG_DIR / f'{name}.gif'
    
    # Check for chunked files
    parts = sorted(IMG_DIR.glob(f'{name}.gif.b64.part*'))
    if parts:
        b64_data = ''.join(p.read_text() for p in parts)
        data = base64.b64decode(b64_data)
        gif_file.write_bytes(data)
        print(f'Decoded {gif_file.name}: {len(data):,} bytes from {len(parts)} chunks')
        for p in parts:
            p.unlink()
    # Check for single .b64 file
    elif (IMG_DIR / f'{name}.gif.b64').exists():
        b64_file = IMG_DIR / f'{name}.gif.b64'
        data = base64.b64decode(b64_file.read_text().strip())
        gif_file.write_bytes(data)
        print(f'Decoded {gif_file.name}: {len(data):,} bytes')
        b64_file.unlink()
    elif gif_file.exists():
        print(f'{gif_file.name} already exists ({gif_file.stat().st_size:,} bytes)')
    else:
        print(f'WARNING: no data for {name}')
