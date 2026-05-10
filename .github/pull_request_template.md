## Summary

<!-- Bullet list of changes; what & why -->

## Test plan

- [ ] Python tests pass: `pytest tests/`
- [ ] Python lints clean: `ruff check scripts/ tests/`
- [ ] Rust tests pass: `cd rust && cargo test --workspace`
- [ ] Rust fmt clean: `cd rust && cargo fmt --check`
- [ ] Rust clippy clean: `cd rust && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Rust release builds: `cd rust && cargo build --release --bin trade-bot`

## Notes

<!-- Anything reviewers should pay extra attention to -->
