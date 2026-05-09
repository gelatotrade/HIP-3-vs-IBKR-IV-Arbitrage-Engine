.PHONY: all install dev test lint format typecheck fetch backtest viz clean docker-build docker-run

all: install test lint

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check scripts/ tests/

format:
	ruff format scripts/ tests/

typecheck:
	mypy scripts/ --ignore-missing-imports

fetch:
	python3 scripts/fetch_all_hip3.py
	python3 scripts/fetch_ibkr_options.py

backtest:
	python3 scripts/run_all_hip3_premium.py

viz:
	python3 scripts/generate_hip3_visualizations.py
	python3 scripts/generate_arbitrage_surfaces.py
	python3 scripts/generate_greeks_strategies_surfaces.py
	python3 scripts/generate_arbitrage_setups_animated.py

clean:
	rm -rf __pycache__ scripts/__pycache__ .pytest_cache .mypy_cache .ruff_cache

docker-build:
	docker build -t hip3-arb-engine .

docker-run:
	docker run --rm hip3-arb-engine
