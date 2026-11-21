.PHONY: install test lint eval-offline eval-live serve mcp docker
install:
	pip install -e ".[dev,all]"
test:
	pytest
lint:
	ruff check src tests && ruff format --check src tests && mypy
eval-offline:
	pmra eval retrieval --out reports && pmra eval forecast --out reports
eval-live:
	pmra eval snapshot --n 30 --horizon 7 --end-date-min 2025-01-01 && pmra eval forecast --out reports
serve:
	pmra serve --port 8080
mcp:
	pmra mcp
docker:
	docker compose up --build
