.PHONY: install test lint eval-offline eval-live serve dashboard screenshots mcp docker
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
# Dashboard at http://localhost:8000/dashboard (offline keyword baseline unless PMRA_LLM_PROVIDER is set)
dashboard:
	PMRA_SOURCES=$${PMRA_SOURCES:-google_news} pmra serve --port 8000
screenshots:
	python scripts/screenshots.py --base http://localhost:8000 --out dashboard-shots
mcp:
	pmra mcp
docker:
	docker compose up --build
