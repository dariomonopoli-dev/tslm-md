# Trajecta stack — common operations.
# All targets assume `docker compose` (v2) is available.

COMPOSE ?= docker compose

.PHONY: help up down restart logs ps build rebuild \
        shell-inference shell-frontend \
        ingest precompute test smoketest clean

help:
	@echo "Targets:"
	@echo "  up               — build + start the stack in the background"
	@echo "  down             — stop and remove containers (keeps volumes)"
	@echo "  restart          — restart inference + frontend"
	@echo "  logs             — tail logs from both services"
	@echo "  ps               — show running services + healthcheck status"
	@echo "  build            — build both images"
	@echo "  rebuild          — build with --no-cache"
	@echo "  shell-inference  — exec a bash shell inside the inference container"
	@echo "  shell-frontend   — exec an sh shell inside the frontend container"
	@echo "  ingest           — run the RAG ingest pipeline inside inference"
	@echo "  precompute       — run scripts/precompute.py (worked examples + failure modes)"
	@echo "  test             — run pytest inside the inference container"
	@echo "  smoketest        — curl /health + /predict + open frontend"
	@echo "  clean            — down + remove the inference_data volume (WARNING: wipes Chroma + cache)"

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart inference frontend

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build

rebuild:
	$(COMPOSE) build --no-cache

shell-inference:
	$(COMPOSE) exec inference bash

shell-frontend:
	$(COMPOSE) exec frontend sh

ingest:
	$(COMPOSE) exec inference python -m rag.ingest

precompute:
	$(COMPOSE) exec inference python scripts/precompute.py

test:
	$(COMPOSE) exec inference pytest -q tests/

smoketest:
	$(COMPOSE) exec inference python scripts/smoketest.py --base http://localhost:8000
	@echo --- frontend ---
	@curl -sI http://localhost:3000/ | head -1 || true
	@echo open http://localhost:3000

clean:
	$(COMPOSE) down -v
	@echo "removed inference_data volume — re-ingest the RAG corpus before /evaluate works again"
