.PHONY: setup install-models docker-build docker-up docker-down \
        docker-logs docker-index docker-test docker-restart clean

# ─── Host setup ──────────────────────────────────────────────────────────────

setup: install-models
	@echo "Models downloaded. Run 'make docker-build' then 'make docker-up'."

install-models:
	bash scripts/download_model.sh

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart: docker-down docker-up

# ─── One-off commands inside the app container ───────────────────────────────

docker-index:
	docker compose exec app python scripts/index_resume.py

docker-test:
	docker compose exec app python scripts/test_pipeline.py

# ─── Systemd for Docker Compose (auto-start on boot) ─────────────────────────

deploy-systemd:
	cp deploy/docker-compose.service /etc/systemd/system/
	systemctl daemon-reload
	systemctl enable docker-compose

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	rm -rf models
	docker compose down --volumes --rmi all

docker-prune:
	docker system prune -af --volumes
