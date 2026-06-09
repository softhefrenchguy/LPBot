# LPBot Dashboard

Real-time dashboard for LPBot paper/live monitoring.

## Services

- `dashboard-api`: FastAPI backend on port `8000`
- `dashboard`: React/Tailwind/Recharts frontend on port `3000`

## Data Sources

The API reads existing LPBot CSV artifacts:

- `artifacts/paper_trade/daily_checks_log.csv`
- `artifacts/paper_trade/completed_trades.csv`
- `artifacts/markets/market_tracker.csv`
- `artifacts/news/news_log.csv`

The API is schema-tolerant and returns safe defaults when optional columns are missing.

## Endpoints

- `GET /api/status`: latest strategy state
- `GET /api/performance`: paper equity curve from start date
- `GET /api/trades`: completed trades
- `GET /api/markets`: latest 10-asset market tracker
- `GET /api/news`: last 7 news rows
- `WS /ws/live`: status push every 60 seconds plus Binance ETH/BTC live prices

## Local Run

Backend:

```bash
uvicorn scripts.dashboard_api:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd dashboard
npm install
npm run dev
```

Set `VITE_API_BASE=http://localhost:8000` for local frontend development if needed.

## Hetzner / Docker

```bash
docker-compose up -d --build dashboard-api dashboard
```

URLs:

- API: `http://89.167.65.103:8000`
- Frontend: `http://89.167.65.103:3000`

## Notes

- Dark mode is default.
- Frontend auto-refreshes every 60 seconds and also listens to WebSocket updates.
- The dashboard is read-only; it does not place orders.
