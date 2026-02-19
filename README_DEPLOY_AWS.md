# AWS Deployment (LPBot Paper Trader)

## Overview
This runs the LPBot paper-trade service 24/7 in Docker. The container:
- updates price + volume data
- regenerates exposure
- appends paper log
- emits a JSON heartbeat every 60s

## 1) Install Docker (Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```
Log out and back in to pick up Docker group membership.

## 2) Deploy
```bash
git clone https://github.com/softhefrenchguy/LPBot.git
cd LPBot
cp .env.example .env
# edit .env and set GRAPH_KEY (required)
# optional: set PRICE_START or PRICE_LOOKBACK_DAYS to limit initial backfill

# build + run
sudo docker compose up -d --build
```

## 3) Optional: CloudWatch Logs
Use the awslogs driver (requires AWS credentials on the host or IAM role):
```bash
# set AWS_REGION and AWSLOGS_GROUP in .env
sudo docker compose -f docker-compose.yml -f deploy/awslogs.override.yml up -d --build
```

## 4) Auto-start on reboot
Docker restart policies are enabled in `docker-compose.yml`.
If you want a systemd wrapper:
```bash
sudo cp deploy/lpbot.service /etc/systemd/system/lpbot.service
sudo systemctl daemon-reload
sudo systemctl enable lpbot.service
sudo systemctl start lpbot.service
```

## 5) Health / Monitoring
- Docker healthcheck uses `/tmp/heartbeat.txt` updated every 60s.
- JSON heartbeat logs go to stdout (and CloudWatch if enabled).

Suggested alarms:
- Container stopped / restarting repeatedly.
- Heartbeat missing for N minutes (CloudWatch metric filter on `event=heartbeat`).

## 7) Auto-updating HTML Report (optional)
The compose file includes a `lpbot-report` service that regenerates
`artifacts/paper/report.html` every hour and serves it over HTTP.

Defaults (override in `.env`):
- `REPORT_PORT=8080`
- `REPORT_INTERVAL_SECONDS=3600`

Access it at:
```
http://<EC2_PUBLIC_IP>:8080/report.html
```

Make sure your EC2 Security Group allows inbound TCP on `REPORT_PORT`.

## 6) Artifacts S3 Sync (Daily)
Set these in `.env` to enable daily S3 sync:
- `S3_BUCKET` (required)
- `S3_PREFIX` (optional)
- `S3_SYNC_LOGS=true` (optional)

The service syncs `artifacts/` (and `logs/` if enabled) once per UTC day.
