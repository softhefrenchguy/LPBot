import os
import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord_message(content: str):
    """
    Post a simple text message to Discord.
    If DISCORD_WEBHOOK_URL is missing, it falls back to a no-op.
    """
    if not WEBHOOK_URL:
        print("⚠️ No DISCORD_WEBHOOK_URL set; skipping webhook.")
        return

    data = {"content": content}

    try:
        resp = requests.post(WEBHOOK_URL, json=data, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"⚠️ Webhook failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"⚠️ Error sending webhook: {e}")
