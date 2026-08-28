from dotenv import load_dotenv
load_dotenv()  # load .env into this process

import os
import time
import subprocess
import pyautogui
import cv2
import numpy as np
from datetime import datetime, timezone
import json
from colorama import Fore, Style, init
from logger import log_cycle

# =========================
# 🚀 Initialization
# =========================
init(autoreset=True)

# ⬅️ Ensure child processes inherit the same environment (.env values)
ENV = os.environ.copy()

# Your laptop’s region for magenta detection (update via get_coords.py if needed)
REGION = (415, 766, 154, 23)

# HSV bounds for “magenta”
LOWER_MAGENTA = np.array([110, 50, 70])
UPPER_MAGENTA = np.array([175, 255, 255])

# Timings / behavior
TRIGGER_DELAY = 60       # Wait time (seconds) after a full maintenance cycle
STEP_DELAY = 10          # Delay between each script step
LOG_FILE = "history.json"
MAX_RETRIES = 1          # 0 = no retry, 1 = one retry, etc.
DRY_RUN = False
SHOW_WINDOW = True       # Show the detection preview window (pink pixels highlighted)

# =========================
# 🧾 Logging helpers
# =========================
def log_action(action: str):
    # Use timezone-aware UTC to avoid deprecation warnings and keep consistent timestamps
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    safe_action = action.encode("ascii", "ignore").decode()  # avoid weird console chars in file
    print(f"[{timestamp}] {action}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}: {safe_action}\n")
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Could not append to {LOG_FILE}: {e}{Style.RESET_ALL}")

# =========================
# 🎨 Color detection (magenta = out-of-range)
# =========================
def detect_magenta(show_window: bool = SHOW_WINDOW) -> bool:
    screenshot = pyautogui.screenshot(region=REGION)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_MAGENTA, UPPER_MAGENTA)
    ratio = np.sum(mask > 0) / mask.size

    print(f"🎨 Magenta ratio: {ratio:.3f}")

    if show_window:
        preview = frame.copy()
        # Highlight detected magenta pixels
        preview[mask > 0] = [255, 0, 255]
        cv2.imshow("Detection Region", preview)
        cv2.waitKey(1)

    # Threshold for triggering. Tweak if you get false positives/negatives.
    return ratio > 0.1

# =========================
# ⚙️ Script runner (with retries)
# =========================
def run_script(script_name: str, label: str) -> bool:
    for attempt in range(MAX_RETRIES + 1):
        try:
            print(f"{Fore.CYAN}⚙️ Running {label} (Attempt {attempt + 1})...{Style.RESET_ALL}")
            if DRY_RUN:
                print(f"[DRY RUN] Would run: python {script_name}")
                time.sleep(2)
            else:
                # ✅ Pass ENV so sub-scripts see WALLET_ADDRESS / PRIVATE_KEY
                subprocess.run(["python", script_name], check=True, env=ENV)
            log_action(f"✅ {label} completed")
            print(f"{Fore.GREEN}✅ {label} finished successfully.\n{Style.RESET_ALL}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}❌ {label} failed (Attempt {attempt + 1}): {e}{Style.RESET_ALL}")
            log_action(f"❌ {label} failed attempt {attempt + 1}")
            if attempt < MAX_RETRIES:
                print("🔁 Retrying...\n")
                time.sleep(5)
            else:
                print("🚨 Permanently failed.\n")
                return False
        except Exception as e:
            # Catch any unexpected errors (e.g., Python not found, permissions)
            print(f"{Fore.RED}❌ {label} crashed: {e}{Style.RESET_ALL}")
            log_action(f"❌ {label} crashed: {e}")
            return False

# =========================
# 🧠 Data helpers
# =========================
def load_last_cycle() -> dict:
    try:
        with open("last_cycle_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Could not read last_cycle_data.json: {e}{Style.RESET_ALL}")
        return {}

def save_last_cycle(data: dict) -> None:
    try:
        with open("last_cycle_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Could not write last_cycle_data.json: {e}{Style.RESET_ALL}")

# =========================
# 🧩 Main automation loop
# =========================
def main():
    print(f"{Fore.MAGENTA}🚀 LP Automation System Started (Arbitrum){Style.RESET_ALL}")
    print(f"👀 Watching for out-of-range condition... (DRY_RUN={DRY_RUN})\n")

    try:
        while True:
            # Light page refresh to keep the UI in sync (as you had)
            pyautogui.press("f5")
            time.sleep(8)

            # Detect magenta (out of range)
            if detect_magenta(SHOW_WINDOW):
                print(f"\n{Fore.RED}⚠️ Out of range detected — initiating maintenance cycle...{Style.RESET_ALL}\n")
                start_time = time.time()

                # --- Execute full maintenance cycle ---
                if run_script("withdraw_liquidity.py", "Withdraw + Collect"):
                    time.sleep(STEP_DELAY)
                if run_script("rebalance_liquidity.py", "Rebalance Portfolio"):
                    time.sleep(STEP_DELAY)
                if run_script("create_position.py", "Create New Position"):
                    time.sleep(STEP_DELAY)

                end_time = time.time()
                duration_min = round((end_time - start_time) / 60, 1)

                # --- Merge & save cycle data ---
                data = load_last_cycle()
                fees = float(data.get("fees_collected_usd", 0.0))
                pnl = float(data.get("pnl_usd", 0.0))
                range_width = int(data.get("range_width", 20))
                cycle_profit = round(fees + pnl, 2)

                data.update({
                    "cycle_profit_usd": cycle_profit,
                    "cycle_duration_min": duration_min,
                    # Use timezone-aware UTC
                    "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
                })

                save_last_cycle(data)
                try:
                    with open("cycle_history.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(data) + "\n")
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠️ Could not append cycle_history.jsonl: {e}{Style.RESET_ALL}")

                # --- Log formatted summary (your existing pretty logger)
                try:
                    log_cycle(data)
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠️ log_cycle failed: {e}{Style.RESET_ALL}")

                print(
                    f"{Fore.GREEN}💹 Cycle Profit: {cycle_profit:+.2f} USD | "
                    f"Width ±{range_width} ticks | Duration {duration_min} min{Style.RESET_ALL}"
                )
                log_action(f"✅ Cycle complete (range ±{range_width} ticks, {duration_min} min)")

                # --- Run portfolio tracker for overall growth (non-fatal)
                try:
                    subprocess.run(["python", "portfolio_tracker_plus.py"], check=True, env=ENV)
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠️ Portfolio tracker failed: {e}{Style.RESET_ALL}")

                print("✅ Full cycle complete. Waiting before next check...\n")
                time.sleep(TRIGGER_DELAY)

            else:
                print(f"{Fore.GREEN}✅ Position still in range.{Style.RESET_ALL} Checking again soon...\n")
                time.sleep(15)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🛑 Automation stopped by user.{Style.RESET_ALL}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main()

