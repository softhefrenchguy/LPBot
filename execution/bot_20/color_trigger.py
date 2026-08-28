import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
# color_trigger.py

import cv2
import numpy as np
import pyautogui
import time
import subprocess

print("Watching for red color...")

# Define the region of the screen to watch (x, y, width, height)
REGION = (440, 405, 120, 35)  # slightly bigger

# Define the "red" color range

# Broad magenta/pink detection tuned for OUT OF RANGE button
LOWER_RED = np.array([110, 50, 70])
UPPER_RED = np.array([175, 255, 255])





TRIGGER_DELAY = 30  # seconds to wait after a trigger

def detect_red():
    try:
        screenshot = pyautogui.screenshot(region=REGION)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
        red_ratio = np.sum(mask > 0) / mask.size
        print(f"Red ratio: {red_ratio:.3f}")  # debug
        return red_ratio > 0.1
    except Exception as e:
        print(f"[!] Error taking screenshot: {e}")
        return False


def trigger_action():
    print("[!] Red detected â€” running rebalance.py ...")
    subprocess.run(["python", "rebalance.py"])

if __name__ == "__main__":
    while True:
        # Refresh the browser page
        print("Refreshing page...")
        pyautogui.press('f5')
        time.sleep(10)  # wait for page to load

        # Check for red
        print("Checking screen...")
        if detect_red():
            trigger_action()
            time.sleep(TRIGGER_DELAY)  # wait after trigger

        time.sleep(2)  # wait before next loop
