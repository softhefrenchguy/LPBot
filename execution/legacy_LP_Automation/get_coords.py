import pyautogui
import time

print("Move your mouse to the top-left of the red area. Waiting 5 seconds...")
time.sleep(5)
x1, y1 = pyautogui.position()
print(f"Top-left corner: ({x1}, {y1})")

print("Move your mouse to the bottom-right of the red area. Waiting 5 seconds...")
time.sleep(5)
x2, y2 = pyautogui.position()
print(f"Bottom-right corner: ({x2}, {y2})")

width = x2 - x1
height = y2 - y1
print(f"Suggested REGION = ({x1}, {y1}, {width}, {height})")
