import pyautogui
import cv2
import numpy as np
import time

print("Move your mouse to the TOP-LEFT corner of the red area. Waiting 5 seconds...")
time.sleep(5)
x1, y1 = pyautogui.position()
print(f"Top-left corner: ({x1}, {y1})")

print("Move your mouse to the BOTTOM-RIGHT corner of the red area. Waiting 5 seconds...")
time.sleep(5)
x2, y2 = pyautogui.position()
print(f"Bottom-right corner: ({x2}, {y2})")

REGION = (x1, y1, x2 - x1, y2 - y1)
print(f"Selected REGION = {REGION}")

# Capture the region
screenshot = pyautogui.screenshot(region=REGION)
frame = np.array(screenshot)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Calculate average color
avg_color_per_row = np.average(frame, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
b, g, r = avg_color
print(f"Average color (BGR): {b:.0f}, {g:.0f}, {r:.0f}")

# Convert to HSV and print average
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
avg_hsv_per_row = np.average(hsv, axis=0)
avg_hsv = np.average(avg_hsv_per_row, axis=0)
h, s, v = avg_hsv
print(f"Average color (HSV): {h:.0f}, {s:.0f}, {v:.0f}")

# Show mask of what counts as "red"
LOWER_RED = np.array([0, 70, 50])
UPPER_RED = np.array([20, 255, 255])
mask1 = cv2.inRange(hsv, LOWER_RED, UPPER_RED)

LOWER_RED2 = np.array([160, 70, 50])
UPPER_RED2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)

mask = mask1 | mask2

cv2.imshow("Detected Red Mask", mask)
cv2.imshow("Original", frame)

print("Press any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
