# utils.py

import pyautogui
import cv2
import pytesseract
import numpy as np

def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    enhanced_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.medianBlur(enhanced_image, 3)
    return enhanced_image

def detect_outcome(screenshot):
    try:
        _, thresh = cv2.threshold(screenshot, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        print(f"OCR Text: {text}")
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

    if 'You have won the coinflip' in text:
        return 'win'
    elif 'You have lost the coinflip' in text:
        return 'lose'
    return None
