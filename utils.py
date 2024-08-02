import pyautogui
import cv2
import pytesseract
import numpy as np
import platform
import os
import logging

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
        logging.info(f"OCR Text: {text}")
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return None

    if 'You have won the coinflip' in text:
        return 'win'
    elif 'You have lost the coinflip' in text:
        return 'lose'
    return None

def send_command(command, delay=0.1, click_coords=None):
    time.sleep(3)
    pyautogui.typewrite('t')
    time.sleep(2)
    for char in command:
        pyautogui.typewrite(char, interval=delay)
    pyautogui.press('enter')
    time.sleep(2)
    if click_coords:
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                pyautogui.click(click_coords[0] + dx, click_coords[1] + dy)
                time.sleep(0.5)

def play_notification_sound():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('play -nq -t alsa synth 0.5 sine 440')
