import os

# Paths and URLs
CHROMEDRIVER_PATH = os.environ.get("CHROMEDRIVER_PATH", "chromedriver")
LITERAKI_URL = os.environ.get("LITERAKI_URL", "https://www.kurnik.pl/literaki/")
DICTIONARY_PATH = os.environ.get("DICTIONARY_PATH", "dictionary.txt")

# Timing and refresh
REFRESH_INTERVAL = float(os.environ.get("REFRESH_INTERVAL", "1.0"))
INITIAL_WAIT_TIME = float(os.environ.get("INITIAL_WAIT_TIME", "5.0"))

# OCR and detection settings
OCR_GPU = os.environ.get("OCR_GPU", "True").lower() in ("1", "true", "yes")
OCR_CONFIDENCE_THRESHOLD = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.3"))
OCR_DEBUG_DIR = os.environ.get("OCR_DEBUG_DIR", "ocr_debug")

# Screenshot and cropping
SCREENSHOT_DIR = os.environ.get("SCREENSHOT_DIR", "screenshots")
BOARD_CROP_COORDS = tuple(map(int, os.environ.get("BOARD_CROP_COORDS", "528,146,1080,702").split(',')))
RACK_CROP_COORDS = tuple(map(int, os.environ.get("RACK_CROP_COORDS", "1184,286,1491,328").split(',')))

# Detection thresholds
MIN_RED_CONTOUR_AREA = int(os.environ.get("MIN_RED_CONTOUR_AREA", "100"))
MIN_BOARD_CONFIDENCE = int(os.environ.get("MIN_BOARD_CONFIDENCE", "75"))
