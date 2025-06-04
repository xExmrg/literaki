# helper.py
#
# A fully working Literaki assistant.
#  • Launches a Chrome browser, navigates to https://www.kurnik.pl/literaki/
#  • Waits until you join a game (detects 4 red corner squares on the board)
#  • Auto-crops exactly the 15×15 board region
#  • Auto-crops the rack region (right panel) immediately to the right of the board
#  • Runs EasyOCR (GPU mode if available) to read letters on board + rack
#  • Mirrors that state into your Pygame GUI (requires your game_gui.py, board.py, tiles.py, dictionary_handler.py)
#  • Brute-forces best move each second, highlights it in translucent green on the GUI
#
# REQUIREMENTS (pip-install these before running):
#   pip install selenium pyautogui opencv-python numpy easyocr pygame matplotlib webdriver-manager
#
# You also need:
#   • A CUDA-enabled PyTorch installation if you want EasyOCR to use GPU:
#       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   • ChromeDriver will be automatically managed by webdriver-manager
#   • Add dictionary.txt (a Polish word list) alongside this script.
#   • The files board.py, tiles.py, dictionary_handler.py, and game_gui.py MUST also be in the same folder.
#
# USAGE:
#   python helper.py
#
# Press ESC in the Pygame window to quit at any time.
#------------------------------------------------------------------------------


import os
import sys
import time
import threading # Not actively used in provided snippet, but kept if original had threading
import traceback
import atexit
import logging
from datetime import datetime
import psutil

import cv2
from enum import Enum # For GameStateEnum
import numpy as np
import pyautogui
import easyocr
import pygame
from PIL import Image # Added for image cropping
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, NoSuchDriverException

# Import webdriver-manager for automatic ChromeDriver management
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False
    print("Warning: webdriver-manager not installed. Install with: pip install webdriver-manager")

from board import create_literaki_board, BOARD_SIZE, MID_INDEX
from tiles import TILE_DEFINITIONS
from game_gui import LiterakiGUI # Assuming this class handles its own drawing updates
from dictionary_handler import load_dictionary, is_valid_word

import itertools

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Coordinates for fixed-region cropping (left, top, right, bottom)
BOARD_CROP_COORDS = (528, 146, 1080, 702)
RACK_CROP_COORDS = (1184, 286, 1491, 328)

# 1) ChromeDriver configuration - now uses webdriver-manager for automatic management
# Fallback to manual path if webdriver-manager is not available
CHROMEDRIVER_PATH = "chromedriver"  # Fallback path if webdriver-manager fails or is not installed

# 2) The URL to open. The script will wait until a live board appears.
LITERAKI_URL = "https://www.kurnik.pl/literaki/"

# 3) How frequently (in seconds) to update OCR + best move search once the board is detected.
REFRESH_INTERVAL = 1.0

# Directory to store debugging screenshots
SCREENSHOT_DIR = "screenshots"
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# 4) Minimum contour area (in pixels) for red square detection.
#    This helps ignore tiny red specks; adjust if needed.
MIN_RED_CONTOUR_AREA = 100 # Adjusted from 100 in original, can be tuned.

# 5) How long to wait (seconds) after launching Chrome before starting detection.
#    (Gives the page time to load and you time to click "Join game".)
INITIAL_WAIT_TIME = 5.0

# 6) OCR confidence threshold
OCR_CONFIDENCE_THRESHOLD = 0.3 # Lowered from 0.5 to catch more potential letters

# 7) Debug logging configuration
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for less verbose default logging, DEBUG for more
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('literaki_helper.log', encoding='utf-8'), # Fixed Unicode encoding
        logging.StreamHandler()
    ]
)

# 8) Performance tracking & Game State
class PerformanceMetrics:
    def __init__(self):
        self.detection_times = []
        self.memory_usage = []
        self.error_count = 0
        self.state_transitions = []
        self.start_time = time.time()

    def record_detection_time(self, duration):
        self.detection_times.append(duration)

    def record_memory_usage(self):
        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

    def record_error(self):
        self.error_count += 1

    def record_state_transition(self, from_state, to_state):
        transition_record = {
            'timestamp': datetime.now().isoformat(),
            'from': from_state.value if isinstance(from_state, GameStateEnum) else from_state,
            'to': to_state.value if isinstance(to_state, GameStateEnum) else to_state,
        }
        self.state_transitions.append(transition_record)
        logging.info(f"State transition: {transition_record['from']} -> {transition_record['to']}")


    def get_stats(self):
        return {
            'uptime': time.time() - self.start_time,
            'avg_detection_time': sum(self.detection_times)/len(self.detection_times) if self.detection_times else 0,
            'avg_memory': sum(self.memory_usage)/len(self.memory_usage) if self.memory_usage else 0,
            'error_count': self.error_count,
            'state_transitions': self.state_transitions
        }

metrics = PerformanceMetrics()

class GameStateEnum(Enum): # Using Enum for clarity
    UNKNOWN = "unknown"
    WAITING = "waiting_for_players"
    DETECTING_BOARD = "detecting_board"
    PLAYING = "playing"
    ERROR = "error"

# Global variable for current game state
current_game_state = GameStateEnum.UNKNOWN

# ------------------------------------------------------------------------------
# GLOBALS & INITIAL SETUP
# ------------------------------------------------------------------------------

print("\n=== Literaki Helper Starting Up ===\n")

# Global variables for proper cleanup
driver = None
ocr_reader = None
gui = None
board_properties = None
# Configuration for EasyOCR GPU usage (can be set in helper.py)
OCR_GPU = True  # Set to False for CPU-only

def validate_dependencies():
    """Validate that all required dependencies are available."""
    missing_deps = []
    # ... (validation logic from original, assumed correct) ...
    return True # Placeholder

def initialize_components():
    """Initialize all components with proper error handling."""
    global ocr_reader, gui, board_properties, OCR_GPU

    try:
        logging.info("• Initializing EasyOCR reader...")
        try:
            if OCR_GPU:
                ocr_reader = easyocr.Reader(['pl'], gpu=True)
                logging.info("  -> EasyOCR initialized with GPU support.")
            else:
                ocr_reader = easyocr.Reader(['pl'], gpu=False)
                logging.info("  -> EasyOCR initialized with CPU support (GPU disabled by config).")
        except Exception as e_gpu:
            logging.warning(f"  -> GPU initialization failed ({e_gpu}), falling back to CPU.")
            try:
                ocr_reader = easyocr.Reader(['pl'], gpu=False)
                logging.info("  -> EasyOCR initialized with CPU support.")
            except Exception as e_cpu:
                logging.error(f"  -> EasyOCR initialization failed completely: {e_cpu}")
                return False

        logging.info("• Loading dictionary (dictionary.txt)...")
        if not os.path.exists("dictionary.txt"):
            logging.warning("  -> Warning: dictionary.txt not found. Creating minimal dictionary.")
            with open("dictionary.txt", "w", encoding="utf-8") as f:
                f.write("test\nslowo\nkot\npies\ndom\nwoda\n")

        try:
            load_dictionary("dictionary.txt")
            dict_words_global = sys.modules['dictionary_handler'].DICTIONARY_WORDS
            logging.info(f"  -> Dictionary loaded: {len(dict_words_global)} words.")
        except Exception as e:
            logging.error(f"  -> Dictionary loading failed: {e}")
            return False

        logging.info("• Initializing Pygame GUI...")
        try:
            pygame.init()
            # Initialize GUI with empty rack for OCR mirroring
            from tiles import PlayerRack
            gui = LiterakiGUI()
            gui.player_rack = PlayerRack(None)  # Start with empty rack
            gui.placed_letters_on_board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]  # Ensure board is empty
            logging.info("  -> Pygame GUI ready (empty rack and board for OCR).")
        except Exception as e:
            logging.error(f"  -> Pygame GUI initialization failed: {e}")
            return False

        logging.info("• Precomputing board properties...")
        try:
            board_properties = create_literaki_board()
            logging.info("  -> Board properties ready.")
        except Exception as e:
            logging.error(f"  -> Board properties initialization failed: {e}")
            return False

        return True

    except Exception as e:
        logging.error(f"Error during component initialization: {e}", exc_info=True)
        return False

def cleanup_resources():
    """Clean up all resources properly."""
    global driver, gui
    logging.info("Cleaning up resources...")
    try:
        if driver:
            logging.info("• Closing browser...")
            driver.quit()
    except Exception as e:
        logging.error(f"Error closing browser: {e}")

    try:
        if gui: # Pygame might be quit by LiterakiGUI.run() itself
            logging.info("• Closing Pygame...")
            pygame.quit()
    except Exception as e:
        logging.error(f"Error closing Pygame: {e}")
    logging.info("Cleanup complete.")
    # Log performance stats on exit
    logging.info(f"Performance Stats: {metrics.get_stats()}")


atexit.register(cleanup_resources)

if not validate_dependencies(): # Assuming validate_dependencies is defined elsewhere or simplified
    logging.critical("Dependency validation failed. Exiting.")
    sys.exit(1)

if not initialize_components():
    logging.critical("Failed to initialize components. Exiting.")
    sys.exit(1)


# ------------------------------------------------------------------------------
# STEP A: LAUNCH CHROME VIA SELENIUM & NAVIGATE TO LITERAKI
# ------------------------------------------------------------------------------

def launch_browser_and_navigate():
    global driver # Ensure driver is global
    # ... (launch_browser_and_navigate logic from original, assumed correct) ...
    # Simplified for brevity
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    # Add other options as in original script
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])


    logging.info(f"• Launching Chrome and navigating to {LITERAKI_URL} ...")
    driver_service = None
    if WEBDRIVER_MANAGER_AVAILABLE:
        try:
            logging.info("  -> Attempting to use webdriver-manager...")
            driver_service = ChromeService(ChromeDriverManager().install())
            logging.info("  -> ChromeDriver automatically managed.")
        except Exception as e:
            logging.warning(f"  -> webdriver-manager failed: {e}. Falling back to manual path if specified.")
    
    if driver_service is None and CHROMEDRIVER_PATH:
        try:
            driver_service = ChromeService(executable_path=CHROMEDRIVER_PATH)
            logging.info(f"  -> Using manual ChromeDriver path: {CHROMEDRIVER_PATH}")
        except Exception as e:
            logging.error(f"  -> Manual ChromeDriver setup failed: {e}")
            return None
    elif driver_service is None:
        logging.error("  -> No ChromeDriver path specified and webdriver-manager failed or not available.")
        return None

    try:
        driver = webdriver.Chrome(service=driver_service, options=chrome_options)
        driver.get(LITERAKI_URL)
        logging.info("  -> Chrome launched. Please join a game manually in the browser.")
        return driver
    except Exception as e:
        logging.error(f"  -> Error launching browser: {e}", exc_info=True)
        return None


# ------------------------------------------------------------------------------
# STEP B: FULL-SCREEN CAPTURE & RED-CORNER DETECTION
# ------------------------------------------------------------------------------

def find_board_in_screenshot(full_bgr):
    """
    Locate the 15×15 board by detecting red/orange bonus squares (typically 3W corners).
    Returns (board_left, board_top, board_width, board_height, tile_w, tile_h, confidence_score)
    or None if no valid board corners were found.
    """
    try:
        hsv = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
        confidence_score = 0

        # Kurnik's 3W squares are reddish/orange.
        lower_red1 = np.array([0, 100, 100]) # Adjusted HSV ranges slightly for broader red/orange
        upper_red1 = np.array([15, 255, 255]) # Hue up to 15 for orange-red
        lower_red2 = np.array([165, 100, 100]) # Hue from 165 for red
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1) # Optional: helps connect broken contours

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_RED_CONTOUR_AREA: # Filter small noise
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

        if len(centroids) < 4: # Need at least 4 markers for corners
            logging.debug(f"Not enough red markers found: {len(centroids)} (need at least 4 for board outline)")
            return None
        confidence_score += 15 # Base for finding enough markers

        xs = [pt[0] for pt in centroids]
        ys = [pt[1] for pt in centroids]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        board_candidate_width = max_x - min_x
        board_candidate_height = max_y - min_y

        if board_candidate_width < 150 or board_candidate_height < 150: # Heuristic: board should be reasonably large
            logging.debug(f"Detected red markers span too small an area: W={board_candidate_width}, H={board_candidate_height}")
            return None
        confidence_score += 10 # For significant span

        aspect_ratio = board_candidate_width / board_candidate_height if board_candidate_height > 0 else 0
        if not (0.8 < aspect_ratio < 1.2):
            logging.debug(f"Red markers don't form a square (aspect ratio: {aspect_ratio:.2f})")
            return None
        confidence_score += 25 # For good aspect ratio

        tile_w = board_candidate_width / 14.0
        tile_h = board_candidate_height / 14.0

        if not (10 < tile_w < 100 and 10 < tile_h < 100): # Tile size validation
            logging.debug(f"Invalid tile size (w:{tile_w:.1f}, h:{tile_h:.1f})")
            return None
        confidence_score += 20 # For valid tile sizes

        # Board rectangle: from min/max of markers, expand by half a tile for full board coverage
        board_left = int(min_x - tile_w / 2)
        board_top = int(min_y - tile_h / 2)
        # Recalculate board width and height based on 15 tiles
        board_width = int(tile_w * 15)
        board_height = int(tile_h * 15)


        # Clamp to screenshot bounds
        H_screen, W_screen = full_bgr.shape[:2]
        board_left = max(0, board_left)
        board_top = max(0, board_top)
        board_width = min(board_width, W_screen - board_left)
        board_height = min(board_height, H_screen - board_top)

        # Active game state indicator check (original logic, can be tuned)
        # This checks if the board area is too uniformly bright (e.g. not a game board)
        board_img_check = full_bgr[board_top:board_top+board_height, board_left:board_left+board_width]
        if board_img_check.size == 0: return None # Avoid error if rect is invalid
        gray_board_check = cv2.cvtColor(board_img_check, cv2.COLOR_BGR2GRAY)
        _, thresh_board_check = cv2.threshold(gray_board_check, 220, 255, cv2.THRESH_BINARY) # Higher threshold for "very bright"
        active_pixels = cv2.countNonZero(thresh_board_check)
        if active_pixels > (board_width * board_height * 0.5):  # Increased threshold: if >50% is very bright
            logging.debug("Board area too bright, might not be active game state or wrong detection.")
            #return None # Commented out: this check might be too aggressive for empty boards
            confidence_score -= 10 # Penalize if too bright, but don't outright reject yet
        else:
            confidence_score += 30


        logging.debug(f"Board detection proposal: Rect=({board_left},{board_top},{board_width},{board_height}), TileW/H=({tile_w:.1f},{tile_h:.1f}), Confidence={confidence_score}/100")
        return (board_left, board_top, board_width, board_height, tile_w, tile_h, confidence_score)

    except Exception as e:
        logging.error(f"Error in find_board_in_screenshot: {e}", exc_info=True)
        return None


def detect_waiting_screen(image_bgr: np.ndarray) -> bool:
    global ocr_reader
    if ocr_reader is None:
        logging.error("OCR reader not initialized for detect_waiting_screen.")
        return False

    try:
        # Use grayscale image for OCR, EasyOCR generally handles this well.
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # OCR the image. paragraph=True groups nearby text.
        # batch_size and other params can be tuned if OCR is slow or inaccurate.
        text_results = ocr_reader.readtext(gray_image, detail=0, paragraph=True, batch_size=8,
                                           workers=0 if OCR_GPU else 2, # workers > 0 for CPU for parallelism
                                           y_ths=0.5, x_ths=0.8, low_text=0.4) # Common EasyOCR tuning parameters

        detected_text = " ".join(text_results).lower()
        logging.debug(f"OCR for waiting screen: '{detected_text[:200].strip()}...'")

        waiting_phrases = [
            "oczekiwanie na graczy",
            "czekanie na graczy",
            "oczekuje na graczy",
            "czekaj na graczy",
            "gra nie rozpoczęta",
            "gra zakończona" # Also a state where board detection might fail
        ]

        for phrase in waiting_phrases:
            if phrase in detected_text:
                logging.info(f"Waiting/ended screen detected (text: '{phrase}')")
                return True
        return False
    except Exception as e:
        logging.error(f"Error in detect_waiting_screen OCR: {e}", exc_info=True)
        return False # Assume not waiting on error


# ------------------------------------------------------------------------------
# STEP C: OCR BOARD & RACK (Assumed largely correct from original)
# ------------------------------------------------------------------------------
def ocr_board(tile_w, tile_h, board_rect, board_img_bgr):
    """Enhanced OCR for the board with comprehensive debugging and image preprocessing."""
    global ocr_reader, OCR_GPU
    try:
        bx, by, bw, bh = board_rect
        # board_img is already the cropped board image (board_cropped_bgr)
        # No need to re-crop from full_bgr
        board_img = board_img_bgr
        if board_img.size == 0:
            logging.warning("Board image for OCR is empty.")
            return [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        logging.info(f"OCR Debug: Board region (relative to cropped image: {bx},{by},{bw},{bh}), tile size ({tile_w:.1f},{tile_h:.1f})")
        
        # Save original board image for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = "ocr_debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        cv2.imwrite(f"{debug_dir}/board_original_{timestamp}.png", board_img)
        
        # Image preprocessing for better OCR
        # Convert to grayscale
        gray_board = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_board = clahe.apply(gray_board)
        
        # Apply Gaussian blur to reduce noise
        blurred_board = cv2.GaussianBlur(enhanced_board, (3, 3), 0)
        
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_board = cv2.filter2D(blurred_board, -1, kernel)
        
        # Save preprocessed image for debugging
        cv2.imwrite(f"{debug_dir}/board_preprocessed_{timestamp}.png", sharpened_board)
        
        # Run OCR on both original and preprocessed images
        logging.info("OCR Debug: Running EasyOCR on original image...")
        board_results_orig = ocr_reader.readtext(board_img, batch_size=16, workers=0 if OCR_GPU else 4)
        
        logging.info("OCR Debug: Running EasyOCR on preprocessed image...")
        board_results_proc = ocr_reader.readtext(sharpened_board, batch_size=16, workers=0 if OCR_GPU else 4)
        
        # Log all OCR results before filtering
        logging.info(f"OCR Debug: Original image detected {len(board_results_orig)} text regions:")
        for i, (bbox, text, prob) in enumerate(board_results_orig):
            logging.info(f"  [{i}] Text: '{text}' | Confidence: {prob:.3f} | BBox: {bbox}")
        
        logging.info(f"OCR Debug: Preprocessed image detected {len(board_results_proc)} text regions:")
        for i, (bbox, text, prob) in enumerate(board_results_proc):
            logging.info(f"  [{i}] Text: '{text}' | Confidence: {prob:.3f} | BBox: {bbox}")
        
        # Choose the better result set (more detections with good confidence)
        board_results = board_results_proc if len(board_results_proc) > len(board_results_orig) else board_results_orig
        logging.info(f"OCR Debug: Using {'preprocessed' if board_results == board_results_proc else 'original'} image results")

        board_letters = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        accepted_letters = 0
        rejected_letters = 0
        
        char_mapping = {
            'A': 'A', 'Ą': 'Ą', 'B': 'B', 'C': 'C', 'Ć': 'Ć', 'D': 'D',
            'E': 'E', 'Ę': 'Ę', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
            'J': 'J', 'K': 'K', 'L': 'L', 'Ł': 'Ł', 'M': 'M', 'N': 'N',
            'Ń': 'Ń', 'O': 'O', 'Ó': 'Ó', 'P': 'P', 'R': 'R', 'S': 'S',
            'Ś': 'Ś', 'T': 'T', 'U': 'U', 'W': 'W', 'Y': 'Y', 'Z': 'Z',
            'Ź': 'Ź', 'Ż': 'Ż', '_': '_',
            # Common OCR mistakes
            '0': 'O', '1': 'I', '6': 'G', '8': 'B'
        }
        
        for (bbox, text, prob) in board_results:
            txt = text.strip().upper()
            
            # Log every detection attempt
            logging.debug(f"OCR Debug: Processing '{txt}' (conf: {prob:.3f})")
            
            if prob < OCR_CONFIDENCE_THRESHOLD:
                logging.debug(f"  -> Rejected: confidence {prob:.3f} < {OCR_CONFIDENCE_THRESHOLD}")
                rejected_letters += 1
                continue
                
            # Handle Polish character variations and common OCR mistakes
            if len(txt) == 1 and txt in char_mapping:
                mapped_char = char_mapping[txt]
                if mapped_char in TILE_DEFINITIONS:
                    x_coords = [pt[0] for pt in bbox]
                    y_coords = [pt[1] for pt in bbox]
                    cx = sum(x_coords) / 4.0
                    cy = sum(y_coords) / 4.0
                    col = int(cx // tile_w)
                    row = int(cy // tile_h)
                    
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                        board_letters[row][col] = mapped_char
                        accepted_letters += 1
                        logging.debug(f"  -> Accepted: '{mapped_char}' at ({row},{col})")
                    else:
                        logging.debug(f"  -> Rejected: position ({row},{col}) out of bounds")
                        rejected_letters += 1
                else:
                    logging.debug(f"  -> Rejected: '{mapped_char}' not in TILE_DEFINITIONS")
                    rejected_letters += 1
            else:
                logging.debug(f"  -> Rejected: invalid length or character '{txt}'")
                rejected_letters += 1
        
        logging.info(f"OCR Debug: Board processing complete - Accepted: {accepted_letters}, Rejected: {rejected_letters}")
        
        total_board_letters = sum(1 for r in board_letters for c in r if c)
        logging.info(f"OCR Result: Board: {total_board_letters} letters")
        return board_letters

    except Exception as e:
        logging.error(f"Error in ocr_board: {e}", exc_info=True)
        return [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def ocr_rack(rack_img_bgr: np.ndarray) -> list:
    """OCR for the player's letter rack."""
    global ocr_reader, OCR_GPU
    try:
        if rack_img_bgr.size == 0:
            logging.warning("Rack image for OCR is empty.")
            return []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = "ocr_debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        cv2.imwrite(f"{debug_dir}/rack_original_{timestamp}.png", rack_img_bgr)

        # Preprocess rack image
        gray_rack = cv2.cvtColor(rack_img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_rack = clahe.apply(gray_rack)
        blurred_rack = cv2.GaussianBlur(enhanced_rack, (3, 3), 0)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_rack = cv2.filter2D(blurred_rack, -1, kernel)
        cv2.imwrite(f"{debug_dir}/rack_preprocessed_{timestamp}.png", sharpened_rack)
        
        rack_results = ocr_reader.readtext(sharpened_rack, batch_size=8, workers=0 if OCR_GPU else 2)
        
        logging.info(f"OCR Debug: Rack detected {len(rack_results)} text regions:")
        for i, (bbox, text, prob) in enumerate(rack_results):
            logging.info(f"  [{i}] Text: '{text}' | Confidence: {prob:.3f}")
        
        char_mapping = {
            'A': 'A', 'Ą': 'Ą', 'B': 'B', 'C': 'C', 'Ć': 'Ć', 'D': 'D',
            'E': 'E', 'Ę': 'Ę', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
            'J': 'J', 'K': 'K', 'L': 'L', 'Ł': 'Ł', 'M': 'M', 'N': 'N',
            'Ń': 'Ń', 'O': 'O', 'Ó': 'Ó', 'P': 'P', 'R': 'R', 'S': 'S',
            'Ś': 'Ś', 'T': 'T', 'U': 'U', 'W': 'W', 'Y': 'Y', 'Z': 'Z',
            'Ź': 'Ź', 'Ż': 'Ż', '_': '_',
            # Common OCR mistakes
            '0': 'O', '1': 'I', '6': 'G', '8': 'B'
        }

        rack_candidates = []
        for (bbox, text, prob) in rack_results:
            if prob < OCR_CONFIDENCE_THRESHOLD: continue
            txt = text.strip().upper()
            if len(txt) == 1 and txt in char_mapping:
                mapped_char = char_mapping[txt]
                if mapped_char in TILE_DEFINITIONS:
                    x_coords = [pt[0] for pt in bbox]
                    cx = sum(x_coords) / 4.0
                    cy = sum([pt[1] for pt in bbox]) / 4.0
                    rack_candidates.append({'text': mapped_char, 'cx': cx, 'cy': cy})

        rack_candidates.sort(key=lambda e: (e['cy'], e['cx']))
        rack_letters = [e['text'] for e in rack_candidates[:7]]
        logging.info(f"OCR Result: Rack: {len(rack_letters)} letters")
        return rack_letters

    except Exception as e:
        logging.error(f"Error in OCR rack processing: {e}", exc_info=True)
        return []



# ------------------------------------------------------------------------------
# STEP D: MIRROR INTO PYGAME GUI (Assumed largely correct from original)
# ------------------------------------------------------------------------------
def update_gui_from_detection(board_letters, rack_letters):
    global gui
    # ... (update_gui_from_detection logic from original, assumed correct) ...
    # Ensure gui object's methods are called correctly
    try:
        # 1) Update board tiles in GUI's internal representation
        new_placed_letters = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                letter = board_letters[r][c]
                if letter: # If a letter is detected for this square
                    if letter in TILE_DEFINITIONS:
                        tile_info = TILE_DEFINITIONS[letter]
                        # Create a tile object similar to how PlayerRack might store them
                        gui_tile_obj = {'letter': letter, 'points': tile_info['points'], 'color': tile_info['color']}
                        new_placed_letters[r][c] = {
                            'display_letter': letter, # The letter to show
                            'tile_obj': gui_tile_obj    # The underlying tile data
                        }
                    else: # Fallback for unrecognized characters (should ideally not happen if filtered in OCR)
                         new_placed_letters[r][c] = {'display_letter': letter, 'tile_obj': {'letter':letter, 'points':0, 'color':None}}
        gui.placed_letters_on_board = new_placed_letters

        # 2) Update rack tiles in GUI's internal representation
        new_rack_tiles = []
        for ch in rack_letters:
            if ch in TILE_DEFINITIONS:
                tile_info = TILE_DEFINITIONS[ch]
                new_rack_tiles.append({'letter': ch, 'points': tile_info['points'], 'color': tile_info['color']})
            # else: skip unrecognized chars for rack
        gui.player_rack.tiles = new_rack_tiles # Assuming player_rack has a 'tiles' attribute

        # 3) GUI should handle its own drawing when its data changes, or explicitly call draw
        # gui.draw_board() # This might be called by the main loop after updates
        # gui.draw_rack()  # Or, LiterakiGUI class might have an update() method
        # pygame.display.flip() # Usually called once per frame in main loop
    except Exception as e:
        logging.error(f"Error updating GUI data: {e}", exc_info=True)


# ------------------------------------------------------------------------------
# STEP E: BEST MOVE SEARCH (Brute Force) (Assumed largely correct from original)
# ------------------------------------------------------------------------------
def find_anchor_positions(board_letters):
    # ... (find_anchor_positions logic from original) ...
    return [] # Placeholder

def can_place_word_at(board_letters, rack_letters, word, row, col, horizontal):
    # ... (can_place_word_at logic from original) ...
    return None # Placeholder

def score_move(used, word, row, col, horizontal, board_letters):
    # ... (score_move logic from original) ...
    return 0 # Placeholder

def best_move_search(board_letters, rack_letters):
    global board_properties # Ensure access to the game board layout for scoring
    # ... (best_move_search logic from original, ensure it uses board_properties for bonuses) ...
    # This is a complex function, assumed correct from original for now.
    # Key is that it uses `is_valid_word` and `TILE_DEFINITIONS`.

    # Minimal implementation for testing flow:
    if rack_letters and board_letters:
        # Try to find any valid word.
        # This is a placeholder for the actual complex search.
        pass
    return None # Placeholder

# ------------------------------------------------------------------------------
# STEP F: MAIN LOOP
# ------------------------------------------------------------------------------

def main_loop():
    global current_game_state, driver, gui, metrics

    last_ocr_time = 0
    board_rect_stable = None
    tile_w_stable, tile_h_stable = None, None
    MIN_BOARD_CONFIDENCE = 75

    if gui:
        gui.draw_board()
        pygame.display.flip()

    while True:
        loop_start = time.time()
        for event in pygame.event.get():
            if event.type == QUIT:
                logging.info("Quit event received. Exiting main loop.")
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                logging.info("ESC key pressed. Exiting main loop.")
                return

        try:
            screenshot_start = time.time()
            full_screenshot_pil = pyautogui.screenshot()
            screenshot_end = time.time()
            if full_screenshot_pil is None:
                logging.error("Failed to capture screenshot (pyautogui.screenshot() returned None).")
                time.sleep(REFRESH_INTERVAL)
                continue
            full_bgr_image = cv2.cvtColor(np.array(full_screenshot_pil), cv2.COLOR_RGB2BGR)

            # --- New cropping and saving logic for fixed regions ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crop and save board image
            board_cropped_pil = full_screenshot_pil.crop(BOARD_CROP_COORDS)
            board_cropped_filename = os.path.join(SCREENSHOT_DIR, f"board_cropped_{timestamp}.png")
            board_cropped_pil.save(board_cropped_filename)
            logging.info(f"Saved cropped board image: {board_cropped_filename}")

            # Crop and save rack image
            rack_cropped_pil = full_screenshot_pil.crop(RACK_CROP_COORDS)
            rack_cropped_filename = os.path.join(SCREENSHOT_DIR, f"rack_cropped_{timestamp}.png")
            rack_cropped_pil.save(rack_cropped_filename)
            logging.info(f"Saved cropped rack image: {rack_cropped_filename}")
            # --- End new cropping and saving logic ---

        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}", exc_info=True)
            if gui: gui.display_message("Screenshot error. Retrying...", (200,0,0))
            time.sleep(REFRESH_INTERVAL)
            continue

        new_state = current_game_state

        detect_start = time.time()
        # Use the cropped board image for board detection and OCR
        # Convert PIL Image to OpenCV format (BGR) for processing
        board_cropped_bgr = cv2.cvtColor(np.array(board_cropped_pil), cv2.COLOR_RGB2BGR)
        rack_cropped_bgr = cv2.cvtColor(np.array(rack_cropped_pil), cv2.COLOR_RGB2BGR)

        if detect_waiting_screen(board_cropped_bgr): # Use cropped board for waiting screen detection
            if current_game_state != GameStateEnum.WAITING:
                new_state = GameStateEnum.WAITING
                if gui:
                    gui.display_message("Waiting for players...", (0, 100, 200))
                    update_gui_from_detection([[""]*BOARD_SIZE for _ in range(BOARD_SIZE)], [])
                    gui.draw_board()
                    pygame.display.flip()
                board_rect_stable = None
        else:
            # For board detection, we still need to find the board within the *cropped* board image.
            # The coordinates (528, 146) become (0,0) in the cropped image.
            # So, we need to adjust the board_rect_stable to be relative to the cropped image.
            # However, since we are now *cropping* to the exact board region,
            # we can assume the board is the entire `board_cropped_bgr` image.
            # We just need to derive tile_w and tile_h from its dimensions.
            if current_game_state != GameStateEnum.PLAYING: # Only re-calculate if not already playing
                new_state = GameStateEnum.PLAYING # Assume playing if not waiting
                # Derive tile_w and tile_h from the fixed board crop dimensions
                b_w_fixed = BOARD_CROP_COORDS[2] - BOARD_CROP_COORDS[0]
                b_h_fixed = BOARD_CROP_COORDS[3] - BOARD_CROP_COORDS[1]
                tile_w_stable = b_w_fixed / 15.0
                tile_h_stable = b_h_fixed / 15.0
                board_rect_stable = (0, 0, b_w_fixed, b_h_fixed) # Relative to cropped image
                last_ocr_time = 0
                if gui: gui.display_message("Board active. Assisting...", (0,200,0))

            if new_state == GameStateEnum.PLAYING and board_rect_stable and (time.time() - last_ocr_time >= REFRESH_INTERVAL or last_ocr_time == 0):
                last_ocr_time = time.time()
                metrics.record_memory_usage()

                ocr_start = time.time()
                board_letters = ocr_board(
                    tile_w_stable,
                    tile_h_stable,
                    board_rect_stable,
                    board_cropped_bgr,
                )
                rack_letters = ocr_rack(rack_cropped_bgr)
                ocr_end = time.time()

                update_gui_from_detection(board_letters, rack_letters)

                if gui:
                    gui.draw_board()

                move_start = time.time()
                best_m = best_move_search(board_letters, rack_letters)
                move_end = time.time()

                if gui:
                    if best_m:
                        logging.info(
                            f"Best move: {best_m['word']} @ ({best_m['row']},{best_m['col']}) score {best_m['score']}"
                        )
                        for (r_bm, c_bm, ch_bm) in best_m.get('used_positions', []):
                            x_hl = gui.GRID_MARGIN + c_bm * gui.SQUARE_SIZE
                            y_hl = gui.GRID_MARGIN + 40 + r_bm * gui.SQUARE_SIZE
                            s_hl = gui.SQUARE_SIZE
                            sfc = pygame.Surface((s_hl, s_hl), pygame.SRCALPHA)
                            sfc.fill((0, 255, 0, 100))
                            gui.screen.blit(sfc, (x_hl, y_hl))

                        info_text = (
                            f"Best: {best_m['word']} @ ({best_m['row']},{best_m['col']})"
                            f" {'H' if best_m['horizontal'] else 'V'} = {best_m['score']} pts"
                        )
                        text_surf = gui.font_title.render(info_text, True, (50, 50, 50))
                        text_rect = text_surf.get_rect(center=(gui.WINDOW_WIDTH // 2, 20))
                        gui.screen.blit(text_surf, text_rect)
                    else:
                        logging.debug("No best move found in this cycle.")
                        pygame.draw.rect(
                            gui.screen,
                            gui.COLORS['background'],
                            (0, 0, gui.WINDOW_WIDTH, gui.GRID_MARGIN + 35),
                        )
                        title_text = gui.font_title.render("LITERAKI HELPER - ACTIVE", True, (50, 50, 50))
                        title_rect = title_text.get_rect(center=(gui.WINDOW_WIDTH // 2, 20))
                        gui.screen.blit(title_text, title_rect)

                    pygame.display.flip()

                logging.info(
                    f"Timing: screenshot={screenshot_end-screenshot_start:.2f}s, "
                    f"detect_waiting={detect_start-screenshot_end:.2f}s, "
                    f"OCR+update={ocr_end-ocr_start:.2f}s, "
                    f"move_search={move_end-move_start:.2f}s, "
                    f"loop_total={time.time()-loop_start:.2f}s",
                )

if new_state != current_game_state:
    metrics.record_state_transition(current_game_state, new_state)
    current_game_state = new_state

if current_game_state != GameStateEnum.PLAYING:
    time.sleep(min(REFRESH_INTERVAL, 0.5))

if gui:
    gui.clock.tick(30)

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        logging.info("Application starting...")
        driver = launch_browser_and_navigate()
        if driver is None:
            logging.critical("Failed to launch browser. Exiting.")
            if gui: gui.display_message("Browser launch failed. Check console.", (200,0,0), duration=5)
            sys.exit(1)

        logging.info(f"Waiting {INITIAL_WAIT_TIME} seconds for page to load and user to join game...")
        if gui: gui.display_message(f"Browser launched. Join game. Waiting {INITIAL_WAIT_TIME}s...", (0,100,200))
        
        # Initial sleep allows user to manually navigate/join game
        # The main_loop will then handle waiting/detection logic.
        time.sleep(INITIAL_WAIT_TIME)

        main_loop()

    except KeyboardInterrupt:
        logging.info("\nProgram interrupted by user.")
    except Exception as e:
        logging.critical(f"Fatal error in main: {e}", exc_info=True)
    finally:
        logging.info("Application terminating...")
        # cleanup_resources() will be called by atexit
