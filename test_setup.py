#!/usr/bin/env python3
"""
Test script to validate AI Literaki Helper setup and dependencies.
Run this before using the main helper.py to ensure everything is working correctly.
"""

import sys
import os
import traceback
from typing import List, Tuple


def test_imports() -> Tuple[bool, List[str]]:
    """Test all required imports."""
    print("Testing imports...")
    missing_deps = []

    # Test core dependencies
    try:
        import cv2

        _ = cv2  # use imported module to satisfy lint
        print("  [OK] OpenCV (cv2) imported successfully")
    except ImportError:
        missing_deps.append("opencv-python")
        print("  [FAIL] OpenCV (cv2) import failed")

    try:
        import numpy as np

        _ = np
        print("  [OK] NumPy imported successfully")
    except ImportError:
        missing_deps.append("numpy")
        print("  [FAIL] NumPy import failed")

    try:
        import pyautogui

        _ = pyautogui
        print("  [OK] PyAutoGUI imported successfully")
    except ImportError:
        missing_deps.append("pyautogui")
        print("  [FAIL] PyAutoGUI import failed")

    try:
        import easyocr

        _ = easyocr
        print("  [OK] EasyOCR imported successfully")
    except ImportError:
        missing_deps.append("easyocr")
        print("  [FAIL] EasyOCR import failed")

    try:
        import pygame

        _ = pygame
        print("  [OK] Pygame imported successfully")
    except ImportError:
        missing_deps.append("pygame")
        print("  [FAIL] Pygame import failed")

    try:
        import selenium

        _ = selenium
        print("  [OK] Selenium imported successfully")
    except ImportError:
        missing_deps.append("selenium")
        print("  [FAIL] Selenium import failed")

    try:
        from webdriver_manager.chrome import ChromeDriverManager

        _ = ChromeDriverManager
        print("  [OK] WebDriver Manager imported successfully")
    except ImportError:
        missing_deps.append("webdriver-manager")
        print("  [FAIL] WebDriver Manager import failed")

    try:
        import matplotlib

        _ = matplotlib
        print("  [OK] Matplotlib imported successfully")
    except ImportError:
        missing_deps.append("matplotlib")
        print("  [FAIL] Matplotlib import failed")

    return len(missing_deps) == 0, missing_deps


def test_project_files() -> bool:
    """Test that all required project files exist."""
    print("\nTesting project files...")
    required_files = [
        "board.py",
        "tiles.py",
        "dictionary_handler.py",
        "game_gui.py",
        "helper.py",
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  [OK] {file} found")
        else:
            print(f"  [FAIL] {file} missing")
            all_exist = False

    # Check for dictionary.txt (will be created if missing)
    if os.path.exists("dictionary.txt"):
        print("  [OK] dictionary.txt found")
    else:
        print("  [WARN] dictionary.txt missing (will be created automatically)")

    return all_exist


def test_chromedriver() -> bool:
    """Test ChromeDriver setup with webdriver-manager."""
    print("\nTesting ChromeDriver setup...")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        # Test webdriver-manager
        print("  • Testing automatic ChromeDriver download...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background for test
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        try:
            driver_service = ChromeService(ChromeDriverManager().install())
            print("  [OK] ChromeDriver automatically downloaded and configured")

            # Test browser creation (but don't navigate anywhere)
            driver = webdriver.Chrome(service=driver_service, options=chrome_options)
            driver.quit()
            print("  [OK] Chrome browser can be launched successfully")
            return True

        except Exception as e:
            print(f"  [FAIL] ChromeDriver test failed: {e}")
            return False

    except Exception as e:
        print(f"  [FAIL] ChromeDriver setup test failed: {e}")
        return False


def test_easyocr_init() -> bool:
    """Test EasyOCR initialization."""
    print("\nTesting EasyOCR initialization...")

    try:
        import easyocr

        # Test CPU mode first
        print("  • Testing EasyOCR CPU mode...")
        _reader = easyocr.Reader(["pl"], gpu=False)
        print("  [OK] EasyOCR CPU mode initialized successfully")

        # Test GPU mode if available
        try:
            print("  • Testing EasyOCR GPU mode...")
            _reader_gpu = easyocr.Reader(["pl"], gpu=True)
            print("  [OK] EasyOCR GPU mode initialized successfully")
        except Exception as e:
            print(f"  [WARN] EasyOCR GPU mode failed (CPU mode will be used): {e}")

        return True

    except Exception as e:
        print(f"  [FAIL] EasyOCR initialization failed: {e}")
        return False


def test_pygame_init() -> bool:
    """Test Pygame initialization."""
    print("\nTesting Pygame initialization...")

    try:
        import pygame

        # Initialize pygame
        pygame.init()
        print("  [OK] Pygame initialized successfully")

        # Test display creation (minimal)
        try:
            _screen = pygame.display.set_mode((100, 100))
            pygame.display.quit()
            print("  [OK] Pygame display can be created")
        except Exception as e:
            print(
                f"  [WARN] Pygame display test failed (may work in full application): {e}"
            )

        pygame.quit()
        return True

    except Exception as e:
        print(f"  [FAIL] Pygame initialization failed: {e}")
        return False


def test_project_imports() -> bool:
    """Test importing project modules."""
    print("\nTesting project module imports...")

    try:
        # Test importing project modules
        from board import create_literaki_board, BOARD_SIZE, MID_INDEX

        _ = (create_literaki_board, BOARD_SIZE, MID_INDEX)

        print("  [OK] board.py imported successfully")

        from tiles import TILE_DEFINITIONS

        _ = TILE_DEFINITIONS

        print("  [OK] tiles.py imported successfully")

        from dictionary_handler import load_dictionary, is_valid_word

        _ = (load_dictionary, is_valid_word)

        print("  [OK] dictionary_handler.py imported successfully")

        from game_gui import LiterakiGUI

        _ = LiterakiGUI

        print("  [OK] game_gui.py imported successfully")

        return True

    except Exception as e:
        print(f"  [FAIL] Project module import failed: {e}")
        traceback.print_exc()
        return False


def run_validation_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("AI LITERAKI HELPER - SETUP VALIDATION")
    print("=" * 60)

    all_passed = True

    # Test 1: Dependencies
    imports_ok, missing_deps = test_imports()
    if not imports_ok:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing_deps)}")
        print("Install them with: pip install " + " ".join(missing_deps))
        all_passed = False

    # Test 2: Project files
    if not test_project_files():
        print("\n[ERROR] Some required project files are missing")
        all_passed = False

    # Test 3: Project imports (only if files exist)
    if imports_ok and test_project_files():
        if not test_project_imports():
            print("\n[ERROR] Project module imports failed")
            all_passed = False

    # Test 4: ChromeDriver
    if imports_ok:
        if not test_chromedriver():
            print("\n[ERROR] ChromeDriver setup failed")
            all_passed = False

    # Test 5: EasyOCR
    if imports_ok:
        if not test_easyocr_init():
            print("\n[ERROR] EasyOCR initialization failed")
            all_passed = False

    # Test 6: Pygame
    if imports_ok:
        if not test_pygame_init():
            print("\n[ERROR] Pygame initialization failed")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED! The setup is ready.")
        print("\nYou can now run: python helper.py")
    else:
        print("[ERROR] SOME TESTS FAILED! Please fix the issues above.")
        print("\nRefer to the setup instructions for help.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    try:
        success = run_validation_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)
