# AI Literaki Helper - Setup Instructions

This guide will help you set up and run the AI Literaki Helper application that automatically assists with playing Literaki (Polish Scrabble) on kurnik.pl.

## Prerequisites

- **Python 3.8 or higher** (recommended: Python 3.9+)
- **Windows, macOS, or Linux** operating system
- **Internet connection** for downloading ChromeDriver and EasyOCR models
- **Google Chrome browser** installed on your system

## Installation Steps

### 1. Install Dependencies

First, install all required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Alternative manual installation:**
```bash
pip install selenium pyautogui opencv-python numpy easyocr pygame matplotlib webdriver-manager
```

### 2. Optional: GPU Acceleration (NVIDIA CUDA)

For faster OCR processing, you can install CUDA-enabled PyTorch (only if you have an NVIDIA GPU with CUDA support):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you don't have CUDA or prefer CPU-only operation, the application will work fine with the default CPU mode.

### 3. Verify Installation

Run the validation script to check if everything is set up correctly:

```bash
python test_setup.py
```

This will test:
- All required dependencies
- Project file availability
- ChromeDriver automatic setup
- EasyOCR initialization
- Pygame functionality
- Project module imports

## Running the Application

### 1. Start the Helper

```bash
python helper.py
```

### 2. What Happens Next

1. **Browser Launch**: Chrome will automatically open and navigate to https://www.kurnik.pl/literaki/
2. **ChromeDriver Setup**: The application will automatically download and configure ChromeDriver (no manual setup needed)
3. **Component Initialization**: EasyOCR, Pygame GUI, and other components will initialize
4. **Game Detection**: The application waits for you to join a game

### 3. Using the Application

1. **Join a Game**: In the opened Chrome browser, manually join a Literaki game
2. **Board Detection**: Once the game starts, the application will automatically detect the 15×15 board
3. **Live Assistance**: The Pygame window will show:
   - Current board state (mirrored from the browser)
   - Your rack tiles
   - Best move suggestions highlighted in green
   - Move details (word, position, score)

### 4. Controls

- **ESC key**: Exit the application
- **Close Pygame window**: Exit the application

## File Structure

Your project directory should contain:

```
AI LITERAKI/
├── helper.py              # Main application file
├── board.py               # Board logic and properties
├── tiles.py               # Tile definitions and scoring
├── dictionary_handler.py  # Dictionary loading and word validation
├── game_gui.py            # Pygame GUI interface
├── requirements.txt       # Python dependencies
├── test_setup.py          # Setup validation script
├── SETUP_INSTRUCTIONS.md  # This file
├── dictionary.txt         # Polish word dictionary (auto-created if missing)
└── chromedriver.exe       # (Optional - auto-managed by webdriver-manager)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` for required packages
**Solution**: 
```bash
pip install -r requirements.txt
```

#### 2. ChromeDriver Issues
**Problem**: ChromeDriver not found or version mismatch
**Solution**: The application uses webdriver-manager for automatic ChromeDriver management. If issues persist:
- Ensure Chrome browser is installed
- Check internet connection
- Try running: `pip install --upgrade webdriver-manager`

#### 3. EasyOCR GPU Errors
**Problem**: GPU initialization fails
**Solution**: The application automatically falls back to CPU mode. For GPU support:
- Ensure NVIDIA GPU with CUDA support
- Install CUDA-enabled PyTorch (see step 2 above)
- Set `OCR_GPU = False` in helper.py for CPU-only mode

#### 4. Pygame Display Issues
**Problem**: Pygame window doesn't appear or crashes
**Solution**: 
- On Linux: Install additional packages: `sudo apt-get install python3-pygame`
- On macOS: Ensure XQuartz is installed for X11 support
- Try running the test script: `python test_setup.py`

#### 5. Dictionary Not Found
**Problem**: dictionary.txt missing
**Solution**: The application will automatically create a minimal dictionary. For better results:
- Download a comprehensive Polish word list
- Save it as `dictionary.txt` in the project directory
- Ensure UTF-8 encoding with one word per line

#### 6. Board Detection Issues
**Problem**: Application doesn't detect the game board
**Solution**:
- Ensure you've joined an active Literaki game
- The board must be visible on screen
- Try adjusting `MIN_RED_CONTOUR_AREA` in helper.py if needed
- Ensure good lighting/contrast on the game board

#### 7. Performance Issues
**Problem**: Slow OCR or move calculation
**Solution**:
- Enable GPU mode for EasyOCR (if available)
- Increase `REFRESH_INTERVAL` in helper.py
- Reduce `OCR_CONFIDENCE_THRESHOLD` if OCR misses letters
- Close other resource-intensive applications

### Configuration Options

You can modify these settings in `helper.py`:

```python
# Update frequency (seconds)
REFRESH_INTERVAL = 1.0

# OCR confidence threshold (0.0 to 1.0)
OCR_CONFIDENCE_THRESHOLD = 0.5

# Minimum red square area for board detection
MIN_RED_CONTOUR_AREA = 100

# Initial wait time after browser launch
INITIAL_WAIT_TIME = 5.0

# EasyOCR GPU mode
OCR_GPU = True  # Set to False for CPU-only
```

## System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: Quad-core 2.5 GHz or better
- **RAM**: 8 GB or more
- **GPU**: NVIDIA GPU with CUDA support (for faster OCR)
- **Storage**: 4 GB free space
- **Python**: 3.9+

## Security Notes

- The application only interacts with kurnik.pl
- No personal data is collected or transmitted
- ChromeDriver is downloaded from official Google repositories
- All processing happens locally on your machine

## Support

If you encounter issues:

1. Run the validation script: `python test_setup.py`
2. Check the console output for specific error messages
3. Ensure all dependencies are correctly installed
4. Verify that Chrome browser is installed and up-to-date

## Legal Notice

This tool is for educational and personal use only. Ensure compliance with kurnik.pl's terms of service when using automated assistance tools.