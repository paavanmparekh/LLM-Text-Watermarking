"""
Shared logger for the Undetectable watermarking scheme.

Creates a timestamped .log file inside
    llm_watermarking/watermarks/undetectable/logs/
on first import.  Both generation.py and detection.py import this module.
"""

import logging
import os
from datetime import datetime

# ------------------------------------------------------------------ #
#  Log directory — same folder as this file, under logs/             #
# ------------------------------------------------------------------ #
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE  = os.path.join(_LOG_DIR, f"undetectable_{_timestamp}.log")

# ------------------------------------------------------------------ #
#  Logger setup                                                        #
# ------------------------------------------------------------------ #
logger = logging.getLogger("undetectable_watermark")
logger.setLevel(logging.DEBUG)

# Avoid adding duplicate handlers if module is re-imported
if not logger.handlers:
    # Logging is currently disabled by user request.
    # To re-enable, uncomment the handlers below.
    logger.addHandler(logging.NullHandler())
    pass

    # --- file handler (DEBUG and above -> everything) ---
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_fh)

    # --- console handler (INFO and above -> important milestones only) ---
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter(fmt="  %(message)s"))
    logger.addHandler(_ch)

# logger.info(f"Log file: {_LOG_FILE}")
