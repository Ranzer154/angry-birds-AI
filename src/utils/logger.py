# src/utils/logger.py

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Creates a named logger with timestamps.
    Use like: logger = get_logger("vision.detector")
    """
    logger = logging.getLogger(name)

    # Don't add duplicate handlers if called twice
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)-20s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    return logger