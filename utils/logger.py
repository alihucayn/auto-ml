"""
Centralized logging configuration.
"""

import logging


def get_logger(name: str):
    """
    Returns a configured logger instance.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance.
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
