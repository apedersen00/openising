"""function for proper logger formatting and coloring"""

import logging


#########################################################################
# Configure Logger
#########################################################################
class ColorFormatter(logging.Formatter):
    """Log formatter that colors the level name (ANSI escape codes) and repeats it on every line."""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET_COLOR = "\033[0m"

    def format(self, record):
        # Apply colors to the levelname
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET_COLOR)
        colored_levelname = f"{log_color}{record.levelname}{self.RESET_COLOR}"
        formatted_message = super().format(record)
        # Split message by line returns and add level to each line
        message_lines = formatted_message.splitlines()
        formatted_lines = [f"{colored_levelname}: {line}" for line in message_lines]
        return "\n".join(formatted_lines)


def log_separator(length=35):
    """
    Print a horizontal separator line at the INFO log level.

    Parameters:
        length (int): The length of the separator line. Default is 35 characters.
    """
    logging.info("-" * length)


def configure_logger(level: int = logging.INFO, logfile: str | None = None):
    """
    Set up the logger with a custom color formatter, and optionally a log file.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(message)s"))
    # Remove the default handler if it exists
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {logfile}")
