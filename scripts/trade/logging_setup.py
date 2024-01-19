import logging
import os
from logging.handlers import RotatingFileHandler

def init_logging(filename):
    # Configure logging
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Define the maximum log file size (in bytes)
    max_log_file_size = 10 * 1024 * 1024  # 10 MB

    # Create a RotatingFileHandler that truncates the log file at max_log_file_size
    log_handler = RotatingFileHandler(os.path.join(log_dir, filename), maxBytes=max_log_file_size, backupCount=1)
    log_handler.setLevel(logging.INFO)

    # Create a logging formatter
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(log_formatter)

    # Create the root logger and add the RotatingFileHandler
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])