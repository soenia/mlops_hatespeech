from loguru import logger
import sys
import os

# Create log directory if it doesn't exist
log_path = "reports/logs"
os.makedirs(log_path, exist_ok=True)

logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Add a new logger with WARNING level

logger.add(f"{log_path}/my_log.log", level="DEBUG", rotation="100 MB")
