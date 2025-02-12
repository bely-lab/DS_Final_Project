"""App logger."""

import logging
import os

# Define log file path inside the `src` folder
log_file_path = os.path.join(os.path.dirname(__file__), "main.log")

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="a"), 
        logging.StreamHandler()  
    ],
)
logger = logging.getLogger("MentalHealthLogger")
