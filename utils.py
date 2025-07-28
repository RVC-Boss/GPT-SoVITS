"""
utils.py - Utility Functions
Exact same utility functions from your working code
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
import logging

logger = logging.getLogger(__name__)

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 image to numpy array - EXACT SAME AS YOUR WORKING CODE
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return None