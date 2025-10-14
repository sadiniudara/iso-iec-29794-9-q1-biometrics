"""
ISO/IEC 29794-9 Quality Component 1 (Effective Area) Implementation
Finger Vascular Biometrics - Second Phalanx

This module implements the mandatory ISO calculation for Q1 (Effective Area)
according to ISO/IEC 29794-9 Clause 5.2.1.

Requirements:
- Coefficient (Sc): 20000 pixels for Finger
- Veto Logic: If Foreground Region (R) is invalid (zero pixels), Q1 = 0
- Formula: Q1 = MIN(100, ROUND(Sunoccluded / Sc * 100))
"""

import cv2
import numpy as np
from typing import Tuple


def calculate_q1(image_path: str) -> Tuple[int, int]:
    """
    Calculate Q1 (Effective Area) quality metric for finger vascular biometrics.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        Tuple[int, int]: (Q1_score, S_unoccluded_pixel_count)
            - Q1_score: Final integer Q1 score (0-100)
            - S_unoccluded_pixel_count: Raw pixel count of unoccluded foreground region
    
    ISO Requirements:
    1. Coefficient (Sc) = 20000 pixels (for Finger)
    2. Veto Logic: If Foreground Region (R) is invalid (zero pixels), Q1 = 0
    3. Formula: Q1 = MIN(100, ROUND(Sunoccluded / Sc * 100))
    """
    
    # ISO Coefficient for Finger (Clause 5.2.1)
    Sc = 20000  # pixels
    
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Foreground Extraction using basic thresholding and contour finding
        # Convert to binary using Otsu's method for automatic threshold selection
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours to identify the foreground region
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours found - invalid foreground region
            return 0, 0
        
        # Find the largest contour (main foreground region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask for the largest contour
        mask = np.zeros_like(binary_mask)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # Calculate S_unoccluded (pixel count of unoccluded foreground region)
        S_unoccluded = np.sum(mask == 255)
        
        # Veto Logic: If Foreground Region (R) is invalid (zero pixels), Q1 = 0
        if S_unoccluded == 0:
            return 0, 0
        
        # ISO Formula: Q1 = MIN(100, ROUND(Sunoccluded / Sc * 100))
        q1_raw = S_unoccluded / Sc * 100
        Q1_score = min(100, round(q1_raw))
        
        return Q1_score, S_unoccluded
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 0, 0


def create_test_image(width: int, height: int, foreground_pixels: int, output_path: str) -> None:
    """
    Create a test image with a specified number of foreground pixels.
    
    Args:
        width (int): Image width
        height (int): Image height  
        foreground_pixels (int): Number of foreground pixels to create
        output_path (str): Path to save the test image
    """
    
    # Create black background
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate how many pixels we can actually fit
    max_pixels = width * height
    actual_pixels = min(foreground_pixels, max_pixels)
    
    if actual_pixels > 0:
        # Create a rectangular region of foreground pixels
        # Calculate dimensions for a roughly square region
        side_length = int(np.sqrt(actual_pixels))
        if side_length * side_length < actual_pixels:
            side_length += 1
            
        # Ensure we don't exceed image boundaries
        side_length = min(side_length, width, height)
        
        # Center the region in the image
        start_x = (width - side_length) // 2
        start_y = (height - side_length) // 2
        
        # Fill the region with white (foreground)
        end_x = min(start_x + side_length, width)
        end_y = min(start_y + side_length, height)
        
        image[start_y:end_y, start_x:end_x] = 255
        
        # If we need more pixels, add them in a systematic way
        remaining_pixels = actual_pixels - (end_x - start_x) * (end_y - start_y)
        if remaining_pixels > 0:
            # Add remaining pixels row by row
            for y in range(start_y, min(start_y + side_length, height)):
                for x in range(start_x, min(start_x + side_length, width)):
                    if remaining_pixels <= 0:
                        break
                    if image[y, x] == 0:  # Only fill empty pixels
                        image[y, x] = 255
                        remaining_pixels -= 1
                if remaining_pixels <= 0:
                    break
    
    # Save the test image
    cv2.imwrite(output_path, image)
    print(f"Created test image: {output_path} with {actual_pixels} foreground pixels")