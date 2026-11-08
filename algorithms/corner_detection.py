
import cv2
import numpy as np


def detect_harris_corners(image, sensitivity=0.04):
    if image is None:
        raise ValueError("Invalid input image.")

    sensitivity = float(sensitivity)

    if not 0.01 <= sensitivity <= 0.1:
        raise ValueError("Sensitivity must be between 0.01 and 0.1.")

    # Ensure the output is a 3-channel color image
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    # Convert image to grayscale
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Harris Corner Detection requires float32 input
    gray_float = np.float32(gray)

    # Apply Harris Corner Detection
    response = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=sensitivity
    )

    # Dilate the response to make detected corners more visible
    response = cv2.dilate(response, None)

    # Identify strong corner responses
    max_response = response.max()

    if max_response > 0:
        corner_mask = response > (0.01 * max_response)

        # Highlight detected corners in red
        output[corner_mask] = [0, 0, 255]

    return output
