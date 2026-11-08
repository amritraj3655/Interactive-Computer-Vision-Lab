
import cv2
import numpy as np


def calculate_optical_flow(previous_frame, current_frame):
    if previous_frame is None or current_frame is None:
        raise ValueError("Two valid consecutive frames are required.")

    # Convert both frames to grayscale
    previous_gray = cv2.cvtColor(
        previous_frame, cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame, cv2.COLOR_BGR2GRAY
    )

    # Calculate dense optical flow using Farneback
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Separate horizontal and vertical motion
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # Create HSV visualization
    hsv = np.zeros_like(current_frame)

    # Hue represents motion direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Saturation represents colour intensity
    hsv[..., 1] = 255

    # Value represents motion magnitude
    normalized_magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    hsv[..., 2] = normalized_magnitude.astype(np.uint8)

    # Convert HSV visualization to BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.putText(
        result,
        "Farneback Optical Flow",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return result
