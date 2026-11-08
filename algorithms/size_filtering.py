
import cv2
import numpy as np


def filter_objects_by_size(image, min_area=100, max_area=100000):
    if image is None:
        raise ValueError("Invalid input image.")

    min_area = int(min_area)
    max_area = int(max_area)

    if min_area < 0:
        raise ValueError("Minimum area cannot be negative.")

    if max_area <= 0:
        raise ValueError("Maximum area must be greater than zero.")

    if min_area > max_area:
        raise ValueError("Minimum area cannot exceed maximum area.")

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create binary image
    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY
    )

    # Identify connected components
    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    # Create a black output image
    output = np.zeros_like(gray)

    kept_count = 0

    # Ignore background label 0
    for label in range(1, number_of_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        # Keep objects within the selected area range
        if min_area <= area <= max_area:
            output[labels == label] = 255
            kept_count += 1

    # Convert to BGR to display information
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        output,
        f"Objects Retained: {kept_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return output

