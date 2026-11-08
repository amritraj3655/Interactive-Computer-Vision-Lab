
import cv2
import numpy as np


def label_and_count_objects(image, threshold_value=127):
    if image is None:
        raise ValueError("Invalid input image.")

    threshold_value = int(threshold_value)

    if not 0 <= threshold_value <= 255:
        raise ValueError("Threshold value must be between 0 and 255.")

    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert image into a binary image
    _, binary = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Find connected components
    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    # Exclude background (label 0)
    object_count = number_of_labels - 1

    # Prepare output image
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Assign a different color and number to each object
    rng = np.random.default_rng(42)

    for label in range(1, number_of_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        center_x, center_y = centroids[label]

        color = tuple(
            int(value) for value in rng.integers(50, 256, size=3)
        )

        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            2
        )

        # Display object number
        cv2.putText(
            output,
            f"Object {label}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Mark centroid
        cv2.circle(
            output,
            (int(center_x), int(center_y)),
            4,
            (0, 0, 255),
            -1
        )

    # Display total object count
    cv2.putText(
        output,
        f"Total Objects: {object_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return output
