
import cv2
import numpy as np


def detect_hough_lines(
    image,
    threshold=100,
    min_line_length=50,
    max_line_gap=10
):
    if image is None:
        raise ValueError("Invalid input image.")

    threshold = int(threshold)
    min_line_length = int(min_line_length)
    max_line_gap = int(max_line_gap)

    if threshold < 1:
        raise ValueError("Hough threshold must be positive.")

    if min_line_length < 1:
        raise ValueError("Minimum line length must be positive.")

    if max_line_gap < 0:
        raise ValueError("Maximum line gap cannot be negative.")

    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = image.copy()
    else:
        gray = image.copy()
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Apply Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    line_count = 0

    if lines is not None:
        # Reshape to ensure every line contains four coordinates
        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:
            cv2.line(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

        line_count = len(lines)

    # Display detected line count
    cv2.putText(
        output,
        f"Detected Lines: {line_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    return output
