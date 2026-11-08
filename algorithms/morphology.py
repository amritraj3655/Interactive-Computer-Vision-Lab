import cv2
import numpy as np


def apply_morphology(
    image,
    operation="Erosion",
    kernel_size=3
):
    if image is None:
        raise ValueError("Invalid input image.")

    kernel_size = int(kernel_size)

    if kernel_size < 1 or kernel_size > 31:
        raise ValueError("Kernel size must be between 1 and 31.")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Convert the image into grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    # Convert grayscale image into binary image
    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Create structuring element
    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    operation = operation.lower()

    # Erosion
    if operation == "erosion":

        result = cv2.erode(
            binary,
            kernel,
            iterations=1
        )

    # Dilation
    elif operation == "dilation":

        result = cv2.dilate(
            binary,
            kernel,
            iterations=1
        )

    # Opening
    elif operation == "opening":

        result = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

    # Closing
    elif operation == "closing":

        result = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

    else:
        raise ValueError("Unsupported morphological operation.")

    return result
