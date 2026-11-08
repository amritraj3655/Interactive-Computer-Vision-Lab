import cv2


def apply_threshold(
    image,
    threshold_type="Binary",
    threshold_value=127
):

    if image is None:
        raise ValueError("Invalid input image.")

    # Thresholding requires a grayscale image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    threshold_value = int(threshold_value)

    if not 0 <= threshold_value <= 255:
        raise ValueError("Threshold value must be between 0 and 255.")

    threshold_type = threshold_type.lower()

    # 1. Binary Thresholding
    if threshold_type == "binary":

        _, result = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

    # 2. Binary Inverted Thresholding
    elif threshold_type == "binary inverted":

        _, result = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV
        )

    # 3. Adaptive Mean Thresholding
    elif threshold_type == "adaptive mean":

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    # 4. Adaptive Gaussian Thresholding
    elif threshold_type == "adaptive gaussian":

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    # 5. Otsu Thresholding
    elif threshold_type == "otsu":

        _, result = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    else:
        raise ValueError("Unsupported thresholding technique.")

    return result