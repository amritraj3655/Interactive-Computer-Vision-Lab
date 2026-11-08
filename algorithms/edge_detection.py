import cv2


def detect_edges(
    image,
    edge_method="Canny",
    threshold1=100,
    threshold2=200
):
    if image is None:
        raise ValueError("Invalid input image.")

    # Convert input into grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edge_method = edge_method.lower()

    if edge_method == "canny":

        if threshold1 < 0 or threshold2 < 0:
            raise ValueError("Thresholds cannot be negative.")

        if threshold1 >= threshold2:
            raise ValueError(
                "Canny lower threshold must be less than upper threshold."
            )

        result = cv2.Canny(
            gray,
            int(threshold1),
            int(threshold2)
        )

    elif edge_method == "sobel":

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            sobel_x,
            sobel_y
        )

        result = cv2.convertScaleAbs(magnitude)

    elif edge_method == "laplacian":

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F,
            ksize=3
        )

        result = cv2.convertScaleAbs(laplacian)

    else:
        raise ValueError("Unsupported edge detection method.")

    return result