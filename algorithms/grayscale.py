import cv2


def convert_to_grayscale(image):
    """
    Convert a BGR colour image into a grayscale image.

    Args:
        image: Input BGR image.

    Returns:
        Grayscale image.
    """

    if image is None:
        raise ValueError("Invalid input image.")

    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return grayscale_image