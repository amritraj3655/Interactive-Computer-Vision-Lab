import cv2


def apply_filter(image, filter_type="Gaussian", kernel_size=5):

    if image is None:
        raise ValueError("Invalid input image.")

    # Kernel size must be a positive odd number
    kernel_size = int(kernel_size)

    if kernel_size < 1:
        raise ValueError("Kernel size must be positive.")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    if kernel_size > 31:
        raise ValueError("Kernel size cannot exceed 31.")

    filter_type = filter_type.lower()

    if filter_type == "average":

        result = cv2.blur(
            image,
            (kernel_size, kernel_size)
        )

    elif filter_type == "gaussian":

        result = cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0
        )

    elif filter_type == "median":

        result = cv2.medianBlur(
            image,
            kernel_size
        )

    else:
        raise ValueError("Unsupported filtering technique.")

    return result