
import cv2


def create_background_subtractor():
    """
    Create a MOG2 background subtractor.
    A new instance should be created for each video.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=40,
        detectShadows=True
    )


def subtract_background(frame, subtractor):
    if frame is None:
        raise ValueError("Invalid video frame.")

    # Apply background subtraction
    foreground_mask = subtractor.apply(frame)

    # Remove shadows (MOG2 typically marks shadows as 127)
    _, foreground_mask = cv2.threshold(
        foreground_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Create a colour foreground result
    foreground = cv2.bitwise_and(
        frame,
        frame,
        mask=foreground_mask
    )

    # Show foreground object count using contours
    contours, _ = cv2.findContours(
        foreground_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    object_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                foreground,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            object_count += 1

    cv2.putText(
        foreground,
        f"Moving Objects: {object_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return foreground
