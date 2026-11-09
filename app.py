from algorithms.grayscale import convert_to_grayscale
from algorithms.filtering import apply_filter
from algorithms.thresholding import apply_threshold
from algorithms.edge_detection import detect_edges
from algorithms.morphology import apply_morphology
from algorithms.corner_detection import detect_harris_corners
from algorithms.object_labelling import label_and_count_objects
from algorithms.size_filtering import filter_objects_by_size
from algorithms.hough_transform import detect_hough_lines
from algorithms.optical_flow import calculate_optical_flow
from algorithms.background_subtraction import (
    create_background_subtractor,
    subtract_background
)

from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

CONCEPTS = [
    {"id": "grayscale", "name": "Grayscale Conversion", "input": "both", "description": "Convert a colour image or video into grayscale.", "learn": "https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html", "parameters": []},
    {"id": "filtering", "name": "Image Filtering", "input": "both", "description": "Smooth images and reduce noise using spatial filters.", "learn": "https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html", "parameters": [{"name":"filter_type","label":"Filter type","type":"select","options":["Gaussian","Median","Average"]},{"name":"kernel_size","label":"Kernel size (odd number)","type":"number","min":3,"max":31,"step":2,"value":5}]},
    {"id": "thresholding", "name": "Image Thresholding", "input": "both", "description": "Separate foreground and background using pixel intensity.", "learn": "https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html", "parameters": [{"name":"threshold_type","label":"Threshold type","type":"select","options":["Binary","Binary Inverted","Adaptive Mean","Adaptive Gaussian","Otsu"]},{"name":"threshold_value","label":"Threshold value","type":"number","min":0,"max":255,"value":127}]},
    {"id": "edge_detection", "name": "Edge Detection", "input": "both", "description": "Highlight object boundaries and intensity changes.", "learn": "https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html", "parameters": [{"name":"edge_method","label":"Method","type":"select","options":["Canny","Sobel","Laplacian"]},{"name":"threshold1","label":"Canny lower threshold","type":"number","min":0,"max":255,"value":100},{"name":"threshold2","label":"Canny upper threshold","type":"number","min":0,"max":255,"value":200}]},
    {"id": "morphology", "name": "Morphological Operations", "input": "both", "description": "Modify shapes in binary images using a structuring element.", "learn": "https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html", "parameters": [{"name":"operation","label":"Operation","type":"select","options":["Erosion","Dilation","Opening","Closing"]},{"name":"kernel_size","label":"Kernel size","type":"number","min":1,"max":31,"step":2,"value":3}]},
    {"id": "corner_detection", "name": "Corner Detection", "input": "both", "description": "Find distinctive corner points in an image or video frame.", "learn": "https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html", "parameters": [{"name":"method","label":"Method","type":"select","options":["Harris"]},{"name":"sensitivity","label":"Corner sensitivity","type":"number","min":0.01,"max":0.1,"step":0.01,"value":0.04}]},
    {"id": "object_counting", "name": "Object Labelling and Counting", "input": "both", "description": "Label connected foreground regions and count detected objects.", "learn": "https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html", "parameters": [{"name":"threshold_value","label":"Binary threshold","type":"number","min":0,"max":255,"value":127}]},
    {"id": "size_filtering", "name": "Size Filtering", "input": "both", "description": "Keep or remove connected regions based on their pixel area.", "learn": "https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html", "parameters": [{"name":"min_area","label":"Minimum area (pixels)","type":"number","min":0,"max":100000,"value":100},{"name":"max_area","label":"Maximum area (pixels)","type":"number","min":1,"max":1000000,"value":100000}]},
    {"id": "hough", "name": "Hough Transform – Line Detection", "input": "both", "description": "Detect straight lines in an image or video frame.", "learn": "https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html", "parameters": [{"name":"threshold","label":"Accumulator threshold","type":"number","min":1,"max":300,"value":50},{"name":"min_line_length","label":"Minimum line length","type":"number","min":1,"max":1000,"value":50},{"name":"max_line_gap","label":"Maximum line gap","type":"number","min":0,"max":500,"value":10}]},
    {"id": "optical_flow", "name": "Optical Flow", "input": "video", "description": "Visualize apparent motion between consecutive video frames.", "learn": "https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html", "parameters": [{"name":"flow_method","label":"Method","type":"select","options":["Farneback"]}]},
    {"id": "background_subtraction", "name": "Background Subtraction", "input": "video", "description": "Separate moving foreground objects from a mostly static background.", "learn": "https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html", "parameters": [{"name":"history","label":"History frames","type":"number","min":10,"max":2000,"value":500},{"name":"var_threshold","label":"Variance threshold","type":"number","min":1,"max":200,"value":16}]}
]

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def parse_parameters():
    params = {}
    for key, value in request.form.items():
        if key == "concept":
            continue
        try:
            params[key] = float(value) if "." in value else int(value)
        except (TypeError, ValueError):
            params[key] = value
    return params

def process_frame(frame, concept, params):

    if concept == "grayscale":

        return convert_to_grayscale(frame)

    elif concept == "filtering":

        filter_type = params.get(
            "filter_type",
            "Gaussian"
        )

        kernel_size = params.get(
            "kernel_size",
            5
        )

        return apply_filter(
            frame,
            filter_type,
            kernel_size
        )

    elif concept == "thresholding":

        threshold_type = params.get(
            "threshold_type",
            "Binary"
        )

        threshold_value = params.get(
            "threshold_value",
            127
        )

        return apply_threshold(
            frame,
            threshold_type,
            threshold_value
        )
    elif concept == "edge_detection":

        edge_method = params.get(
            "edge_method",
            "Canny"
        )

        threshold1 = params.get(
            "threshold1",
            100
        )

        threshold2 = params.get(
            "threshold2",
            200
        )
    

        return detect_edges(
            frame,
            edge_method,
            threshold1,
            threshold2
        )    
        
    elif concept == "morphology":
        operation = params.get("operation", "Erosion")
        kernel_size = params.get("kernel_size", 3)

        return apply_morphology(
            frame,
            operation,
            kernel_size
        )

    elif concept == "corner_detection":
        sensitivity = params.get("sensitivity", 0.04)
        return detect_harris_corners(frame, sensitivity)
    elif concept == "object_counting":
        threshold_value = params.get("threshold_value", 127)
        return label_and_count_objects(frame, threshold_value)
    elif concept == "size_filtering":
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", 100000)

        return filter_objects_by_size(
            frame,
            min_area,
            max_area
        )    
    elif concept == "hough":
        threshold = params.get("threshold", 100)
        min_line_length = params.get("min_line_length", 50)
        max_line_gap = params.get("max_line_gap", 10)

        return detect_hough_lines(
            frame,
            threshold,
            min_line_length,
            max_line_gap
        )    
    
    
    
    raise ValueError(
        f"'{concept}' processing has not been implemented yet."
    )
def process_image(input_path, output_path, concept, params):
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")
    result = process_frame(image, concept, params)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError("Could not save the processed image.")



def process_video(input_path, output_path, concept, params):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read video dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the processed video.")

    # Store the previous frame for Optical Flow
    previous_frame = None

    # Create one background subtractor for the entire video
    background_subtractor = None

    if concept == "background_subtraction":
        background_subtractor = create_background_subtractor()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # -----------------------------------------
            # FEATURE 10: OPTICAL FLOW
            # -----------------------------------------
            if concept == "optical_flow":

                if previous_frame is None:
                    result = np.zeros_like(frame)

                    cv2.putText(
                        result,
                        "Waiting for next frame...",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                else:
                    result = calculate_optical_flow(
                        previous_frame,
                        frame
                    )

                # Store current frame for next comparison
                previous_frame = frame.copy()

            # -----------------------------------------
            # FEATURE 11: BACKGROUND SUBTRACTION
            # -----------------------------------------
            elif concept == "background_subtraction":

                result = subtract_background(
                    frame,
                    background_subtractor
                )

            # -----------------------------------------
            # OTHER VIDEO PROCESSING ALGORITHMS
            # -----------------------------------------
            else:
                result = process_frame(
                    frame,
                    concept,
                    params
                )

            # Convert grayscale output to BGR
            if len(result.shape) == 2:
                result = cv2.cvtColor(
                    result,
                    cv2.COLOR_GRAY2BGR
                )

            # Ensure output dimensions match input dimensions
            if (
                result.shape[1] != width
                or result.shape[0] != height
            ):
                result = cv2.resize(
                    result,
                    (width, height)
                )

            # Write processed frame
            writer.write(result)

    finally:
        cap.release()
        writer.release()



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/concepts")
def concepts():
    return jsonify(CONCEPTS)

@app.route("/api/process", methods=["POST"])
def process_upload():
    uploaded = request.files.get("file")
    concept = request.form.get("concept", "").strip()
    known_concepts = {item["id"] for item in CONCEPTS}

    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Please choose an image or video."}), 400
    if concept not in known_concepts:
        return jsonify({"error": "Please select a valid concept."}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, BMP, WEBP, MP4, AVI, MOV or MKV."}), 400

    extension = Path(uploaded.filename).suffix.lower()
    is_video = extension in ALLOWED_VIDEO_EXTENSIONS
    concept_info = next(item for item in CONCEPTS if item["id"] == concept)
    if concept_info["input"] == "video" and not is_video:
        return jsonify({"error": "This concept requires a video input."}), 400

    safe_name = secure_filename(uploaded.filename) or "upload"
    token = uuid4().hex
    input_path = UPLOAD_DIR / f"{token}_{safe_name}"
    output_extension = ".mp4" if is_video else ".png"
    output_name = f"{token}_processed{output_extension}"
    output_path = RESULT_DIR / output_name
    uploaded.save(input_path)
    params = parse_parameters()

    try:
        if is_video:
            process_video(input_path, output_path, concept, params)
        else:
            process_image(input_path, output_path, concept, params)
    except (ValueError, RuntimeError, cv2.error) as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    finally:
        input_path.unlink(missing_ok=True)

    return jsonify({
        "message": "Processing completed successfully.",
        "result_url": f"/static/results/{output_name}",
        "download_url": f"/api/download/{output_name}",
        "output_type": "video" if is_video else "image"
    })

@app.route("/api/download/<path:filename>")
def download_result(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "File is too large. Maximum upload size is 100 MB."}), 413

if __name__ == "__main__":
    app.run(debug=True)
