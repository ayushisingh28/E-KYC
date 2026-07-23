import os
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from utils import read_yaml, file_exists, resolve_path
except Exception:
    from importlib import util
    spec = util.spec_from_file_location("utils", str(Path(__file__).resolve().parent / "utils.py"))
    utils_module = util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    read_yaml = utils_module.read_yaml
    file_exists = utils_module.file_exists
    resolve_path = utils_module.resolve_path

# Logging configuration
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

# ---------------DEBUGGING--------------
# Testing the functionality of logging (Easier for Debugging)
# # Example log messages
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.yaml")
config = read_yaml(config_path) or {}
# print(config)

artifacts = config.get('artifacts', {}) or {}
intermediate_dir_path = artifacts.get('INTERMIDEIATE_DIR', 'data/02_intermediate_data')
conour_file_name = artifacts.get('CONTOUR_FILE', 'contour_id.jpg')
# print(intermediate_dir_path)

def read_image(image_path, is_uploaded=False):
    if is_uploaded:
        try:
            image_bytes = image_path.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if img is None:
                logging.info("Failed to read image: {}".format(image_path))
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            logging.info(f"Error reading image: {e}")
            print("Error reading image:", e)
            return None
    else:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.info("Failed to read image: {}".format(image_path))
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            logging.info(f"Error reading image: {e}")
            print("Error reading image:", e)
            return None

# ---------------DEBUGGING-----------

# Example usage of read_image
# image_path = "data/02_intermediate_data/id_card.png"
# image = read_image(image_path, is_uploaded=False)

# if image is not None:
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to load image.")


def extract_id_card(img):
    if img is None:
        logging.error("Cannot extract ID card from an empty image")
        return None, None

    # Convert image to grayscale
    # ---------------------- Reduces Computational Complexity involved ----------------
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    #----This helps in creating smoother contours and reduces the chances of detecting false contours------
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Adaptive thresholding
    # -------- This helps in distinguishing the foreground (ID card) from the background -------
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # --------- MORPHOLOGICAL EXPRESSIONS ------------------
    # Apply morphological operations
    # kernel = np.ones((5, 5), np.uint8)

    # # Apply opening to remove small noise
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # # Apply closing to fill small holes
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assuming the ID card is the largest object)
    largest_contour = None
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_contour = cnt
            largest_area = area

    # If no large contour is found, assume no ID card is present
    if largest_contour is None:
        logging.warning("No ID card contour detected")
        return None, None

    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    logging.info(f"contours are found at, {(x, y, w, h)}")
    # logging.info("Area largest_area)

    # Apply additional filtering (optional):
    # - Apply bilateral filtering for noise reduction
    # filtered_img = cv2.bilateralFiltering(img[y:y+h, x:x+w], 9, 75, 75)
    # - Morphological operations (e.g., erosion, dilation) for shape refinement
    output_dir = Path(resolve_path(intermediate_dir_path, str(BASE_DIR)) or str(BASE_DIR / "data" / "02_intermediate_data"))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = str(output_dir / conour_file_name)
    contour_id = img[y:y+h, x:x+w]
    if file_exists(filename):
        os.remove(filename)

    cv2.imwrite(filename, contour_id)

    return contour_id, filename


# ----------- DEBUGGING ----------------
# # Example usage of extract_id_card
# image_path = "data/02_intermediate_data/deskew.png"
# image = cv2.imread(image_path)

# if image is not None:
#     extracted_image, file_path = extract_id_card(image)
#     if extracted_image is not None:
#         logging.info(f"Extracted ID card saved to: {file_path}")
#     else:
#         logging.error("No ID card detected in the image.")
# else:
#     logging.error("Failed to load image.")



# ------ Saving the Image ----------

def save_image(image, filename, path="."):
    output_dir = Path(resolve_path(path, str(BASE_DIR)) or str(BASE_DIR / "data" / "02_intermediate_data"))
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / filename
    full_path_str = str(full_path)

    if file_exists(full_path_str):
        os.remove(full_path_str)

    success = cv2.imwrite(full_path_str, image)
    if not success:
        logging.warning(f"Failed to save image to {full_path_str}")
        return None

    logging.info(f"Image saved successfully: {full_path_str}")
    return full_path_str

# -------------- DEBUGGING ------------------

# # Example image (a simple black square)
# image = np.zeros((100, 100, 3), dtype=np.uint8)

# # Save the image
# saved_path = save_image(image, "black_square.jpg", "images")
