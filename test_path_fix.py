from preprocess import save_image
from pathlib import Path
import cv2
import numpy as np


def test_save_image_uses_normalized_path():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    saved_path = save_image(image, 'test_face.jpg', path='data/02_intermediate_data')
    assert saved_path is not None
    assert Path(saved_path).exists()
    Path(saved_path).unlink(missing_ok=True)


test_save_image_uses_normalized_path()
print('path fix OK')
