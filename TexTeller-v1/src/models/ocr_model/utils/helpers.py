import cv2
import numpy as np
from typing import List, Union


def convert2rgb(image_paths: List[Union[str, bytes]]) -> List[np.ndarray]:
    processed_images = []
    for path in image_paths:
        if isinstance(path, str):
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        elif isinstance(path, bytes):
            image = cv2.imdecode(np.frombuffer(path, np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            raise TypeError(f"Unsupported type {type(path)} for image path.")
        if image is None:
            print(f"Image at {path} could not be read.")
            continue
        if image.dtype == np.uint16:
            print(f'Converting {path} to 8-bit, image may be lossy.')
            image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_images.append(image)

    return processed_images
