import os
from PIL import Image
import numpy as np
import face_recognition


def get_face_encodings():
    photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photos')

    names = []
    images = []

    for name in os.listdir(photos_dir):
        if os.path.isdir(os.path.join(photos_dir, name)):
            for image_path in os.listdir(os.path.join(photos_dir, name)):
                img_path = os.path.join(photos_dir, name, image_path)
                if any([img_path.endswith('jpg'),img_path.endswith('png'),img_path.endswith('jpeg')]):
                    images.append(Image.open(img_path, 'r').convert("RGB"))
                    names.append(name)

    images = [np.array(each) for each in images]
    face_encodings = [face_recognition.face_encodings(img)[0] for img in images]
    return {
        'names': names,
        'face_encodings': face_encodings
    }
