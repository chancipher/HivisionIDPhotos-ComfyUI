from hivision.creator.face_detector import *


def choose_handler(creator, matting_model_option=None, face_detect_option=None):
    creator.detection_handler = detect_face_mtcnn
