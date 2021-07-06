from .detector import dlib_detector
from .mask_function import aux_functions as aux_function
from .opentool import Config
import cv2
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AUTO_MASK(object):
    def __init__(self, config):
        self.detector = dlib_detector.DLIB_DETECTOR()
        self.function = aux_function
        self.config = config

    def mask(self, image):
        is_success = False
        face_locations, bboxes = self.detector.detect_faces(image)
        masked_image, mask, mask_binary, original_image = self.function.mask_image(self.detector, face_locations, image, self.config)
        if len(masked_image) == 0:
            masked_image = image
        else:
            is_success = True
            masked_image = masked_image[0]
        return masked_image, is_success

