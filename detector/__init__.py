import cv2
import numpy as np
from collections import namedtuple
from PIL import Image
from os.path import join as opj

from .retinaface.align_faces import warp_and_crop_face, get_reference_facial_points
import time
import torch



def face_alignment(img, facial5point, output_size):
    """rotate image based on facial5point

    args:
        img: cv2 image
        facial5point: 5point landmarks (support for retinaface and mtcnn1)
        output_size: size image need to return
    return:
        dst_img: rotated image in cv2 image
    """
    facial5point = np.reshape(facial5point[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img = warp_and_crop_face(img, facial5point,
                                 reference_pts=reference_5pts,
                                 crop_size=output_size)
    return dst_img


def initial_face_detector(conf, device):
    """initial face detector

    args:
        conf: config for application
        logger: logging setting
        device: device setting
    return:
        face_detector: one of [retinaface, dlib_detector]
    """
    if conf.retinaface:
        from .retinaface.detector import RetinafaceDetector
        face_detector = RetinafaceDetector(conf, device)
    elif conf.dlib_detector:
        from . import dlib_detector as dlib
        face_detector = dlib.DLIB_DETECTOR()
    return face_detector


def detect_faces(conf, face_detector, original_image):
    """detect faces use retinaface/dlib

    agrs:
        conf: config for application
        face_detector: mtcnn1, retinaface for face detector
        original_image: cv2 image
    return:
        boxes: coordinate of bounding box of all faces on image
        faces: croped and aligned all faces on image
    """
    # start_time = time.time()
    image = original_image.copy()

    if conf.retinaface:
        # logger.info(f'using Retinaface')
        short_edge = min(image.shape[0], image.shape[1])
        scale_percent = 1
        image_copy = image.copy()
        if short_edge >= 500:
            scale_percent = float(500 / short_edge)
            width = int(image.shape[1] * scale_percent)
            height = int(image.shape[0] * scale_percent)
            dim = (width, height)
            image = cv2.resize(image, dim, cv2.INTER_AREA)

        # logger.info(f'put image through Retina neural network')
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # frame_bgr = image
        _, facial5points, frame_bgr, boxes = face_detector.detect_faces(frame_bgr)
        faces = []
        if scale_percent != 1:
            boxes = np.multiply(boxes, (1/scale_percent))
            facial5points = np.multiply(facial5points, (1/scale_percent))
        frame_bgr = cv2.cvtColor(np.array(image_copy), cv2.COLOR_RGB2BGR)
        #     frame_bgr = image_copy
        # logger.info(f'{len(boxes)} face/faces detected')
        if len(facial5points) > 0 and len(boxes) > 0 and len(facial5points) == len(boxes):
            for facial5point in facial5points:
                facial5point = np.expand_dims(facial5point, axis=0)
                face = face_alignment(frame_bgr, facial5point,
                                      (conf.require_size, conf.require_size))

                # # if model trained on RGB image
                # # convert cv2 format(BGR) to Image format(RGB)
                face_ = Image.fromarray(face[..., ::-1])

                # else model trained on BGR iaimage
                # face_ = Image.fromarray(face)
                faces.append(face_)
                # logger.info(f'infer through faces')
    return boxes, faces, facial5points
