import os
import requests
from zipfile import ZipFile
from tqdm import tqdm
import bz2
import shutil
import numpy as np
import dlib
import cv2
from PIL import Image


def download_dlib_model(path_to_dlib_model):
    print("Get dlib model", 60)
    dlib_model_link = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print("Downloading dlib model...")
    with requests.get(dlib_model_link, stream=True) as r:
        print("Zip file size: ", np.round(len(r.content) / 1024 / 1024, 2),
              "Mb")
        destination = ('mask_module' + os.path.sep + 'detector' +
                       os.path.sep + 'dlib_detector' + os.path.sep +
                       'shape_predictor_68_face_landmarks.dat.bz2')
        if not os.path.exists(destination.rsplit(os.path.sep, 1)[0]):
            os.mkdir(destination.rsplit(os.path.sep, 1)[0])
        print("Saving dlib model...")
        with open(destination, "wb") as fd:
            for chunk in r.iter_content(chunk_size=32678):
                fd.write(chunk)
    print("Extracting dlib model...")
    with bz2.BZ2File(destination) as fr, open(path_to_dlib_model, 'wb') as fw:
        shutil.copyfileobj(fr, fw)
    print("Saved: ", destination)
    print("done", 60)
    os.remove(destination)


def init():
    path_to_dlib_model = os.path.join('mask_module', 'detector', 'dlib_detector', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model(path_to_dlib_model)
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_dlib_model)
    return face_detector, predictor


class DLIB_DETECTOR(object):
    def __init__(self):
        self.face_detector, self.predictor = init()
        self.faces = dlib.full_object_detections()

    def rect_to_bb(self, rect):
        x1 = rect.left()
        x2 = rect.right()
        y1 = rect.top()
        y2 = rect.bottom()
        bbox = [x1, y1, x2, y2]
        return bbox

    def detect_faces(self, image, require_size=112):
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_locations = self.face_detector(gray, 0)
        if len(face_locations) == 0:
            return [], []
        else:
            bboxes = []
            for face in face_locations:
                bboxes.append(self.rect_to_bb(face))
        bboxes = np.vstack(bboxes)
        return face_locations, bboxes

    def align(self, img):
        face_locations, bboxes = self.detect_faces(img)
        shape = self.predictor(img, face_locations[0])
        self.faces.append(shape)
        images = dlib.get_face_chips(img, self.faces, size=112)
        self.faces = dlib.full_object_detections()
        return Image.fromarray(images[0])

    def align_multi(self, img):
        face_result = []
        face_locations, bboxes = self.detect_faces(img)
        for location in face_locations:
            self.faces.append(self.predictor(img, location))
        for face in self.faces:
            image = dlib.get_face_chip(img, face, size=112)
            face_result.append(Image.fromarray(image))
        self.faces = dlib.full_object_detections()
        return bboxes, face_result
