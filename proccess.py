import cv2
import os
from os.path import join as opj
import tqdm
import time
import config as conf
import detector
import torch
import numpy as np
from tqdm import tqdm
import mask_module


path_proccess = conf.image_folder
mask_machine = mask_module.AUTO_MASK(conf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector =  detector.initial_face_detector(conf, device)


def process_mode(mode, image, name, x_landmarks, y_landmarks):
    x, y = 0, 0
    if mode == 0:
        w = x_landmarks
        h = y_landmarks - 5
        # crop = img_original[y:y+h, x:112]
        modify_image = np.zeros((112, 112, 3), dtype = "uint8")
        modify_image[y:y+h, x:112] =  img_original[y:y+h, x:112]
    elif mode == 1:
        w = x_landmarks
        h = y_landmarks - 5
        # crop = img_original[y:y+h, x:112]
        modify_image = np.zeros((112, 112, 3), dtype = "uint8")
        modify_image[h:112, 0:112] =  img_original[h:112, 0:112]
    elif mode == 2:
        try:
            modify_image, vat = mask_machine.mask(image)
        except Exception as e:
            print(e)
    elif mode == 3:
        return
    # name = str(int(time.time()*1000)) + '.jpg'
    cv2.imwrite(name, modify_image)
    return 1


for folder in tqdm(os.listdir(path_proccess)):
    if os.path.isfile(folder):
        continue
    folder_image_path = opj(path_proccess, folder)
    total_items = len(os.listdir(folder_image_path))
    if total_items < conf.minimum_image:
        continue
    count_item = 0
    mode = 0
    number_aug = int(conf.prob*total_items)
    for i in os.listdir(folder_image_path):
       if mode < 3:
            img = cv2.imread(opj(folder_image_path, i))
            img_original = img.copy()
            if mode < 2:
                bboxs, faces, landmarks = detector.detect_faces(conf, face_detector, img)
            if count_item < number_aug:
                count_item += 1
                # img_ = np.array(faces[0])
            else:
                mode += 1
                count_item = 1
            if type(landmarks) == list:
                landmarks = landmarks[0]
            if mode < 2 and len(faces) == 1:
                process_mode(mode=mode, image=img_original,
                            name=opj(folder_image_path, i),
                            x_landmarks=int(landmarks[2]),
                            y_landmarks=int(landmarks[7]))
            elif mode == 2:
                process_mode(mode=mode, image=img_original,
                             name=opj(folder_image_path, i),
                             x_landmarks=0,
                             y_landmarks=0)
            else:
                continue

