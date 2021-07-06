from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Absolute import: from engine.retinaface
# use relative path for moving folder easily
from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .loader import load_model
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class RetinafaceDetector(object):
    def __init__(self, conf, device):
        cudnn.benchmark = True
        self.conf = conf
        self.net = conf.network
        self.device = device
        self.model = load_model(conf, device).to(self.device)
        self.model.eval()
        if self.conf.network == 'mnet':
            self.detector_conf = cfg_mnet
        else:
            self.detector_conf = cfg_re50

    def detect_faces(self, img_raw, confidence_threshold=0.8, top_k=50,
                     nms_threshold=0.4, keep_top_k=10, resize=1):
        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        img.requires_grad = False
        scale = scale.to(self.device)

        # tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))
        self.model.zero_grad()
        priorbox = PriorBox(self.detector_conf, image_size=(im_height, im_width))
        # priors = priorbox.forward().to(self.device)
        # priors = priors.to(self.device)
        prior_data = priorbox.forward().to(self.device).data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.detector_conf['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]]).to(self.device)
        # scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        dets_landms = np.concatenate((dets, landms), axis=1)

        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10)
        # print(landms.shape)
        del img
        # only get good box and landmard for good face
        boxes = []
        fine_landms = []
        for (b, landm) in zip(dets_landms, landms):
            if b[4] < self.conf.vis_thres:
                continue
            boxes.append(b)
            fine_landms.append(landm)
            # text = f'{b[4]:.4f}'
            # b = list(map(int, b))
            # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cx = b[0]
            # cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # # landms
            # # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            # # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            # # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            # # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            # # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        # return dets, landms, img_raw, boxes
        return dets, fine_landms, img_raw, boxes
