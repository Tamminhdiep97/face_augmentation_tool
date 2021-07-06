from __future__ import print_function

import torch

# Absolute import: from engine.face_detector.retinaface
# Use relative import for moving folder easily
from .data import cfg_mnet, cfg_re50
from .models.retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    def split_(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {split_(key): value for key, value in state_dict.items()}


def load_model(conf, device):
    pretrained_path = conf.weights
    if conf.network == 'mnet':
        print('Loading pretrained model from {}'.format(pretrained_path))
        model = RetinaFace(cfg=cfg_mnet, phase='test',
                           pretrained_path=conf.detector_backbone,
                           device=device)
    else:
        print('Loading pretrained model from {}'.format(pretrained_path))
        model = RetinaFace(cfg=cfg_re50, phase='test',
                           pretrained_path=conf.detector_backbone,
                           device=device)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path,
                                    map_location=lambda storage,
                                    loc: storage.cuda(device))
    else:
        pretrained_dict = torch.load(pretrained_path,
                                    map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model = model.eval()
    print('Finished loading model!')
    return model
