
mask_type = 'random'  # choices=["surgical", "N95", "KN95", "cloth", "inpaint", "random"]
color = True
color_weight = 0.3
code = ''
write_original_image = ''
maskconfig_path = 'mask_module/masks/masks.cfg'
pattern = False
pattern_weight = False


# retina detector
retinaface = True
require_size = 112
if retinaface:
    # network = 'mnet'
    network = 'resnet50'
    if network == 'resnet50':
        weights = 'detector/retinaface/weights/Resnet50_Final.pth'
        detector_backbone = 'detector/retinaface/weights/resnet50-19c8e357.pth'
    else:
        weights = 'detector/retinaface/weights/mobilenet0.25_Final.pth'

    vis_thres = 0.8

image_folder = 'data_train'
prob = 0.05
minimum_image = 20
