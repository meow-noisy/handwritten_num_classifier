

from torchvision import transforms
import numpy as np
import cv2


class Gray2BGR(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # おそらくPILが入力される
        sample = np.array(sample)
        sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        # import pdb; pdb.set_trace()
        return sample


class Invert(object):
    # 画像反転
    def __init__(self):
        pass

    def __call__(self, sample):
        # おそらくPILが入力される
        sample = 255 - sample
        # import pdb; pdb.set_trace()
        return sample


def get_transform(transform_name_list):

    transform_obj_list = []
    
    for t_dic in transform_name_list:
        t_name = list(t_dic.keys())[0]
        if t_name == 'ToTensor':
            transform_obj_list.append(transforms.ToTensor())
        if t_name == 'Resize':
            transform_obj_list.append(transforms.Resize((t_dic['height'], t_dic['width'])))
        if t_name == 'Gray2BGR':
            transform_obj_list.append(Gray2BGR())
        if t_name == 'Invert':
            transform_obj_list.append(Invert())

    return transforms.Compose(transform_obj_list)