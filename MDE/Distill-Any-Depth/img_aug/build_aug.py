from typing import Any
from . import transforms as aug_trans
import matplotlib.pyplot as plt

AUG_LIST = ['ColorJitter', 'GaussianBlur', 'VerticalFlip', 'HorizontalFlip', 'Rotate90', 'Rotate180', 'Rotate270']

class AugTransform(object):
    def __init__(self, aug_list=AUG_LIST):
        self.trans = []
        for aug in aug_list:
            trans = getattr(aug_trans, aug)()
            self.trans.append(trans)

    def __call__(self, image):
        aug_images = [image]
        for trans in self.trans:
            trans_img=trans(image)
            aug_images.append(trans_img)
        return aug_images
    
    def recover(self, aug_imgs_depth):
        recover_imgs_depth = [aug_imgs_depth[0]]
        aug_imgs_depth = aug_imgs_depth[1:]
        for i, aug_depth in enumerate(aug_imgs_depth):
            recover_depth = self.trans[i].recover(aug_depth)
            recover_imgs_depth.append(recover_depth)

        return recover_imgs_depth
