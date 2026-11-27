import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, label):
#         for t in self.transforms:
#             image = t(image)
#         return image, label

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string
    

class ColorJitter(object):
    def __init__(self, 
                 brightness=0.2,
                 contrast=0.2,
                 saturation=0.2,
                 hue=0.2):
        self.color_jitter = torchvision.transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,)

    def __call__(self, image):
        pil_image = Image.fromarray(image)
        jittered_image = self.color_jitter(pil_image)
        jittered_ndarray = np.array(jittered_image)
        return jittered_ndarray
    
    def __name__(self):
        return "ColoJitter"
    
    def recover(self, image):
        return image
    

class GaussianBlur(object):
    def __init__(self,
                   kernel_size=9,
                   sigma=7):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, image):
        image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        return image

    def __name__(self):
        return "GaussianBlur"

    def recover(self, image):
        return image
    

class VerticalFlip(object):
    def __init__(self) -> None:
        pass

    def __call__(self, image):
        
        return np.flip(image, axis=0)
    
    def __name__(self):
        return "VerticalFlip"
    
    def recover(self, image):
        return np.flip(image, axis=0)
    
    
class HorizontalFlip(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self, image):
        return np.flip(image, axis=1)
    
    def __name__(self):
        return "HorizontalFlip"
    
    def recover(self, image):
        return np.flip(image, axis=1)
    

# class Rotate(object):
#     def __init__(self, angle):
#         self.k = int(angle / 90)
#         # 旋转定为90,180,270度
#         if self.k not in [1, 2, 3]:
#             raise ValueError("Rotate angle is invalid.")
    
#     def __call__(self, image):
#         return np.rot90(image, k=self.k)
    

class Rotate90(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return np.rot90(image, k=1)
    
    def __name__(self):
        return "Rotate90"
    
    def recover(self, image):
        return np.rot90(image, k=3)
    

class Rotate180(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return np.rot90(image, k=2)
    
    def __name__(self):
        return "Rotate180"
    
    def recover(self, image):
        return np.rot90(image, k=2)
    

class Rotate270(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return np.rot90(image, k=3)
    
    def __name__(self):
        return "Rotate270"
    
    def recover(self, image):
        return np.rot90(image, k=1)