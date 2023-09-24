import numpy as np
import torch
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision import transforms
from typing import Callable, List, Tuple

from src.target import target_id

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

class GradCamVisualize():
    def __init__(self, 
                 model,
                 target_layers,
                 use_cuda,
                 preproc
                 ):
        self.model = model
        self.preproc = preproc
        self.use_cuda = use_cuda
        self.gradcam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        self.transform = transforms.ToTensor()
        
    def process(self, img_path: str)->np.ndarray:
        img = Image.open(img_path)
        input_t = torch.stack((self.transform(img),))
        targets = [ClassifierOutputTarget(target_id(self.model, img, self.preproc, self.use_cuda))]
        grayscale_cam = self.gradcam(input_tensor=input_t, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(normalize(img), grayscale_cam, use_rgb=True)
        return visualization

def save_image(img, img_path, dir_to_save):
    img_name = os.path.splitext(os.path.basename(img_path))
    img = img.save(dir_to_save + img_name[0] + '_out' + img_name[1])
    
    