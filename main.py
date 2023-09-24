import argparse
import os
import torch

from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, 
    ResNet101_Weights, ResNet152_Weights
)
from PIL import Image
from src.gradcam_vis import GradCamVisualize, save_image

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-path',
                        type=str,
                        default='',
                        help='Input image path')
    
    parser.add_argument('--model',
                        type=str,
                        default='resnet50',
                        choices=[
                            'resnet18',
                            'resnet34',
                            'resnet50',
                            'resnet101',
                            'resnet152'
                        ],
                        help='choose model to use'
                        )
    
    parser.add_argument('--layer',
                        type=int,
                        default=4,
                        choices=[1, 2, 3, 4],
                        help='Choose layer to visualize')
    
    parser.add_argument('--block',
                        type=int,
                        default=-1,
                        help='choose block in layer')
    
    parser.add_argument('--output_dir',
                        type=str,
                        default='./output/',
                        help='Output directory to save image')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    models = {
        'resnet18' : (resnet18, ResNet18_Weights.DEFAULT),
        'resnet34' : (resnet34, ResNet34_Weights.DEFAULT),
        'resnet50' : (resnet50, ResNet50_Weights.DEFAULT),
        'resnet101' : (resnet101, ResNet101_Weights.DEFAULT),
        'resnet152' : (resnet152, ResNet152_Weights.DEFAULT)
    }
    
    args = get_args()
    
    if os.path.isfile(args.image_path) == False:
        raise Exception('Wrong path to image!')
    img_path = args.image_path        
    
    if models.get(args.model) == None:
        raise Exception('Wrong name of model!')
    model, weights = models[args.model]
    model = model(weights=weights)
    preprocessing = weights.transforms()
    
    layers = {
        1 : model.layer1,
        2 : model.layer2,
        3 : model.layer3,
        4 : model.layer4
    }
    
    if layers.get(args.layer) == None:
        raise Exception('Wrong layer! Choose int form 1 to 4')
    layer = layers[args.layer]
    
    if len(layer) <= args.block:
        raise Exception(f'Wrong number of block! Choose int form 0 to {len(layer) - 1}')
    target_layers = [layer[args.block]] 
    
    
    gradcamvis = GradCamVisualize(model, target_layers, 
                                  torch.cuda.is_available(),
                                  preprocessing)
    
    vis = gradcamvis.process(img_path)
    img_with_vis = Image.fromarray(vis)
    
    if os.path.isdir(args.output_dir) == False:
      print(f'Creating new dir {args.output_dir}')
      os.mkdir(args.output_dir)

    save_image(img_with_vis, img_path, args.output_dir)      
    
    
    