import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from Gradcam.utils import GradCAM, show_cam_on_image, center_crop_img
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from model import convnext_base as create_model
from modelconvnext import convnext_base as create_model1



def main():
    # build models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes = 3).to(device)
    model_weight_path = '../weights/0.97590584746548058e-5.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)

    # model1 = create_model1(num_classes = 3).to(device)
    # model1_weight_path = '../weights/nolight1/....'
    # model1.load_state_dict(torch.load(model1_weight_path, map_location=device), strict=False)


    target_layers = [model.hams_out.sa.conv]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path ="../24.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 1  # 指定目标类别索引为 1
    # snow:0 dry:1 wet:2

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    plt.imshow(visualization)
    plt.savefig('../image/{}.png'.format('z'), dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()