import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])


        # data_lowlight = Image.open(self.images_path[item])


        # data_lowlight = data_lowlight.cuda().unsqueeze(0)
        if img.mode != 'RGB':
            img = img.convert("RGB")
            #img = Image.open("images/pytorch.png").convert('RGB')

        #raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
            # data_lowlight=self.transform(data_lowlight)

        # data_lowlight = (np.asarray(data_lowlight) / 255.0)
        # data_lowlight = torch.from_numpy(data_lowlight).float()
        # data_lowlight = data_lowlight.permute(2, 0, 1)


        # return img, label ,data_lowlight
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考

        # # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        # data_lowlight = torch.stack(data_lowlight, dim=0)

        labels = torch.as_tensor(labels)
        return images, labels

