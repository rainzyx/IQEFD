import os
import argparse

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import random
import numpy as np

from my_dataset import MyDataSet
from model import convnext_base as create_model
from modelconvnext import convnext_base as create_model1
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, evaluate_fgsm
from robust_black import add_random_gaussian_noise,add_salt_pepper_noise

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import torch

import matplotlib.pyplot as plt

torch.cuda.empty_cache()


def load_train_data():
    train_fila_path = []
    train_file_label = []
    # path = r'data23/mattr_train.txt'
    path = r'data23/nolight_train.txt'
    for line in open(path):
        line = line.strip()
        data = line.split(' ', 1)
        train_fila_path.append(data[0])
        train_file_label.append(int(data[1]))
    return train_fila_path,train_file_label

def load_val_data():
    val_fila_path = []
    val_file_label = []
    path = r'data23/nolight_val.txt'
    for line in open(path):
        line = line.strip()
        data = line.split(' ', 1)
        val_fila_path.append(data[0])
        val_file_label.append(int(data[1]))
    return val_fila_path, val_file_label

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 3
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")


    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # tb_writer = SummaryWriter()

    train_images_path, train_images_label = load_train_data()
    val_images_path, val_images_label = load_val_data()

    img_size = 256
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.Lambda(lambda x: add_random_gaussian_noise(x, 0.2, 10)),  # 添加随机噪声，之前流模型用的是10，0很大，可以调小一些，
                                     transforms.ToTensor(),
                                     # transforms.Lambda(lambda x: add_salt_pepper_noise(x, noise_probability=0.2, max_amount=0.1, salt_vs_pepper=0.5)), # 盐椒噪声
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    print(train_dataset)
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    model1 = create_model1(num_classes=args.num_classes).to(device)

    print(model.to(device))

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)

            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    Epoch = []
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                model1=model1,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     model1=model1,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # FGSM
        # epsilons = [0, 0.05, 0.1, 0.2, 0.3]
        # fgsm_acc = evaluate_fgsm(model=model,
        #                          model1=model1,
        #                          device=device,
        #                          test_loader=val_loader,   # 20% or 100%
        #                          epsilon=0.05,
        #                          epoch=epoch)
        # print(fgsm_acc)

        if(val_acc>best_acc):
            best_acc=val_acc



        torch.save(model.state_dict(), "./weights/nolight/")
        print('bestacc'+str(best_acc))

        Epoch.append(epoch)
        Train_loss.append(train_loss)
        Train_acc.append(train_acc)
        Val_loss.append(val_loss)
        Val_acc.append(val_acc)

    save_dir = './image'  # 结果保存文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    sa = pd.DataFrame(
            {'epoch': Epoch, 'train_loss': Train_loss, 'train_acc': Train_acc, 'val_loss': Val_loss,
             'val_acc': Val_acc})
    sa.to_csv(r'save_two/nolight_lr_8e-5.csv', index=None, encoding='utf8')
    print('--------')

    print('--------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-5)  # 0.0004
    parser.add_argument('--wd', type=float, default=5e-4)

    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="")

    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA
    parser.add_argument('--weights', type=str, default=r'convnext_base_1k_224_ema.pth',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
