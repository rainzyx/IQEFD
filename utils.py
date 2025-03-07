import os
import sys
import json
import pickle
import random
import math

from torch import nn

import cat
import light_model
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from focal_loss import FocalLoss
from HAM import HAMBlock
from modelconvnext import ConvNeXt
from robust_white import fgsm_attack


# from sklearn.classifier import classifiers
def feature_distillation_loss(F_s, F_t):
    return F.mse_loss(F_s, F_t)

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "not find data for train."
    assert len(val_images_path) > 0, "not find data for eval"


    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, model1, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    model1.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function1 = FocalLoss(3)
    loss_function2 = FocalLoss(3)
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        norm1 = nn.LayerNorm(1024, eps=1e-6).to('cuda')
        norm2 = nn.LayerNorm(1024, eps=1e-6).to('cuda')
        head1 = nn.Linear(1024, 3).to('cuda')
        head2 = nn.Linear(1024, 3).to('cuda')

        DCE_net = light_model.enhance_net_nopool().cuda()

        checkpoint = torch.load('snapshots/Epoch146.pth', weights_only=True)
        DCE_net.load_state_dict(checkpoint)

        _, enhanced_image, _ = DCE_net(images.to(device))

        sample_num += images.shape[0]

        pred_enhance,a= model1(enhanced_image.to(device))
        pred,out,x1_feature,x2_feature,convnext_feature,final_feature = model(images.to(device))

        multiB = final_feature * a   # 8,1024,8,8
        multiB = norm1(multiB.mean([-2, -1]))   # 8,1024
        multiB = head1(multiB)  # 8,3

        loss3 = feature_distillation_loss(final_feature, a)

        # head1 = nn.Linear(dims[-1], 3)
        # head2 = nn.Linear(dims[-1], 3)
        pred_classes = torch.max(pred, dim=1)[1]  # shape [8,3] ->
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss1 = loss_function1(pred, labels.to(device))
        loss2 = loss_function2(multiB, labels.to(device))


        loss = loss1 + loss2 + 0.000001 * loss3
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.5f}, lr: {:.6f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, model1, data_loader, device, epoch):
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = FocalLoss(3)
    model.eval()
    model1.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        norm1 = nn.LayerNorm(1024, eps=1e-6).to('cuda')  # 分类头1的layernorm
        norm2 = nn.LayerNorm(1024, eps=1e-6).to('cuda')  # 分类头2的layernrom
        head1 = nn.Linear(1024, 3).to('cuda')  # 分类头1的全连接层
        head2 = nn.Linear(1024, 3).to('cuda')  # 分类头2的全连接层

        DCE_net = light_model.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load('snapshots/Epoch146.pth'))

        _, enhanced_image, _ = DCE_net(images.to(device))

        sample_num += images.shape[0]

        pred, out, x1_feature, x2_feature, convnext_feature, final_feature = model(images.to(device))
        pred_enhance, a = model1(enhanced_image.to(device))


        # cat.gets(labels, pred, step)

        # tsne
        # out_pred_1=out_pred[3].cpu().numpy()
        # feature = out_fea[3].cpu().numpy()
        # lb = labels.cpu().numpy()
        # f_name = '/DATA/YM/convenxt_d/feature_label/fea/' + str(step) + '.txt'
        # p_name = '/DATA/YM/convenxt_d/feature_label/pre/' + str(step) + '.txt'
        # out_fea_name = '/DATA/YM/convenxt_d/feature_label/out/' + str(step) + '.txt'
        # l_name = '/DATA/YM/convenxt_d/feature_label/lb/' + str(step) + '.txt'

        # np.savetxt(p_name, out_pred_1, fmt='%.3f')
        # np.savetxt(l_name, lb, fmt='%d')

        multiB = final_feature * a
        multiB = norm1(multiB.mean([-2, -1]))
        multiB = head1(multiB)

        loss3 = feature_distillation_loss(final_feature, a)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss1 = loss_function(pred, labels.to(device))
        loss2 = loss_function(multiB, labels.to(device))

        loss = loss1 + loss2 + 0.000001 * loss3
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
    # cat.get_cat_pre()
    # cat.get_cat_lb()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def evaluate_fgsm(model, model1, device, test_loader, epsilon, epoch):
    """
    """
    correct = 0
    sample_num = 0
    model.eval()

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        perturbed_data = fgsm_attack(model, model1, data, target, epsilon)

        _,a = model1(perturbed_data)
        output, _, _, _, _, _ = model(perturbed_data, a)
        pred = output.max(1, keepdim=True)[1]

        correct += pred.eq(target.view_as(pred)).sum().item()

        sample_num += data.shape[0]
        test_loader.desc = "[valid epoch {}]  acc: {:.5f}".format(
            epoch,
            correct.item() / sample_num
        )

    final_acc = correct / len(test_loader.dataset)
    # print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc:.4f}")

    return final_acc

# epsilons = [0, 0.05, 0.1, 0.2, 0.3]
#
# accuracies = []
# for eps in epsilons:
#     acc = test_fgsm(model, device, test_loader, eps)
#     accuracies.append(acc)



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)




def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

# def con_loss(features, labels):
#     B, _ = features.shape
#     features = F.normalize(features)
#     cos_matrix = features.mm(features.t())
#     pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
#     neg_label_matrix = 1 - pos_label_matrix
#     pos_cos_matrix = 1 - cos_matrix
#     neg_cos_matrix = cos_matrix - 0.4
#     neg_cos_matrix[neg_cos_matrix < 0] = 0
#     loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
#     loss /= (B * B)
#     return loss

def feature_loss_function(fea,target_fea):
    loss = (fea -target_fea)**2*((fea>0)|(target_fea>0)).float()
    return torch.abs(loss).sum()