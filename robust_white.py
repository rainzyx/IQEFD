import torch
import torch.nn.functional as F

def fgsm_attack(model, model1, data, target, epsilon):
    model.eval()
    model1.eval()

    data.requires_grad = True
    output,_,_,_,_,_ = model(data)
    loss = F.cross_entropy(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data

    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data
