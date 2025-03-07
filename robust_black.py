import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms

def add_random_gaussian_noise(img, noise_prob=0.2, max_sigma=10.0):
    if random.random() < noise_prob:
        if not isinstance(img, torch.Tensor):
            img = transforms.functional.to_tensor(img)
        sigma = random.uniform(0, max_sigma)
        noise = torch.randn_like(img) * sigma
        img = img + noise
        img = transforms.functional.to_pil_image(img.clamp(0, 1))

    return img

def add_salt_pepper_noise(image, noise_probability=0.2, max_amount=0.1, salt_vs_pepper=0.5):
    if random.random() < noise_probability:
        noisy_image = image.clone()
        c, h, w = noisy_image.shape
        num_pixels = h * w

        amount = random.uniform(0, max_amount)
        num_salt = int(amount * num_pixels * salt_vs_pepper)
        num_pepper = int(amount * num_pixels * (1 - salt_vs_pepper))

        for channel in range(c):
            salt_coords = [torch.randint(0, h, (num_salt,)), torch.randint(0, w, (num_salt,))]
            noisy_image[channel, salt_coords[0], salt_coords[1]] = 1.0

            pepper_coords = [torch.randint(0, h, (num_pepper,)), torch.randint(0, w, (num_pepper,))]
            noisy_image[channel, pepper_coords[0], pepper_coords[1]] = 0.0

        return noisy_image
    else:
        return image