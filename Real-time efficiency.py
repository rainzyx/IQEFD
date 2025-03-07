import torch
import os
from my_dataset import MyDataSet
from model import convnext_base as create_model
import time
from torchvision import transforms


def load_val_data():
    val_fila_path = []
    val_file_label = []
    path = r'data23/nolight_val_1000.txt'
    for line in open(path):
        line = line.strip()
        data = line.split(' ', 1)
        val_fila_path.append(data[0])
        val_file_label.append(int(data[1]))
    return val_fila_path, val_file_label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(3).to(device)
checkpoint_path = "./weights/"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
val_images_path, val_images_label = load_val_data()

img_size = 256
data_transform = {
    "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])

batch_size = 1
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=0,
                                         collate_fn=val_dataset.collate_fn)

total_images = 0
total_inference_time = 0.0

start_total = time.time()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        total_images += images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        start_infer = time.time()
        pred = model(images)
        _, preds = torch.max(pred, 1)
        end_infer = time.time()

        inference_time = end_infer - start_infer
        total_inference_time += inference_time

end_total = time.time()

overall_time = end_total - start_total
overall_fps = total_images / overall_time
inference_fps = total_images / total_inference_time
overall_latency = 1/overall_fps
inference_latency = 1/inference_fps

print(f"Overall FPS (including dataloader overhead): {overall_fps:.2f} images/sec")
print(f"Inference FPS (excluding dataloader overhead): {inference_fps:.2f} images/sec")
print(f"Overall Latency (including dataloader overhead): {overall_latency:.6f} sec")
print(f"Inference Latency (excluding dataloader overhead): {inference_latency:.6f} sec")





