from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import time

start = time.time()

model = AutoModelForImageSegmentation.from_pretrained(r'E:\RMBG\RMBG-2.0', trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型转移到适当的设备（GPU 或 CPU）
model.eval()  # 切换到评估模式，确保推理时使用正确的操作

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_path = r'E:\RMBG\test5.jpg'
image = Image.open(image_path)
input_images = transform_image(image).unsqueeze(0).to('cpu')

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image2.png")

import os
end = time.time()
print(f"图片大小为{os.path.getsize(image_path) / 1024}Kb，共计用时：{end - start}s")
