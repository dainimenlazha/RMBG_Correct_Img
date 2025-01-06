import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

class ImagePreProcessor:
    
    def __init__(self, model_path, original_image):
        self.load_model(model_path)
        self.original_image = original_image
        
    def load_model(self, model_path):
        """加载图像分割模型"""
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self.model = model
    
    def preprocess_image(self):
        """加载和预处理图像, 图像分割并返回透明背景图像"""
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transformed_image = transform_image(self.original_image).unsqueeze(0)
        
        with torch.no_grad():
            preds = self.model(transformed_image)[-1].sigmoid()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(self.original_image.size)
        self.original_image.putalpha(mask)
        
        # 转换为 NumPy 格式并清除背景
        mask_array = np.array(mask) / 255.0
        image_array = np.array(self.original_image.convert("RGBA"))
        image_array[mask_array < 0.5] = [0, 0, 0, 0]
        
        segmented_image = cv2.cvtColor(np.array(Image.fromarray(image_array, "RGBA")), cv2.COLOR_RGBA2BGRA)
        return segmented_image, mask
    