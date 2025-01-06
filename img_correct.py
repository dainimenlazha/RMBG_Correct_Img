from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import cv2
import math

# ============================ 函数定义 ============================

def load_model(model_path):
    """加载分割模型"""
    model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
    torch.set_float32_matmul_precision('high')
    model.eval()
    return model


def preprocess_image(image_path, image_size=(1024, 1024)):
    """加载和预处理图像"""
    image = Image.open(image_path)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_images = transform_image(image).unsqueeze(0)  # 数据添加批次维度
    return image, input_images


def segment_image(model, input_images, original_image):
    """图像分割并返回透明背景图像"""
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_image.size)
    original_image.putalpha(mask)

    # 转换为 NumPy 格式并清除背景
    mask_array = np.array(mask) / 255.0
    image_array = np.array(original_image.convert("RGBA"))
    image_array[mask_array < 0.5] = [0, 0, 0, 0]

    # 转换为 OpenCV 格式
    return cv2.cvtColor(np.array(Image.fromarray(image_array, "RGBA")), cv2.COLOR_RGBA2BGRA), mask


def find_contour_and_corners(mask):
    """检测最大轮廓并提取四个角点"""
    mask_array = np.array(mask)
    edges = cv2.Canny(mask_array, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # 多边形逼近
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # 转换为元组并排序角点
    points = [tuple(point[0]) for point in approx]
    points = sorted(points, key=lambda p: p[1])
    top_points = sorted(points[:2], key=lambda p: p[0])
    bottom_points = sorted(points[2:], key=lambda p: p[0])
    return top_points + bottom_points, max_contour


def perspective_transformation(final_points, src):
    """透视变换"""
    src_points = np.array(final_points, dtype=np.float32)
    width = int(math.sqrt((final_points[0][0] - final_points[1][0])**2 +
                          (final_points[0][1] - final_points[1][1])**2))
    height = int(math.sqrt((final_points[0][0] - final_points[2][0])**2 +
                           (final_points[0][1] - final_points[2][1])**2))
    dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    h, _ = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(src, h, (width, height))


def image_enhance(image):
    """图像增强"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


# ============================ 主逻辑 ============================

if __name__ == "__main__":
    # 加载模型
    model_path = r'E:\RMBG\RMBG-2.0'
    model = load_model(model_path)

    # 预处理输入图像
    image_path = r'E:\RMBG\test.png'
    original_image, input_images = preprocess_image(image_path)

    # 分割图像
    segmented_image,mask = segment_image(model, input_images, original_image)

    # 查找轮廓和角点
    final_points, max_contour = find_contour_and_corners(mask)

    # 绘制轮廓和角点
    contour_image = segmented_image.copy()
    cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0, 255), 2)
    for point in final_points:
        cv2.circle(contour_image, point, 8, (0, 0, 255, 255), -1)

    # 透视变换
    transformed_image = perspective_transformation(final_points, segmented_image)

    # 图像增强
    enhanced_image = image_enhance(transformed_image)

    # 显示结果
    cv2.imshow("Segmented Image", contour_image)
    cv2.imshow("Transformed", transformed_image)
    cv2.imshow("Enhanced", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
