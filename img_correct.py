import cv2
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ImageProcessor:
    """图像处理类，包含分割、轮廓提取、透视变换等功能"""
    
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """加载图像分割模型"""
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        model.eval()
        return model
    
    def preprocess_image(self, image_path, image_size=(1024, 1024)):
        """加载和预处理图像"""
        image = Image.open(image_path)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_images = transform_image(image).unsqueeze(0)  # 数据添加批次维度
        return image, input_images
    
    def segment_image(self, input_images, original_image):
        """图像分割并返回透明背景图像"""
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid()
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
    
    def find_contour_and_corners(self, mask, image):
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
        final_points = self.process_points(points, image)
        return final_points, max_contour
    
    def kmeans_clustering(self, points, n_clusters=4):
        """使用 K-Means 聚类算法将点分为 n_clusters 个簇"""
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_
        centroids_int = [(int(centroid[0]), int(centroid[1])) for centroid in centroids]
        return centroids_int
    
    def get_furthest_point(self, points, center):
        """根据给定的中心点，找出与该中心点距离最远的点"""
        distances = [np.linalg.norm(np.array(point) - np.array(center)) for point in points]
        max_index = np.argmax(distances)
        return points[max_index]
    
    def divide_points(self, points, image):
        """根据坐标将点分为左上、左下、右上、右下四部分"""
        width, height = image.size
        width = width / 2  # 图像垂直中线
        height = height / 2  # 图像水平中线
        
        # 定义四个区域点的列表
        left_top_points = []
        left_down_points = []
        right_top_points = []
        right_down_points = []
        
        for point in points:
            x, y = point
            if x < width:
                if y < height:
                    left_top_points.append((x, y))  # 左上区域
                else:
                    left_down_points.append((x, y))  # 左下区域
            else:
                if y < height:
                    right_top_points.append((x, y))  # 右上区域
                else:
                    right_down_points.append((x, y))  # 右下区域
        
        # 计算每个区域的中心点
        left_top_center = np.mean(left_top_points, axis=0) if left_top_points else (0, 0)
        left_down_center = np.mean(left_down_points, axis=0) if left_down_points else (0, 0)
        right_top_center = np.mean(right_top_points, axis=0) if right_top_points else (0, 0)
        right_down_center = np.mean(right_down_points, axis=0) if right_down_points else (0, 0)
        
        # 从每个区域选择与中心点最远的点
        left_top_furthest = self.get_furthest_point(left_top_points, left_top_center) if left_top_points else (0, 0)
        left_down_furthest = self.get_furthest_point(left_down_points, left_down_center) if left_down_points else (0, 0)
        right_top_furthest = self.get_furthest_point(right_top_points, right_top_center) if right_top_points else (0, 0)
        right_down_furthest = self.get_furthest_point(right_down_points, right_down_center) if right_down_points else (0, 0)

        return [left_top_furthest, right_top_furthest, left_down_furthest, right_down_furthest]
    
    def process_points(self, points, image):
        # 将点进行 K-Means 聚类
        clustered_points = self.kmeans_clustering(points, n_clusters=4)
        
        # 对聚类后的点按 x, y 坐标排序
        clustered_points_sorted = sorted(clustered_points, key=lambda p: p[1])
        
        # 使用 divide_points 函数将点分成四个区域的顶点
        final_points = self.divide_points(clustered_points_sorted, image)
        
        return final_points
    
    def perspective_transformation(self, final_points, src):
        """透视变换"""
        src_points = np.array(final_points, dtype=np.float32)
        width = int(math.sqrt((final_points[0][0] - final_points[1][0])**2 +
                              (final_points[0][1] - final_points[1][1])**2))
        height = int(math.sqrt((final_points[0][0] - final_points[2][0])**2 +
                               (final_points[0][1] - final_points[2][1])**2))
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        h, _ = cv2.findHomography(src_points, dst_points)
        return cv2.warpPerspective(src, h, (width, height))
    
    def image_enhance(self, image):
        """图像增强"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)


# ============================ 主逻辑 ============================

if __name__ == "__main__":
    model_path = r'E:\RMBG\RMBG-2.0'
    # image_path = r'E:\RMBG\test2.png'
    image_path = r'E:\RMBG\test5.jpg'
    
    # 初始化图像处理类
    processor = ImageProcessor(model_path)
    
    # 预处理输入图像
    original_image, input_images = processor.preprocess_image(image_path)
    
    # 分割图像
    segmented_image, mask = processor.segment_image(input_images, original_image)
    
    # 查找轮廓和角点
    final_points, max_contour = processor.find_contour_and_corners(mask, original_image)
    print(final_points)
    
    # 绘制轮廓和角点
    contour_image = segmented_image.copy()
    cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0, 255), 2)
    for point in final_points:
        cv2.circle(contour_image, point, 8, (0, 0, 255, 255), -1)
    
    # 透视变换
    transformed_image = processor.perspective_transformation(final_points, segmented_image)
    
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow("Segmented Image", contour_image)
    cv2.imshow("Transformed", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
