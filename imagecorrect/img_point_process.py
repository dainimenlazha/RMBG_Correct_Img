import numpy as np
from sklearn.cluster import KMeans

class ImagePointProcessor:
    """
    图像处理工具类，包括点聚类、区域划分和关键点提取等功能。
    """

    @staticmethod
    def kmeans_clustering(points, n_clusters=4):
        """
        使用 K-Means 聚类算法将点分为 n_clusters 个簇。
        """
        if not points:
            raise ValueError("Points list is empty.")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_
        return [(int(centroid[0]), int(centroid[1])) for centroid in centroids]

    @staticmethod
    def get_furthest_point(points, center):
        """
        根据给定的中心点，找出与该中心点距离最远的点。
        
        :param points: 输入点的坐标列表
        :param center: 中心点坐标，形如 (cx, cy)
        :return: 与中心点距离最远的点，形如 (x, y)
        """
        if not points:
            return (0, 0)
        
        distances = [np.linalg.norm(np.array(point) - np.array(center)) for point in points]
        max_index = np.argmax(distances)
        return points[max_index]

    @staticmethod
    def divide_points(points, width, height):
        """
        根据坐标将点分为左上、左下、右上、右下四部分。
        
        :param points: 输入点的坐标列表
        :param width: 图像的宽度
        :param height: 图像的高度
        :return: 每个区域内距离中心最远的点列表，按 [左上, 右上, 左下, 右下] 排序
        """
        if not points:
            raise ValueError("Points list is empty.")
        
        # 中心线
        mid_width = width / 2
        mid_height = height / 2

        # 区域划分
        left_top_points = []
        left_down_points = []
        right_top_points = []
        right_down_points = []

        for x, y in points:
            if x < mid_width:
                if y < mid_height:
                    left_top_points.append((x, y))
                else:
                    left_down_points.append((x, y))
            else:
                if y < mid_height:
                    right_top_points.append((x, y))
                else:
                    right_down_points.append((x, y))

        # 计算每个区域的中心点
        left_top_center = np.mean(left_top_points, axis=0) if left_top_points else (0, 0)
        left_down_center = np.mean(left_down_points, axis=0) if left_down_points else (0, 0)
        right_top_center = np.mean(right_top_points, axis=0) if right_top_points else (0, 0)
        right_down_center = np.mean(right_down_points, axis=0) if right_down_points else (0, 0)

        # 获取每个区域内距离中心最远的点
        left_top_furthest = ImagePointProcessor.get_furthest_point(left_top_points, left_top_center)
        left_down_furthest = ImagePointProcessor.get_furthest_point(left_down_points, left_down_center)
        right_top_furthest = ImagePointProcessor.get_furthest_point(right_top_points, right_top_center)
        right_down_furthest = ImagePointProcessor.get_furthest_point(right_down_points, right_down_center)

        return [left_top_furthest, right_top_furthest, left_down_furthest, right_down_furthest]

    @staticmethod
    def process_points(points, width, height):
        """
        将点进行聚类并按区域划分，返回最终的四个关键点。
        
        :param points: 输入点的坐标列表
        :param width: 图像的宽度
        :param height: 图像的高度
        :return: 四个关键点的坐标列表，按 [左上, 右上, 左下, 右下] 排序
        """
        if not points:
            raise ValueError("Points list is empty.")
        
        clustered_points = ImagePointProcessor.kmeans_clustering(points, n_clusters=4)
        final_points = ImagePointProcessor.divide_points(clustered_points, width, height)
        return final_points
