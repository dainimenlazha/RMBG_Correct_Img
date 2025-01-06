import cv2
import math
import numpy as np
import os

from PIL import Image

from .img_pre_process import ImagePreProcessor
from .img_point_process import ImagePointProcessor

class ImageProcess:

    def __init__(self, model_path, image_path):
        self.original_image = Image.open(image_path)
        self.original_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        self.pre_process = ImagePreProcessor(model_path, self.original_image)
        
    
    def process_image(self, output_dir,show = True):
        segmented_image, mask = self.pre_process.preprocess_image()
        final_points, max_contour = self.find_contour_and_corners(mask, show)
        transformed_image = self.perspective_transformation(final_points, segmented_image)
        cv2.imwrite(output_dir, transformed_image)
        if show :
            contour_image = segmented_image.copy()
            cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0, 255), 2)
            for point in final_points:
                cv2.circle(contour_image, point, 8, (0, 0, 255, 255), -1)
            cv2.imshow("Contour Image", contour_image)
            cv2.imshow("Transformed", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def find_contour_and_corners(self, mask, show):
        mask_array = np.array(mask)
        edges = cv2.Canny(mask_array, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        points = [tuple(point[0]) for point in approx]


        if show:
            if len(mask_array.shape) == 2:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
            for contour in contours:
                cv2.drawContours(mask_array, [contour], -1, (0, 255, 0, 255), 2)
            for point in points:
                cv2.circle(mask_array, point, 8, (0, 0, 255, 255), -1)
            cv2.imshow("Points Image", mask_array)
            cv2.waitKey(0)



        width, height = self.original_image.size
        final_points = ImagePointProcessor.process_points(points, width, height)
        return final_points, max_contour


    def perspective_transformation(self, final_points, segmented_image):
        src_points = np.array(final_points, dtype=np.float32)
        width = int(math.sqrt((final_points[0][0] - final_points[1][0])**2 +
                              (final_points[0][1] - final_points[1][1])**2))
        height = int(math.sqrt((final_points[0][0] - final_points[2][0])**2 +
                               (final_points[0][1] - final_points[2][1])**2))
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        h, _ = cv2.findHomography(src_points, dst_points)
        return cv2.warpPerspective(segmented_image, h, (width, height))
