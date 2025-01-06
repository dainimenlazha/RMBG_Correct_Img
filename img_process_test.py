from imagecorrect import ImageProcess
import os
import cv2

if __name__ == "__main__":
    show = True

    script_dir = os.path.dirname(__file__)
    # image_path = os.path.join(script_dir, 'test_data', 'test2.jpg')
    # image_path = os.path.join(script_dir, 'test_data', 'test3.jpg')
    # image_path = os.path.join(script_dir, 'test_data', 'test4.jpg')
    image_path = 'E:/RMBG/test5.jpg'
    model_path = 'E:/RMBG/RMBG-2.0'
    processor = ImageProcess(model_path, image_path)

    segmented_image, mask, final_points, max_contour, transformed_image = processor.process_image()
    cv2.imwrite(os.path.join(script_dir, 'result.png'), transformed_image)

    contour_image = segmented_image.copy()
    cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0, 255), 2)
    for point in final_points:
        cv2.circle(contour_image, point, 8, (0, 0, 255, 255), -1)
    cv2.imshow("Contour Image", contour_image)
    cv2.imshow("Transformed Image", transformed_image)

    enhanced_image = processor.enhance_image_pillow(transformed_image)
    cv2.imwrite(os.path.join(script_dir, 'result_enhanced.png'), enhanced_image)
    cv2.imshow("Enhanced Image", enhanced_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
