from imagecorrect import ImageProcess
import os

if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, 'test_data', 'test2.jpg')
    model_path = 'E:/RMBG/RMBG-2.0'
    processor = ImageProcess(model_path, image_path)
    output_path = os.path.join(script_dir, 'result.png')
    processor.process_image(output_path)
