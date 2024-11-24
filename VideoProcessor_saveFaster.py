import cv2
import os
import concurrent.futures
from tqdm import tqdm

from YOLOv8Detector_save import YOLOv8Detector

class VideoProcessor:
    def __init__(self, detector, image_dir, output_video_path, output_txt_path, fps=30, num_threads=4):
        """
        初始化 VideoProcessor

        参数:
            detector (YOLOv8Detector): YOLOv8 检测器实例
            image_dir (str): 图片序列所在的目录
            output_video_path (str): 输出视频文件路径
            output_txt_path (str): 输出文本文件路径
            fps (int): 输出视频的帧率，默认为 30
            num_threads (int): 并行处理的线程数，默认为 4
        """
        self.detector = detector
        self.image_dir = image_dir
        self.output_video_path = output_video_path
        self.output_txt_path = output_txt_path
        self.fps = fps
        self.num_threads = num_threads

    def process_images(self):
        """
        读取图片序列，进行检测，并将结果保存为视频文件
        同时将检测结果输出到文本文件中
        """
        images = sorted(os.listdir(self.image_dir))
        first_image = cv2.imread(os.path.join(self.image_dir, images[0]))
        height, width, _ = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (width, height))

        with open(self.output_txt_path, 'w') as txt_file:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self.process_image, os.path.join(self.image_dir, image_name), image_name, txt_file) for image_name in images]

                completed_count = 0
                total_images = len(images)
                progress_bar = tqdm(total=total_images, desc="Processing Images")

                for future in concurrent.futures.as_completed(futures):
                    detected_image, image_name = future.result()
                    out.write(detected_image)
                    completed_count += 1
                    progress_bar.update(1)

                progress_bar.close()

        out.release()

    def process_image(self, image_path, image_name, txt_file):
        """
        处理单张图片，返回带有检测框的图片和检测信息

        参数:
            image_path (str): 图片文件路径
            image_name (str): 图片文件名
            txt_file (file object): 文本文件对象

        返回:
            tuple: (numpy.ndarray, str) 带有检测框的图片和图片文件名
        """
        image = cv2.imread(image_path)
        detected_image, detection_info = self.detector.detect_with_info(image)

        # 写入检测信息到文本文件
        txt_file.write(f"Image: {image_name}\n")
        txt_file.write(f"Number of persons: {len(detection_info)}\n")
        for i, info in enumerate(detection_info):
            txt_file.write(f"Person {i + 1} - Bounding Box: {info['bbox']}, Confidence: {info['confidence']:.2f}\n")
        txt_file.write("\n")

        return detected_image, image_name

# 使用示例
if __name__ == "__main__":
    model_path = 'model/yolov8n.pt'
    image_dir = 'data/img3'
    output_video_path = 'output/output_video3n_f.mp4'
    output_txt_path = 'output/detection_info3n_f.txt'

    detector = YOLOv8Detector(model_path)
    video_processor = VideoProcessor(detector, image_dir, output_video_path, output_txt_path, num_threads=4)
    video_processor.process_images()
