import cv2
import os

from YOLOv8Detector import YOLOv8Detector


class VideoProcessor:
    def __init__(self, detector, image_dir, output_video_path, fps=30):
        """
        初始化 VideoProcessor

        参数:
            detector (YOLOv8Detector): YOLOv8 检测器实例
            image_dir (str): 图片序列所在的目录
            output_video_path (str): 输出视频文件路径
            fps (int): 输出视频的帧率，默认为 30
        """
        self.detector = detector
        self.image_dir = image_dir
        self.output_video_path = output_video_path
        self.fps = fps

    def process_images(self):
        """
        读取图片序列，进行检测，并将结果保存为视频文件同时实时显示
        """
        images = sorted(os.listdir(self.image_dir))
        first_image = cv2.imread(os.path.join(self.image_dir, images[0]))
        height, width, _ = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (width, height))

        for image_name in images:
            image_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(image_path)
            detected_image = self.detector.detect(image)
            out.write(detected_image)
            cv2.imshow('YOLOv8 Detection', detected_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    model_path = 'model/yolov8n.pt'
    image_dir = 'img1'
    output_video_path = 'output/output_video.mp4'

    detector = YOLOv8Detector(model_path)
    video_processor = VideoProcessor(detector, image_dir, output_video_path)
    video_processor.process_images()
