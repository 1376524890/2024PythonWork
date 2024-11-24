import cv2
from YOLOv8Detector import YOLOv8Detector

class VideoProcessor:
    def __init__(self, detector, camera_index=1, output_video_path=None, fps=30):
        """
        初始化 VideoProcessor

        参数:
            detector (YOLOv8Detector): YOLOv8 检测器实例
            camera_index (int): 摄像头索引，默认为 0
            output_video_path (str): 输出视频文件路径，如果为 None，则不保存视频
            fps (int): 输出视频的帧率，默认为 30
        """
        self.detector = detector
        self.camera_index = camera_index
        self.output_video_path = output_video_path
        self.fps = fps

    def process_video(self):
        """
        从摄像头获取视频流，进行检测，并将结果实时显示
        如果指定了输出视频路径，则将结果保存为视频文件
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise IOError("无法打开摄像头")

        if self.output_video_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected_frame = self.detector.detect(frame)

            if self.output_video_path:
                out.write(detected_frame)

            cv2.imshow('YOLOv8 Detection', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if self.output_video_path:
            out.release()
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    model_path = 'model/yolov8s.pt'
    output_video_path = 'output/output_video.mp4'

    detector = YOLOv8Detector(model_path)
    video_processor = VideoProcessor(detector, camera_index=0, output_video_path=output_video_path)
    video_processor.process_video()
