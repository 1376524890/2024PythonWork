# pic2video.py
import cv2
import os
import glob

import peopleIdentify


class ImageSequencePlayer:
    def __init__(self, image_dir, frame_rate=30):
        """
        初始化 ImageSequencePlayer 类

        :param image_dir: 图片文件所在的目录
        :param frame_rate: 播放帧率，默认为30帧/秒
        """
        self.image_dir = image_dir
        self.frame_rate = frame_rate
        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    def play(self):
        """
        按指定帧率播放图片序列
        """
        delay = int(1000 / self.frame_rate)  # 计算每帧之间的延迟时间（毫秒）

        for image_path in self.images:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"无法读取图片: {image_path}")
                continue

            # 绘制方框
            for(x,y,w,h)in peopleIdentify.detect_faces(image_path):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制文字
            text = "Sample Text"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 0, 0)
            thickness = 2
            text_position = (50, 50)  # 文字的位置
            cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness)

            cv2.imshow('Image Sequence', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    image_dir = 'img1'
    player = ImageSequencePlayer(image_dir, frame_rate=30)
    player.play()
