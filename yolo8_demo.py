from ultralytics import YOLO
import cv2

# 加载预训练的 YOLOv8 模型
model = YOLO('model/yolov8l.pt')

# 读取图像
image_path = 'img1/000017.jpg'
image = cv2.imread(image_path)

# 进行推理
results = model(image)

# 处理结果
for result in results:
    boxes = result.boxes  # 获取检测框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示结果
cv2.imshow('YOLOv8 Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
