import cv2


def detect_faces(image_path):
    """
    检测给定图片中的人脸，并返回人脸的坐标。

    参数:
        image_path (str): 图片文件的路径。

    返回:
        list: 人脸坐标的列表，每个元素为 (x, y, w, h)，其中 x 和 y 是人脸框左上角的坐标，w 和 h 分别是宽度和高度。
    """
    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图片无法读取，请检查路径是否正确")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 输出每个人脸的位置
    for (x, y, w, h) in faces:
        print(f"检测到人脸位置：({x}, {y}), 宽度：{w}, 高度：{h}")

    return faces


# 使用示例
image_path = 'frames/frame_0000.jpg'  # 替换为你的图片路径
faces = detect_faces(image_path)
print(faces)