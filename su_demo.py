from AIDetector_pytorch import Detector
import imutils
import rich
import cv2
import time
import numpy as np
import requests

def main():
    name = 'demo'
    det = Detector()
    url = "http://10.52.61.126:8080?action=snapshot"

    frame_count = 0
    total_time = 0

    response = requests.get(url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)

        # 转换为OpenCV图像
    im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    cv2.imshow('snapshot',im)
    rich.print('niuniu')

    while True:
        start_time = time.time()

        # 获取图片数据
        response = requests.get(url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)

        # 转换为OpenCV图像
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if im is None:
            break

        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)

        cv2.imshow(name, result)
        
        # 显示图像并设置退出按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1

        if frame_count > 0:
            processing_fps = frame_count / total_time
            rich.print('Processing fps:', processing_fps)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
