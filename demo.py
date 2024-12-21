from AIDetector_pytorch import Detector
import imutils
import rich
import cv2
import time

def main():

    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture('mot.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    frame_count = 0
    total_time = 0

    videoWriter = None

    while True:

        start_time = time.time()

        # try:
        _, im = cap.read()
        if im is None:
            break
        #rich.print("im:",im)
        result = det.feedCap(im)
        #rich.print(result)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count +=1

        if frame_count > 0:
            processing_fps = frame_count / total_time
            rich.print('Processing fps:', processing_fps)


        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()