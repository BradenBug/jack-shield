from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import gi
gi.require_version('Wnck', '3.0')
from gi.repository import Wnck
import time

CONFIDENCE = 0.5
PROTOTXT = './resources/deploy.prototxt.txt'
MODEL = './resources/res10_300x300_ssd_iter_140000.caffemodel'
MAX_HUMANS = 1

def detect(frame, net):
    detection_count = 0

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detecetions and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            detection_count += 1

    return detection_count

def is_browser(window):
    return 'Firefox' in w.get_name() or 'Chromium' in w.get_name()


if __name__ == "__main__":
    screen = Wnck.Screen.get_default()

    # load serialized model from disk
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    vid = VideoStream(src=0).start()
    time.sleep(2)

    from_minimized = False

    while(True):
        frame = vid.read()
        frame = imutils.resize(frame, width=400)

        detection_count = detect(frame, net)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key was pressed, break
        if key == ord("q"):
            break

        # more than 1 human detected
        if detection_count > MAX_HUMANS:
            screen.force_update()
            windows = screen.get_windows()
            for w in windows:
                if is_browser(w):
                    w.minimize()
            from_minimized = True

        # 1 or 0 humans detected
        else:
            # only unminimize window if its last state was minimized
            if from_minimized:
                screen.force_update()
                w = screen.get_previously_active_window()
                if w is not None:
                    if not w.is_active() and is_browser(w):
                        w.activate(int(time.time()))
                        w.unminimize(int(time.time()))
                from_minimized = False

    cv2.destroyAllWindows()
    vc.stop()
