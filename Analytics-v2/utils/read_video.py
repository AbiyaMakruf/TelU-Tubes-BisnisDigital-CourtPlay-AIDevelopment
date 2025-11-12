import cv2
def read_video(path_video, resize=False, width=1280, height=720):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame =  cap.read()
        if ret:
            if resize:
                frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps, original_width, original_height