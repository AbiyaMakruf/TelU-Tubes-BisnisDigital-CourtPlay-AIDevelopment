import torch
import cv2
import numpy as np
from .tracknet import BallTrackerNet
from tqdm import tqdm
from scipy.spatial import distance
class BallDetector:
    def __init__(self, path_model, original_width, original_height):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.original_width = original_width
        self.original_height = original_height
        self.width = 640
        self.height = 360
        self.scale_factor = self.original_width / self.width
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()

    
    def infer_model(self, frames):
        if len(frames) == 0:
            return []
        ball_track = [(None, None)] * len(frames)
        prev_pred = [None, None]
        if len(frames) < 3:
            return ball_track

        resized_prev2 = cv2.resize(frames[0], (self.width, self.height))
        resized_prev1 = cv2.resize(frames[1], (self.width, self.height))

        with torch.no_grad():
            for num in tqdm(range(2, len(frames))):
                resized_curr = cv2.resize(frames[num], (self.width, self.height))
                imgs = np.concatenate((resized_curr, resized_prev1, resized_prev2), axis=2)
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.transpose(imgs, (2, 0, 1))
                inp = np.expand_dims(imgs, axis=0)

                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                x_pred, y_pred = self.postprocess(output, prev_pred)
                prev_pred = [x_pred, y_pred]
                ball_track[num] = (x_pred, y_pred)

                resized_prev2, resized_prev1 = resized_prev1, resized_curr
        return ball_track

    def postprocess(self, feature_map, prev_pred, max_dist=80):
        scale = self.scale_factor
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

        x,y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
