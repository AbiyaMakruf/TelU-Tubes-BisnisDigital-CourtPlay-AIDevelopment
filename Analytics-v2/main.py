from utils.read_video import read_video
from utils.BallDetector import BallDetector
from utils.CourtDetector import CourtDetector
from utils.PlayerDetector import PersonDetector
from utils.BounceDetector import BounceDetector
from utils.combine import combine
from utils.write_video import write
from utils.scene_manager import scene_detect

video_name = "15s.mp4"
video_path = f"test_videos/{video_name}"
output_path = f"results/output"

# Read video
frames, fps, original_width, original_height = read_video(video_path)

# Scene detection
scenes = scene_detect(video_path)

#  Inference ball track
ball_detector = BallDetector(path_model='models/ball_track.pt',
                             original_width=original_width, 
                             original_height=original_height)
ball_track = ball_detector.infer_model(frames)

# Inference court
court_detector = CourtDetector(path_model='models/court_detector.pt',
                               original_width=original_width,
                               original_height=original_height)
homography_matrices, kps_court = court_detector.infer_model(frames)

# Inference player
person_detector = PersonDetector('cuda')
persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

# Inference bounce
bounce_detector = BounceDetector(path_model='models/bounce_detector.cbm')
x_ball = [x[0] for x in ball_track]
y_ball = [x[1] for x in ball_track]
bounces = bounce_detector.predict(x_ball, y_ball)

# Combine into image
image_result = combine(frames,
                       scenes, 
                       bounces, 
                       ball_track, 
                       homography_matrices, 
                       kps_court, 
                       persons_top, 
                       persons_bottom,
                       draw_trace=True)

# Save output video
write(image_result, fps, output_path)