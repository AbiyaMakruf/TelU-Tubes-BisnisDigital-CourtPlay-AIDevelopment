from utils.read_video import read_video
from utils.BallDetector import BallDetector
from utils.CourtDetector import CourtDetector
from utils.PlayerDetector import PersonDetector
from utils.BounceDetector import BounceDetector
from utils.combine import combine
from utils.write import write_video, write_image
from utils.scene_manager import scene_detect

video_name = "5s.mp4"
video_path = f"test_videos/{video_name}"
output_path = f"results/output"

# Read video
frames, fps, original_width, original_height = read_video(video_path, resize=False)

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
image_result, imgs_minimap_ball, imgs_minimap_player, imgs_heatmap_player = combine(frames,
                       scenes, 
                       bounces, 
                       ball_track, 
                       homography_matrices, 
                       kps_court, 
                       persons_top, 
                       persons_bottom,
                       draw_trace=True)


convert_mp4 = False
# Save output video
print("Saving output video...")
write_video(image_result, fps, output_path, convert_mp4=convert_mp4)

print("Saving minimap and heatmap images...")
write_video(imgs_minimap_ball, fps, "results/minimap_ball", convert_mp4=convert_mp4)
write_video(imgs_minimap_player, fps, "results/minimap_player", convert_mp4=convert_mp4)
write_video(imgs_heatmap_player, fps, "results/heatmap_player", convert_mp4=convert_mp4)

print("Saving sample images...")
write_image(imgs_minimap_ball, "results/", True, "minimap_ball")
# write_image(imgs_heatmap_player, "results/", False, "heatmap_player")