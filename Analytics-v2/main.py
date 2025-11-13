from collections import defaultdict
from time import perf_counter
from utils.read_video import video_metadata, VideoFrameAccessor
from utils.BallDetector import BallDetector
from utils.CourtDetector import CourtDetector
from utils.PlayerDetector import PersonDetector
from utils.BounceDetector import BounceDetector
from utils.combine import combine_stream, CombineRenderOptions
from utils.write import VideoStreamWriter, write_image
from utils.scene_manager import scene_detect

import time

start_time = time.time()
video_name = "120s.mp4"
video_path = f"test_videos/{video_name}"
output_path = "results/output"

# Toggle which visualizations to render to save memory/time.
OUTPUT_SELECTION = CombineRenderOptions(
    combined=True,
    minimap_ball=True,
    minimap_player=True,
    heatmap_player=True,
    heatmap_ball=True,
)
PLAYER_DETECTION_STRIDE = 2  # >=2 skips frames for faster player detection (slight accuracy drop)
CONVERT_MP4 = True


def ensure_length(seq, target_len, filler):
    if len(seq) < target_len:
        seq.extend([filler] * (target_len - len(seq)))
    elif len(seq) > target_len:
        del seq[target_len:]

# Track timings for each pipeline stage
timings = []

# Read metadata
print("[Read] Gathering video metadata...")
start = perf_counter()
fps, original_width, original_height, frame_count = video_metadata(video_path)
timings.append(("video_metadata", perf_counter() - start))

# Scene detection
print("[Scene] Detecting scene boundaries...")
start = perf_counter()
scenes = scene_detect(video_path)
timings.append(("scene_detect", perf_counter() - start))

# Inference ball track
print("[Ball] Running ball detector...")
start = perf_counter()
ball_detector = BallDetector(path_model='models/ball_track.pt',
                             original_width=original_width,
                             original_height=original_height)
ball_track = ball_detector.infer_video(video_path)
ensure_length(ball_track, frame_count, (None, None))
timings.append(("ball_detection", perf_counter() - start))

# Inference court
print("[Court] Estimating homography per frame...")
start = perf_counter()
court_detector = CourtDetector(path_model='models/court_detector.pt',
                               original_width=original_width,
                               original_height=original_height)
homography_matrices, kps_court = court_detector.infer_video(video_path)
ensure_length(homography_matrices, frame_count, None)
ensure_length(kps_court, frame_count, None)
timings.append(("court_detection", perf_counter() - start))

# Inference player (stride reduces runtime dramatically on long clips)
print(f"[Player] Tracking players with stride={PLAYER_DETECTION_STRIDE}...")
start = perf_counter()
person_detector = PersonDetector('cuda')
persons_top, persons_bottom = person_detector.track_players_video(
    video_path,
    homography_matrices,
    filter_players=False,
    stride=PLAYER_DETECTION_STRIDE,
)
ensure_length(persons_top, frame_count, [])
ensure_length(persons_bottom, frame_count, [])
timings.append(("player_detection", perf_counter() - start))

# Inference bounce
print("[Bounce] Predicting ball bounces...")
start = perf_counter()
bounce_detector = BounceDetector(path_model='models/bounce_detector.cbm')
x_ball = [x[0] for x in ball_track]
y_ball = [x[1] for x in ball_track]
bounces = bounce_detector.predict(x_ball, y_ball)
timings.append(("bounce_detection", perf_counter() - start))

frame_accessor = VideoFrameAccessor(video_path)

# Combine and stream-render videos to minimize memory usage
print("[Combine] Rendering selected visualizations...")
start = perf_counter()
combine_outputs = combine_stream(
    frame_accessor,
    scenes,
    bounces,
    ball_track,
    homography_matrices,
    kps_court,
    persons_top,
    persons_bottom,
    draw_trace=True,
    render_options=OUTPUT_SELECTION,
)

output_paths = {
    "combined": output_path,
    "minimap_ball": "results/minimap_ball",
    "minimap_player": "results/minimap_player",
    "heatmap_player": "results/heatmap_player",
    "heatmap_ball": "results/heatmap_ball_drop",
}
output_flags = {
    "combined": OUTPUT_SELECTION.combined,
    "minimap_ball": OUTPUT_SELECTION.minimap_ball,
    "minimap_player": OUTPUT_SELECTION.minimap_player,
    "heatmap_player": OUTPUT_SELECTION.heatmap_player,
    "heatmap_ball": OUTPUT_SELECTION.heatmap_ball,
}
writers = {
    name: VideoStreamWriter(fps, output_paths[name], convert_mp4=CONVERT_MP4)
    for name, enabled in output_flags.items() if enabled
}
last_frames = {name: None for name in writers.keys()}
enabled_streams = list(writers.keys())
render_stats = {name: {"frames": 0, "time": 0.0} for name in enabled_streams}
combine_breakdown = defaultdict(float)

print(f"[Combine] Rendering streams: {', '.join(enabled_streams) if enabled_streams else 'none'}")
for output in combine_outputs:
    if output_flags["combined"] and output.combined is not None:
        t_stream = perf_counter()
        writers["combined"].write(output.combined)
        render_stats["combined"]["time"] += perf_counter() - t_stream
        render_stats["combined"]["frames"] += 1
        last_frames["combined"] = output.combined
    if output_flags["minimap_ball"] and output.minimap_ball is not None:
        t_stream = perf_counter()
        writers["minimap_ball"].write(output.minimap_ball)
        render_stats["minimap_ball"]["time"] += perf_counter() - t_stream
        render_stats["minimap_ball"]["frames"] += 1
        last_frames["minimap_ball"] = output.minimap_ball
    if output_flags["minimap_player"] and output.minimap_player is not None:
        t_stream = perf_counter()
        writers["minimap_player"].write(output.minimap_player)
        render_stats["minimap_player"]["time"] += perf_counter() - t_stream
        render_stats["minimap_player"]["frames"] += 1
        last_frames["minimap_player"] = output.minimap_player
    if output_flags["heatmap_player"] and output.heatmap_player is not None:
        t_stream = perf_counter()
        writers["heatmap_player"].write(output.heatmap_player)
        render_stats["heatmap_player"]["time"] += perf_counter() - t_stream
        render_stats["heatmap_player"]["frames"] += 1
        last_frames["heatmap_player"] = output.heatmap_player
    if output_flags["heatmap_ball"] and output.heatmap_ball is not None:
        t_stream = perf_counter()
        writers["heatmap_ball"].write(output.heatmap_ball)
        render_stats["heatmap_ball"]["time"] += perf_counter() - t_stream
        render_stats["heatmap_ball"]["frames"] += 1
        last_frames["heatmap_ball"] = output.heatmap_ball
    if output.profiling:
        for key, value in output.profiling.items():
            combine_breakdown[key] += value

for writer in writers.values():
    writer.close()
timings.append(("combine_render", perf_counter() - start))
frame_accessor.release()

start = perf_counter()
for name, enabled in output_flags.items():
    if not enabled:
        continue
    frame = last_frames.get(name)
    if frame is None:
        continue
    print(f"Saving sample image for {name}...")
    write_image(frame, "results/", array=False, img_name=name)
timings.append(("sample_export", perf_counter() - start))

print("\n[Summary] Stage durations (seconds):")
for label, duration in timings:
    print(f"  - {label}: {duration:.2f}s")
if render_stats:
    print("  Render streams:")
    for name, stats in render_stats.items():
        print(f"    * {name}: {stats['frames']} frames, write time {stats['time']:.2f}s")
if combine_breakdown:
    print("  Combine breakdown:")
    for name, duration in sorted(combine_breakdown.items()):
        print(f"    * {name}: {duration:.2f}s")

print(f"{time.time() - start_time:.2f} seconds elapsed.")
