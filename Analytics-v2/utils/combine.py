import cv2
import numpy as np
from court_reference import CourtReference


MINIMAP_WIDTH = 166
MINIMAP_HEIGHT = 350
HEAT_INCREMENT = 5
HEAT_PERCENTILE = 98  # percentile that should map near red
HEAT_TARGET_DECAY = 0.985  # smoothing for dynamic scaling
HEAT_GAUSSIAN_SIGMA = 25
HEAT_RADIUS = 10
HEAT_ALPHA = 0.65
CONTOUR_LEVELS = [40, 90, 140, 190, 230]


def get_court_img():
    court_reference = CourtReference()
    court_img = court_reference.court.copy()
    court_img = cv2.dilate(court_img, np.ones((10, 10), dtype=np.uint8))
    return court_img


def _draw_ball(frame, ball_track, index, draw_trace, trace_len):
    if not ball_track[index][0]:
        return frame
    if draw_trace:
        for offset in range(trace_len):
            prev = index - offset
            if prev < 0 or not ball_track[prev][0]:
                continue
            draw_x, draw_y = map(int, ball_track[prev])
            frame = cv2.circle(frame, (draw_x, draw_y), radius=3, color=(0, 255, 0), thickness=2)
    else:
        bx, by = map(int, ball_track[index])
        frame = cv2.circle(frame, (bx, by), radius=5, color=(0, 255, 0), thickness=2)
        frame = cv2.putText(frame, "ball", (bx + 8, by + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
    return frame


def _draw_court_keypoints(frame, kps):
    if kps is None:
        return frame
    for kp in kps:
        x, y = int(kp[0, 0]), int(kp[0, 1])
        frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10)
    return frame


def _project_point(point, inv_mat, shape):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt = cv2.perspectiveTransform(pt, inv_mat)
    x = int(np.clip(pt[0, 0, 0], 0, shape[1] - 1))
    y = int(np.clip(pt[0, 0, 1], 0, shape[0] - 1))
    return x, y


def _blend_contour_map(base, heat_norm):
    if not np.any(heat_norm):
        return base.copy()
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    overlay = base.copy()
    mask = heat_norm > 0
    overlay[mask] = cv2.addWeighted(heat_color[mask], HEAT_ALPHA, overlay[mask], 1 - HEAT_ALPHA, 0)
    for level in CONTOUR_LEVELS:
        _, thresh = cv2.threshold(heat_norm, level, 255, cv2.THRESH_BINARY)
        contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            continue
        level_color = cv2.applyColorMap(np.full((1, 1), level, dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in level_color), 2)
    return overlay


def combine(frames, scenes, bounces, ball_track, homography_matrices, kps_court,
            persons_top, persons_bottom, draw_trace=False, trace=10):
    """
    :return
        imgs_res: list of resulting images
        imgs_ball: list of ball-only minimaps
        imgs_person: list of player-only minimaps
        imgs_heatmap: list of contour heatmaps
    """
    imgs_res, imgs_ball, imgs_person, imgs_heatmap = [], [], [], []
    is_track = [mat is not None for mat in homography_matrices]

    for scene_start, scene_end in scenes:
        tracked = is_track[scene_start:scene_end]
        if not tracked or sum(tracked) / (len(tracked) + 1e-15) <= 0.5:
            imgs_res.extend(frames[scene_start:scene_end])
            continue

        court_base = get_court_img()
        court_ball = court_base.copy()
        court_heat = court_base.copy()
        heatmap_accum = np.zeros(court_base.shape[:2], dtype=np.float32)
        heat_target = HEAT_INCREMENT * 15

        for idx in range(scene_start, scene_end):
            frame = frames[idx].copy()
            inv_mat = homography_matrices[idx]
            frame = _draw_ball(frame, ball_track, idx, draw_trace, trace)
            frame = _draw_court_keypoints(frame, kps_court[idx] if idx < len(kps_court) else None)

            if idx in bounces and inv_mat is not None and ball_track[idx][0]:
                ball_point = _project_point(ball_track[idx], inv_mat, court_base.shape[:2])
                cv2.circle(court_base, ball_point, radius=0, color=(0, 255, 255), thickness=50)
                cv2.circle(court_ball, ball_point, radius=0, color=(0, 255, 255), thickness=50)

            minimap = court_base.copy()
            minimap_ball = court_ball.copy()
            if minimap_ball.ndim == 2:
                minimap_ball = cv2.cvtColor(minimap_ball, cv2.COLOR_GRAY2BGR)
            imgs_ball.append(minimap_ball)

            persons = persons_top[idx] + persons_bottom[idx]
            court_person = get_court_img()
            for bbox, person_point in persons:
                if len(bbox) == 0 or inv_mat is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                px, py = _project_point(person_point, inv_mat, court_base.shape[:2])
                cv2.circle(minimap, (px, py), radius=0, color=(255, 0, 0), thickness=80)
                cv2.circle(court_person, (px, py), radius=0, color=(255, 0, 0), thickness=80)
                heatmap_accum = cv2.circle(heatmap_accum, (px, py), HEAT_RADIUS, HEAT_INCREMENT, -1)

            minimap_resized = cv2.resize(minimap, (MINIMAP_WIDTH, MINIMAP_HEIGHT))
            frame[30:30 + MINIMAP_HEIGHT, frame.shape[1] - 30 - MINIMAP_WIDTH:frame.shape[1] - 30] = minimap_resized
            imgs_res.append(frame)

            if court_person.ndim == 2:
                court_person = cv2.cvtColor(court_person, cv2.COLOR_GRAY2BGR)
            imgs_person.append(court_person)

            heat_blurred = cv2.GaussianBlur(heatmap_accum, (0, 0),
                                            sigmaX=HEAT_GAUSSIAN_SIGMA,
                                            sigmaY=HEAT_GAUSSIAN_SIGMA)
            heat_values = heat_blurred[heat_blurred > 0]
            if heat_values.size == 0:
                imgs_heatmap.append(court_heat.copy())
                continue
            percentile = np.percentile(heat_values, HEAT_PERCENTILE)
            heat_target = max(heat_target * HEAT_TARGET_DECAY, percentile, HEAT_INCREMENT)
            heat_norm = np.clip((heat_blurred / (heat_target + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
            contour_overlay = _blend_contour_map(court_heat, heat_norm)
            imgs_heatmap.append(contour_overlay)

    return imgs_res, imgs_ball, imgs_person, imgs_heatmap
