from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

# Mengimpor konfigurasi lapangan tenis
from config.tennis_court import TennisCourtConfiguration


def draw_court(
    config: TennisCourtConfiguration,
    background_color: sv.Color = sv.Color(58, 83, 164), # Biru lapangan tenis
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 2,
    scale: float = 0.4
) -> np.ndarray:
    """
    Menggambar lapangan tenis dengan dimensi, warna, dan skala tertentu.
    """
    scaled_length = int(config.length * scale)
    scaled_width = int(config.width * scale)

    # Membuat kanvas potrait (tinggi, lebar)
    court_image = np.ones(
        (scaled_length + 2 * padding,
         scaled_width + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Menggunakan config.vertices (17 titik) untuk menggambar
    drawing_vertices = config.vertices
    for start, end in config.edges:
        point1 = (int(drawing_vertices[start - 1][0] * scale) + padding,
                  int(drawing_vertices[start - 1][1] * scale) + padding)
        point2 = (int(drawing_vertices[end - 1][0] * scale) + padding,
                  int(drawing_vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=court_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
    
    return court_image


def draw_points_on_court(
    config: TennisCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.4,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Menggambar titik-titik di atas lapangan tenis.
    """
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        # Koordinat (x,y) dari xy sudah dalam mode potrait jika berasal dari model
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=court, center=scaled_point, radius=radius,
            color=face_color.as_bgr(), thickness=-1
        )
        cv2.circle(
            img=court, center=scaled_point, radius=radius,
            color=edge_color.as_bgr(), thickness=thickness
        )

    return court


def draw_paths_on_court(
    config: TennisCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.4,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Menggambar jejak (paths) di atas lapangan tenis.
    """
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=court, pt1=scaled_path[i], pt2=scaled_path[i + 1],
                color=color.as_bgr(), thickness=thickness
            )

    return court