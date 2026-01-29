from typing import List

import numpy as np
import cv2
import os

def convert_mp4_and_save(frames: List[np.ndarray], vid_dir: str, vid_name: str, fps=30):
    height, width, _ = frames[0].shape

    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"{vid_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # frame_dir = os.path.join(vid_dir, f"frames_{vid_name}")
    # os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        # cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), frame_bgr)

    out.release()

    # frames = np.mean(frames, axis=0).astype(np.uint8)
    # frame_bgr = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(os.path.join(frame_dir, f"frame_all.png"), frame_bgr)

