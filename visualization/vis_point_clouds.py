#!/usr/bin/env python3
"""
Example using an example depth dataset from NYU.

https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import requests
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
from tqdm import tqdm
import pickle
import pdb

DESCRIPTION = """
# Dust3r Reconstruction
"""

# Display accumulated point cloud data
DISPLAY_ACCUMULATED_DATA = True

def log_data(pkl_data_path: Path) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    with open(pkl_data_path, "rb") as f:
        scene_list = pickle.load(f)

    len_scene_list = len(scene_list)
    print(f"Number of scenes: {len_scene_list}")

    pts3d_accumulated = None

    for idx, scene in enumerate(scene_list):
    # for time, f in files_with_timestamps:
        rr.set_time_sequence("frame_idx", idx)

        # extract data
        pts3d_0 = scene[3][0]
        # pts3d_1 = scene[3][1]
        img_0 = scene[0][0]
        # img_1 = scene[0][1]
        pose_0 = scene[2][0].detach().cpu().numpy()

        # only first image
        points = pts3d_0.reshape(-1, 3)
        colors = img_0.reshape(-1, 3)

        if DISPLAY_ACCUMULATED_DATA:
                # # * subsample the data in scene_list, (imgs, pts3d_list, mask) -> failed
            skip = 20
            # subsample these data into 1/skip
            points = points[::skip]
            colors = colors[::skip]

            if pts3d_accumulated is None:
                pts3d_accumulated = points
                colors_accumulated = colors
            else:
                pts3d_accumulated = np.concatenate((pts3d_accumulated, points), axis=0)
                colors_accumulated = np.concatenate((colors_accumulated, colors), axis=0)
            points = pts3d_accumulated
            colors = colors_accumulated

        # add the point cloud data to the log
        rr.log('world', rr.Points3D(points, colors=colors, radii=0.003))

        # Define the translation vector
        translation_vector = pose_0[:3, 3]
        # translation_vector = np.array([0, 0, -1])
        # pdb.set_trace()
        # log the camera transforms:
        rr.log(
            "world/camera/image",
            rr.Transform3D(translation=translation_vector,
                           mat3x3=pose_0[:3, :3],
                           ),
        )
        H, W = img_0.shape[:2]
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=min(H, W) * 1.1,
                # camera_xyz=rr.ViewCoordinates.RDF,
                # focal_length=min(H, W) * 0.5,
                # Intentionally off-center to demonstrate that we support it
                # principal_point=[0.45 * H, 0.55 * W],
            ),
        )

        # add image
        rr.log("world/camera/image/rgb", rr.Image(img_0))

        # break

def main() -> None:
    parser = argparse.ArgumentParser(description="Dust3r Reconstruction")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="",
        help="File name. Set path to the file in the code.",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(
        args,
        "rerun_example_rgbd",
        default_blueprint=rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.Spatial2DView(
                name="RGB",
                origin="world/camera/image",
            ),
            column_shares=[2, 1],
        ),
    )

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), timeless=True)

    # path of pickle file
    data_dir = "C:/Users/haruk/Box/Personal/MyDocs_box/留学/TUM/授業/SLAM/scene_list_data/"
    if args.file == "":
        # manual input
        pkl_data_path = Path(data_dir) / "scene_list_20210805_200654_8imgs.pkl"
    else:
        pkl_data_path = Path(data_dir) / args.file

    log_data(
        pkl_data_path=pkl_data_path,
    )

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
