import platform
from pathlib import Path
from shutil import rmtree

import pixel3dmm.tracking.tracker
import torch
import tyro

import sys

from omegaconf import OmegaConf
from pixel3dmm import env_paths
from pixel3dmm.tracking.tracker import Tracker, get_intrinsics, get_extrinsics

sys.path.append(str(Path(__file__).parent))

from run_cropping import main as main_cropping
from run_mica import main_mica
from run_facer_segmentation import main as main_facer
from network_inference import main as main_network_inference

def project_points_screen_space(points3d, focal_length, principal_point, R_base, t_base, size : int = 512):
    # construct camera matrices
    intrinsics = get_intrinsics(focal_length, principal_point, size=size)
    w2c_openGL = get_extrinsics(R_base, t_base).repeat(focal_length.shape[0], 1, 1)

    B = points3d.shape[0]
    reps_extr = B if w2c_openGL.shape[0] == 1 else 1
    reps_intr = B if intrinsics.shape[0] == 1 else 1
    # apply w2c transformation
    lmk68_cam_space = torch.bmm(
        torch.cat([points3d, torch.ones_like(points3d[..., :1])], dim=-1),
        w2c_openGL.permute(0, 2, 1).repeat(reps_extr, 1, 1))

    # project from cam_space to screen_space
    lmk68_cam_space_prime = lmk68_cam_space[..., :3] / -lmk68_cam_space[..., [2]]
    lmk68_screen_space = (-1) * torch.bmm(lmk68_cam_space_prime, intrinsics.permute(0, 2, 1).repeat(reps_intr, 1, 1))[..., :2]
    lmk68_screen_space = torch.stack([size - 1 - lmk68_screen_space[..., 0], lmk68_screen_space[..., 1], lmk68_cam_space[..., 2]], dim=-1)
    return lmk68_screen_space


def main(video_path: str, preprocessing_folder: str, tracking_folder: str, /, cleanup: bool = False, is_discontinuous: bool = False):

    video_name = Path(video_path).stem

    preprocessing_folder2 = f"{preprocessing_folder}/{video_name}"
    tracking_folder2 = f"{tracking_folder}/{video_name}"

    # Preprocessing
    main_cropping(video_path, preprocessing_folder2, video_name='tracking')
    main_mica(preprocessing_folder2, 'tracking')
    main_facer(preprocessing_folder2, 'tracking')

    # Run Pixel3DMM UV and normal prediction models
    base_conf = OmegaConf.load(f'{env_paths.CODE_BASE}/configs/base.yaml')
    base_conf.video_name = 'tracking'
    base_conf.model.prediction_type = "normals"
    main_network_inference(preprocessing_folder2, base_conf)

    base_conf.model.prediction_type = "uv_map"
    main_network_inference(preprocessing_folder2, base_conf)

    # Run FLAME tracking
    cfg_tracker = OmegaConf.load(f'{env_paths.CODE_BASE}/configs/tracking.yaml')
    cfg_tracker.video_name = 'tracking'

    device_capability = torch.cuda.get_device_capability()
    if platform.system() != 'Linux' or device_capability[0] < 7:
        # Cannot compile on Windows
        pixel3dmm.tracking.tracker.COMPILE = False
        pixel3dmm.tracking.tracker.project_points_screen_space = project_points_screen_space  # Undo compilation

    if is_discontinuous:
        cfg_tracker.is_discontinuous = is_discontinuous

    tracker = Tracker(preprocessing_folder2, tracking_folder2, cfg_tracker)
    tracker.run()

    # Cleanup
    if cleanup:
        processing_folder = f"{preprocessing_folder2}/tracking"
        if Path(f"{processing_folder}/arcface").is_dir():
            rmtree(f"{processing_folder}/arcface")
        if Path(f"{processing_folder}/cropped").is_dir():
            rmtree(f"{processing_folder}/cropped")
        if Path(f"{processing_folder}/mica").is_dir():
            rmtree(f"{processing_folder}/mica")
        if Path(f"{processing_folder}/p3dmm_wGT").is_dir():
            rmtree(f"{processing_folder}/p3dmm_wGT")
        if Path(f"{processing_folder}/p3dmm_extraViz").is_dir():
            rmtree(f"{processing_folder}/p3dmm_extraViz")
        if Path(f"{processing_folder}/p3dmm/uv_map").is_dir():
            rmtree(f"{processing_folder}/p3dmm/uv_map")
        if Path(f"{processing_folder}/pipnet").is_dir():
            rmtree(f"{processing_folder}/pipnet")
        if Path(f"{processing_folder}/PIPnet_annotated_images").is_dir():
            rmtree(f"{processing_folder}/PIPnet_annotated_images")
        if Path(f"{processing_folder}/rgb").is_dir():
            rmtree(f"{processing_folder}/rgb")

        tr_folder = f"{tracking_folder2}/tracking"
        if Path(f"{tr_folder}_nV1_noPho_uv2000.0_n1000.0/mesh").is_dir():
            rmtree(f"{tr_folder}_nV1_noPho_uv2000.0_n1000.0/mesh")


if __name__ == '__main__':
    tyro.cli(main)
