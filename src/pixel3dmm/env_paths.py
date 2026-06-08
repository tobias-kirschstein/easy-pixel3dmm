import json
from pathlib import Path
from environs import Env

REPO_ROOT = f"{Path(__file__).parent.resolve()}/../.."
PACKAGE_ROOT = f"{Path(__file__).parent.resolve()}"

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/pixel3dmm/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)


with env.prefixed("PIXEL3DMM_"):
    CODE_BASE = env("CODE_BASE", REPO_ROOT)
    # PREPROCESSED_DATA = env("PREPROCESSED_DATA")
    # TRACKING_OUTPUT = env("TRACKING_OUTPUT")



head_template = f'{PACKAGE_ROOT}/assets/head_template.obj'
head_template_color = f'{PACKAGE_ROOT}/assets/head_template_color.obj'
head_template_ply = f'{PACKAGE_ROOT}/assets/test_rigid.ply'
VALID_VERTICES_WIDE_REGION = f'{PACKAGE_ROOT}/assets/uv_valid_verty_noEyes_debug.npy'
VALID_VERTS_UV_MESH = f'{PACKAGE_ROOT}/assets/uv_valid_verty.npy'
VERTEX_WEIGHT_MASK = f'{PACKAGE_ROOT}/assets/flame_vertex_weights.npy'
MIRROR_INDEX = f'{PACKAGE_ROOT}/assets/flame_mirror_index.npy'
EYE_MASK = f'{PACKAGE_ROOT}/assets/uv_mask_eyes.png'
FLAME_UV_COORDS = f'{PACKAGE_ROOT}/assets/flame_uv_coords.npy'
VALID_VERTS_NARROW = f'{PACKAGE_ROOT}/assets/uv_valid_verty_noEyes.npy'
VALID_VERTS = f'{PACKAGE_ROOT}/assets/uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy'
FLAME_ASSETS = f'{PACKAGE_ROOT}/src/pixel3dmm/preprocessing/MICA/data/'

# paths to pretrained pixel3dmm checkpoints
CKPT_UV_PRED = f"{Path.home()}/.cache/pixel3dmm/uv.ckpt"
CKPT_N_PRED = f"{Path.home()}/.cache/pixel3dmm/normals.ckpt"
# CKPT_UV_PRED = f'{CODE_BASE}/pretrained_weights/uv.ckpt'
# CKPT_N_PRED = f'{CODE_BASE}/pretrained_weights/normals.ckpt'