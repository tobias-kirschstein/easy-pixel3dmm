import argparse
import os
import sys

import tyro

import pixel3dmm.preprocessing.MICA
sys.path.append(pixel3dmm.preprocessing.MICA.__path__[0])  # Ensure MICA packages are importable

import numpy as np
from pixel3dmm.preprocessing.MICA.configs.config import get_cfg_defaults
from pixel3dmm.preprocessing.MICA.demo import main

def main_mica(preprocessing_folder: str, video_name: str, /):
    # parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    # parser.add_argument('-video_name', required=True, type=str)
    # parser.add_argument('-a', default='demo/arcface', type=str, help='Processed images for MICA input')
    # parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')

    # args = parser.parse_args()
    args = argparse.Namespace()
    cfg = get_cfg_defaults()
    args.i = f'{preprocessing_folder}/{video_name}/cropped/'
    args.o = f'{preprocessing_folder}/{video_name}/mica/'
    args.a = f'{preprocessing_folder}/{video_name}/arcface/'
    args.m = f"{pixel3dmm.preprocessing.MICA.__path__[0]}/data/pretrained/mica.tar"
    if os.path.exists(f'{preprocessing_folder}/{video_name}/mica/'):
        if len(os.listdir(f'{preprocessing_folder}/{video_name}/mica/')) >= 10:
            print(f'''
                                <<<<<<<< ALREADY COMPLETE MICA PREDICTION FOR {video_name}, SKIPPING >>>>>>>>
                                ''')
            return

    # Changed!
    np.Inf = np.inf

    main(cfg, args)

if __name__ == '__main__':
    tyro.cli(main_mica)