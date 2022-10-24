import cv2
import os

def config_parse():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True)
    parser.add_argument('--original_frames', type=str)
    parser.add_argument('--height_coeff', type=float)
    parser.add_argument('--width_coeff', type=float)

    args = parser.parse_args()
    return args

def shrink_video_size(args):
    

if __name__ == '__main__':
    args = config_parse()

    shrink_video_size(args)
