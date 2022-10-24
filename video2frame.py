import cv2
import os

def config_parse():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--target_frame_path', type=str)

    args = parser.parse_args()
    return args

def generate_frames(args):
    if not os.path.exists(args.target_frame_path):
        os.makedirs(args.target_frame_path)
    
    cap= cv2.VideoCapture(args.video_path)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        tmp_frame_name = f'{i:08d}.png'
        tmp_frame_path = os.path.join(args.target_frame_path, tmp_frame_name)
        cv2.imwrite(tmp_frame_path,frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
    print('Successfully generated frames from video!')

if __name__ == '__main__':
    args = config_parse()

    generate_frames(args)
