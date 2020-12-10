import os
import argparse
import cv2
import pandas as pd

parser = argparse.ArgumentParser(description='blur kitti object dataset')
parser.add_argument('--data_dir', help='path to data diirectory')
parser.add_argument('--output_dir', help='path to output directory')
parser.add_argument('--res_dir', help='path to resolution file')
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--height', type=int, default=256)
args = parser.parse_args()

def main():
    file_names = os.listdir(args.data_dir)
    file_names.sort()
    file_paths = [args.data_dir + '/' + file_name for file_name in file_names]

    save_paths = [args.output_dir + '/' + file_name for file_name in file_names]

    res = (args.width, args.height)
    for idx, file in enumerate(file_paths):

        img = cv2.imread(file)
        input_height, input_width, _ = img.shape
        input_res = (input_width, input_height)
        resized_image = cv2.resize(img, res)
        re_resized_image = cv2.resize(resized_image, input_res)

        cv2.imwrite(save_paths[idx], re_resized_image)

        if idx%200 == 0:
            print('blurred : ' + save_paths[idx])

if __name__ == "__main__":
    main()        
