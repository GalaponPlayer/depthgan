import os
import argparse
import cv2
import pandas as pd

parser = argparse.ArgumentParser(description='resize kitti object dataset')
parser.add_argument('--data_dir', help='path to data diirectory')
parser.add_argument('--output_dir', help='path to output directory')
parser.add_argument('--res_dir', help='path to resolution file')
args = parser.parse_args()

def main():
    file_names = os.listdir(args.data_dir)
    file_names.sort()
    file_paths = [args.data_dir + '/' + file_name for file_name in file_names]

    res_data = pd.read_csv(args.res_dir, sep=',', header=None, index_col=0)

    ext='.png'
    save_paths = [os.path.splitext(file_name)[0] for file_name in file_names]
    save_paths = [args.output_dir + '/' + file_name + ext for file_name in save_paths]

    for idx, file in enumerate(file_paths):
        basename_png = os.path.splitext(os.path.basename(file))[0] + ext
        res = res_data.loc[basename_png]
        res = (res[1], res[2])

        img = cv2.imread(file)
        resized_image = cv2.resize(img, res) 

        cv2.imwrite(save_paths[idx], resized_image)

        if idx%200 == 0:
            print('resized : ' + save_paths[idx])

if __name__ == "__main__":
    main()
