import os
import cv2
import argparse


parser = argparse.ArgumentParser(description='resize images')
parser.add_argument('--data_dir', help='path to data diirectory')
parser.add_argument('--output_dir', help='path to output directory')
parser.add_argument('--width', default=1242, type=int)
parser.add_argument('--height', default=375, type=int)
args = parser.parse_args()

def main():
    file_names = os.listdir(args.data_dir)
    file_names.sort()
    file_paths = [args.data_dir + '/' + file_name for file_name in file_names]

    width = args.width
    height = args.height
    print('WILL CHANGE TO SIZE(' + str(width) + ',' + str(height) + ')')

    ext = '.png'
    save_paths = [os.path.splitext(file_name)[0] for file_name in file_names]
    save_paths = [args.output_dir + '/' + file_name + ext for file_name in save_paths]

    for idx, file in enumerate(file_paths):
        img = cv2.imread(file)

        resized_img = cv2.resize(img, (width, height))

        cv2.imwrite(save_paths[idx], resized_img)

        if idx % 200 == 0:
            print('saved : ' + save_paths[idx])


if __name__ == "__main__":
    main()
