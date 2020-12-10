import os
import argparse
from PIL import Image


parser = argparse.ArgumentParser(description='resize images')
parser.add_argument('--data_dir', help='path to data diirectory')
parser.add_argument('--output_dir', help='path to output directory')
parser.add_argument('--ext_from', help='which extension you wanna change from', default='.jpg')
args = parser.parse_args()

def main():
    file_names = os.listdir(args.data_dir)
    file_names.sort()
    file_paths = [args.data_dir + '/' + file_name for file_name in file_names]

    ext = os.path.splitext(file_paths[1])[1]
#    if ext == '.png':
        #ext = '.jpg'
    #else:
        #ext = '.png'
    if args.ext_from == '.jpg':
        ext = '.png'
    else:
        ext = '.jpg'
    
    save_paths = [os.path.splitext(file)[0] for file in file_paths]
    save_paths = [path + ext for path in save_paths]

    for idx, name in enumerate(file_paths):
        img = Image.open(name)
        img.save(save_paths[idx])

        if idx % 200 == 0:
            print('saved ' + save_paths[idx])

if __name__ == "__main__":
    main()
