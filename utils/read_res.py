import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='read image resolutions')
parser.add_argument('--data_dir')
parser.add_argument('--output_dir')
args = parser.parse_args()


def main():
    file_names = os.listdir(args.data_dir)
    file_names.sort()
    file_paths = [args.data_dir + '/' + file_name for file_name in file_names]

    with open(args.output_dir, 'w') as writer:
        for idx, file in enumerate(file_paths):
            img = Image.open(file)
            row = os.path.basename(file) + ',' + str(img.size[0]) + ',' + str(img.size[1]) + '\n'
            writer.write(row)

            if idx%200 == 0:
                print('read ' + row)

if __name__ == "__main__":
    main()

