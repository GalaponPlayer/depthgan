import torch
import torchvision
import os
import subprocess
# Project imports
from utils import *
from options import MainOptions
from data_loader import prepare_dataloader, transforms
from architectures import create_architecture
from PIL import Image
import cv2
from losses.monodepth_loss import MonodepthLoss

def check_reconstruct_right(args):
    """ Function to reconstruct the right view of stereo pairs from left view
    """
    # Since it is clear post-processing is better in all runs I have done, I will only
    # save post-processed results. Unless explicitly stated otherwise.
    # Also for Pilzer, the disparities are already post-processed by their own FuseNet.
    do_post_processing = args.postprocessing and 'pilzer' not in args.architecture

    input_height = args.input_height
    input_width = args.input_width
    input_left = args.left_view

    output_directory = args.output_dir
    # n_img, test_loader = prepare_dataloader(args, 'test')
    # Create model
    model = create_architecture(args)
    which_model = 'final' if args.load_final else 'best'
    model.load_networks(which_model)
    model.to_test()

    # Make Fake loss module to use monodepthloss.generate_image_right
    # Can use it with "import MonodepthLoss"  and MonodepthLoss.generate_image...?
    fake_loss = MonodepthLoss(args)
    fake_loss = fake_loss.to(args.device)

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop

    file_names = ['000025.png', '000031.png', '000036.png', '000049.png']
    file_names = ['~/depthgan/sample_kitti_obj/' + file_name for file_name in file_names]

    for filename in file_names:
        # Conver input PIL image to Tensor with transformation
        left_image = Image.open(filename)
        input_size = (left_image.width, left_image.height)
        resize = transforms.ResizeImage(train=False, size=(256, 512))
        totensor = transforms.ToTensor(train=False)
        left_image = totensor(resize(left_image))
        left_image = torch.stack((left_image, torch.flip(left_image, [2])))
        # Make dicctionary to feed model.fit()
        left_data = {'left_image': left_image}


        with torch.no_grad():
            # Estimate disparity
            disps = model.fit(left_data)
            disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disps]
            disp_right_est = disp_right_est[0]

            # Using estimated disparity, apply it to left view and obtain right view
            print('reconstructing right view from left view')
            fake_right = fake_loss.generate_image_right(left_image.to(args.device), disp_right_est)

            # convert Tensor(fake_right) to PIL image.
            output_dir = os.path.dirname(filename)
            output_name = os.path.splitext(os.path.basename(filename))[0]
            model_name = os.path.basename(args.model_name)
            save_path = os.path.join(output_dir, '{}_rec_{}.jpg'.format(output_name, model_name))
            save_right = torchvision.transforms.functional.to_pil_image(fake_right[0].cpu())
            save_right.resize(input_size)
            save_right.save(save_path)
            print('Saved image : ' + save_path)
            arguments = ['display', save_path]
            subprocess.call(arguments)

def main():
    parser = MainOptions()
    args = parser.parse()

    check_reconstruct_right(args)

if __name__ == "__main__":
    main()

    print('U R TERMINATED!')



