import time
import torch
import torchvision
import os

# Project imports
from utils import *
from options import MainOptions
from data_loader import prepare_dataloader, transforms
from architectures import create_architecture

# kishida imports
from PIL import Image
from losses.monodepth_loss import MonodepthLoss


def reconstruct_right(args):
    """ Function to reconstruct the right view of stereo pairs from left view
    """
    # Since it is clear post-processing is better in all runs I have done, I will only
    # save post-processed results. Unless explicitly stated otherwise.
    # Also for Pilzer, the disparities are already post-processed by their own FuseNet.
    do_post_processing = args.postprocessing and 'pilzer' not in args.architecture

    input_height = args.input_height
    input_width = args.input_width

    output_directory = args.output_dir

    file_names = os.listdir(args.data_dir)
    file_names.sort()
    file_names = [args.data_dir + '/' + file_name for file_name in file_names]

    # Create model
    model = create_architecture(args)
    which_model = 'final' if args.load_final else 'best'
    model.load_networks(which_model)
    model.to_test()

    # Make Fake loss module to use monodepthloss.generate_image_right
    # Can use it with "import MonodepthLoss"  and MonodepthLoss.generate_image...?
    fake_loss = MonodepthLoss(args)
    fake_loss = fake_loss.to(args.device)

    for idx, left in enumerate(file_names):
        left_image = Image.open(left)
        input_size = (left_image.width, left_image.height)
        resize = transforms.ResizeImage(train=False, size=(input_height, input_width))
        totensor = transforms.ToTensor(train=False)
        left_image = totensor(resize(left_image))
        left_image = torch.stack((left_image, torch.flip(left_image, [2])))
        # Make dicctionary to feed model.fit()
        left_data = {'left_image': left_image}

        # used in test time, wrapping `forward` in no_grad() so we don't save
        # intermediate steps for backprop
        with torch.no_grad():
            # Estimate disparity
            disps = model.fit(left_data)
            disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disps]
            disp_right_est = disp_right_est[0]

            # Using estimated disparity, apply it to left view and obtain right view
            fake_right = fake_loss.generate_image_right(left_image.to(args.device), disp_right_est)

            # convert Tensor(fake_right) to PIL image and save it!
            output_name = os.path.splitext(os.path.basename(left))[0]
            save_path = os.path.join(output_directory, '{}.png'.format(output_name))
            save_right = torchvision.transforms.functional.to_pil_image(fake_right[0].cpu())
            save_right = save_right.resize(input_size)
            save_right.save(save_path)

            if idx % 200 == 0:
                print('Processed ' + save_path)





def main():
    parser = MainOptions()
    args = parser.parse()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'rec_right':
        reconstruct_right(args)
    elif args.mode == 'verify-data':
        from utils.reduce_image_set import check_if_all_images_are_present
        check_if_all_images_are_present('kitti', args.data_dir)
        check_if_all_images_are_present('eigen', args.data_dir)
        check_if_all_images_are_present('cityscapes', args.data_dir)


if __name__ == '__main__':
    main()

    # Do an Arnold.
    print("YOU ARE TERMINATED!")
