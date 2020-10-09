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

def test(args):
    """ Function to test the architecture by saving disparities to the output directory
    """
    # Since it is clear post-processing is better in all runs I have done, I will only
    # save post-processed results. Unless explicitly stated otherwise.
    # Also for Pilzer, the disparities are already post-processed by their own FuseNet.
    do_post_processing = args.postprocessing and 'pilzer' not in args.architecture

    input_height = args.input_height
    input_width = args.input_width

    output_directory = args.output_dir
    n_img, test_loader = prepare_dataloader(args, 'test')

    model = create_architecture(args)
    which_model = 'final' if args.load_final else 'best'
    model.load_networks(which_model)
    model.to_test()

    disparities = np.zeros((n_img, input_height, input_width), dtype=np.float32)
    inference_time = 0.0

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 100 == 0 and i != 0:
                print('Testing... Now at image: {}'.format(i))

            t_start = time.time()
            # Do a forward pass
            disps = model.fit(data)
            # Some architectures output a single disparity, not a tuple of 4 disparities.
            disps = disps[0][:, 0, :, :] if isinstance(disps, tuple) else disps.squeeze()

            if do_post_processing:
                disparities[i] = post_process_disparity(disps.cpu().numpy())
            else:
                disp = disps.unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
            t_end = time.time()
            inference_time += (t_end - t_start)

    if args.test_time:
        test_time_message = 'Inference took {:.4f} seconds. That is {:.2f} imgs/s or {:.6f} s/img.'
        print(test_time_message.format(inference_time, (n_img / inference_time), 1.0 / (n_img / inference_time)))

    disp_file_name = 'disparities_{}_{}.npy'.format(args.dataset, model.name)
    full_disp_path = os.path.join(output_directory, disp_file_name)

    if os.path.exists(full_disp_path):
        print('Overwriting disparities at {}...'.format(full_disp_path))
    np.save(full_disp_path, disparities)
    print('Finished Testing')


def reconstruct_right(args):
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

    # Conver input PIL image to Tensor with transformation
    left_image = Image.open(input_left)
    resize = transforms.ResizeImage(train=False, size=(256, 512))
    totensor = transforms.ToTensor(train=False)
    left_image = totensor(resize(left_image))
    left_image = torch.stack((left_image, torch.flip(left_image, [2])))
    # Make dicctionary to feed model.fit()
    left_data = {'left_image': left_image}

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
    with torch.no_grad():
        # Estimate disparity
        disps = model.fit(left_data)
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disps]
        disp_right_est = disp_right_est[0]

        # Using estimated disparity, apply it to left view and obtain right view
        print('reconstructing right view from left view')
        fake_right = fake_loss.generate_image_right(left_image.to(args.device), disp_right_est)

        # convert Tensor(fake_right) to PIL image and save it!
        print('Saving reconstructed right view...')
        output_dir = os.path.dirname(input_left)
        output_name = os.path.splitext(os.path.basename(input_left))[0]
        model_name = os.path.basename(args.model_name)
        save_path = os.path.join(output_dir, '{}_rec_{}.jpg'.format(output_name, model_name))
        save_right = torchvision.transforms.functional.to_pil_image(fake_right[0].cpu())
        save_right.save(save_path)
        print('Saved image : ' + save_path)





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
