import argparse
import napari
import sys
import os
from napari_animation import Animation
import numpy as np
from PIL import Image
from preprocess.wsi_core.img_utils import read_img, clip_and_normalize_img

parser = argparse.ArgumentParser(description="Run an experiment")
parser.add_argument('img_dir_path', type=str, default='.',
                    help='Path to the image slices forming a 3D image.')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--rgb', action='store_true', default=False,
                    help='If included, the image will be read as an RGB image.')
parser.add_argument('--snapshot', action='store_true', default=False,
                    help='If included, all images from the supplied directory will be loaded and a snapshot of their '
                    '3D volume taken at a slanted angle will be generated for each one.')
parser.add_argument('--mode', type=str, default='OTLS', choices=['OTLS', 'CT'],
                    help='The image type (which affects the image shape and resulting camera angles for snapshots).')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='If included, the image will be clipped between top/bottom 1 percent intensity values and '
                    'normalized within the range between.')
parser.add_argument('--resolution', default=1.0, type=float,
                    help='The resolution of the image, with 1.0 indicating full resolution and 0.5 indicating a '
                    'resized image of (width*0.5, height*0.5, depth*0.5)')
parser.add_argument('--reduce_factor', default=1.0, type=float,
                    help='Same effect as --resolution, but specifying the factor by which to reduce the image size. '
                    'Inputting 2.0 gives a resized image of (width/2, height/2, depth/2)')
parser.add_argument('--animation_mode', action='store_true', default=False,
                    help='If included, the Napari console will include options for capturing keyframes for an animation.')
parser.add_argument('--black_thresh', default=-1.0, type=float,
                    help='The black threshold used when reading and displaying the image, if -1 (default) then no thresholding is used.')
args = parser.parse_args()

def make_translucent(img_arr, translucent_rgb=[0,0,0]):
    print(f'img_arr shape: {np.shape(img_arr)}')
    # Add alpha channel
    img_arr = np.pad(img_arr, [(0, 0), (0, 0), (0, 0), (0, 1)], mode='constant', constant_values=255)
    print(f'Expanded img_arr shape: {np.shape(img_arr)}')
    img_arr = np.where(img_arr == translucent_rgb+[255], [0]*4, img_arr)
    print(f'Processed img_arr shape: {np.shape(img_arr)}')
    return img_arr

if __name__ == '__main__':
    res = 1.0
    if args.resolution != 1.0:
        res = args.resolution
    elif args.reduce_factor != 1.0:
        res = 1.0 / args.reduce_factor

    img_arr, _ = read_img(args.img_dir_path, args.black_thresh, resolution=res)
    img_shape = np.shape(img_arr)
    print(f'Image shape: {img_shape}')
    
    if args.normalize:
        img_temp = img_arr.flatten()
        img_temp.sort()
        clip_min_adaptive = img_temp[len(img_temp) // 100]
        clip_max_adaptive = img_temp[-len(img_temp) // 100]
        img_arr = clip_and_normalize_img(img_arr, clip_min=clip_min_adaptive, clip_max=clip_max_adaptive)
        img_arr = (img_arr*256).astype(np.uint8)
        print("Image normalized!")

    viewer = napari.Viewer(ndisplay=3)
    interp = "nearest"
    # interp='bicubic'
    render = "attenuated_mip"
    if args.rgb == True:
        render = "translucent"
    blend = "additive"

    if len(img_shape) == 4:
        if img_shape[3] == 3:
            viewer.add_image(img_arr, channel_axis=3, colormap=["red", "green", "blue"], interpolation=[interp]*3, 
                            rendering=[render]*3, name=["red", "green", "blue"], blending = [blend]*3)
        else:
            viewer.add_image(img_arr, channel_axis=3, rgb=False, interpolation=interp, rendering=render, blending=blend)
    elif len(img_shape) == 3:
        viewer.add_image(img_arr, rgb=False, interpolation=interp, rendering=render, blending=blend)
    else:
        sys.exit('Could not read image due to invalid image shape!')
    initial_zoom = viewer.camera.zoom

    if args.img_dir_path[-1] == '/': 
        args.img_dir_path = args.img_dir_path[:-1]

    if args.snapshot:
        img_name = args.img_dir_path.split('/')[-1]
        # print(img_name)

        OTLS_angles = [
        #               (90, 45, 45),
                       (-30, 30, -50),
        #                (-90, -45, -45),
                       # (150, -30, 50),
                       # (90, 45, 225),
                       # (-30, 30, 130),
                       # (-90, -45, 135),
                       # (150, -30, 230)
                       ]

        CT_angles = [(-25, 32, -42)]
        # CT_angles = [(30, -30, -45), (30, -30, 45), (30, -30, 135), (30, -30, -135),
        #                 (-150, 30, -45), (-150, 30, 45), (-150, 30, 135), (-150, 30, -135)]

        angles = []
        if args.mode == 'OTLS':
            angles = OTLS_angles
            # viewer.camera.zoom = initial_zoom * 1.6
            viewer.camera.zoom = initial_zoom * 0.6
        elif args.mode == 'CT':
            angles = CT_angles
            viewer.camera.zoom = initial_zoom * 0.75
        else:
            print(f"Unsupported image type: {args.mode}")
        best_img_arr = None
        best_avg_intensity = 300.0
        best_idx = 0
        for idx, angle in enumerate(angles):
            viewer.camera.angles = angle
            # snapshot_arr = viewer.screenshot()
            snapshot_arr = viewer.screenshot(path=None)
            snapshot_avg_intensity = np.mean(snapshot_arr)
            if idx == 0 or snapshot_avg_intensity < best_avg_intensity:
                best_img_arr = snapshot_arr
                best_avg_intensity = snapshot_avg_intensity
                best_idx = idx

        if args.save_dir is None:
            save_path = args.img_dir_path.split('/'+img_name)[0] + "/"+img_name+"_snapshot.png"
        else:
            save_path = os.path.join(args.save_dir, '{}_snapshot.png'.format(img_name))

        if args.rgb:
            shape = np.shape(best_img_arr)
            # for x in range(shape[0]):
            #     for y in range(shape[1]):
            #         if best_img_arr[x,y].tolist() == [0,0,0,255]:
            #             best_img_arr[x,y] = np.array([255, 255, 255, 255])
        best_img = Image.fromarray(best_img_arr)
        best_img.save(save_path)
        print(f"Saved snapshot at: {save_path}")
    else:
        if args.animation_mode:
            animation = Animation(viewer)
            viewer.update_console({'animation': animation})
        napari.run()
