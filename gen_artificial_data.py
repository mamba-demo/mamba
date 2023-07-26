import argparse
from utils.image_gen_utils import *
from tqdm import tqdm
import pandas as pd
import numpy as np

def gen_data(type, n_samples, save_prefix, mode='RGB', n_obj=100, d=360, w=360, h=360, size=24, filetype='tiff'):
    n_digits = len(str(int(n_samples)))
    if type == 'cells':
        for i in tqdm(range(n_samples)):
            cells_phantom = gen_3d_img(n_obj, d, w, h, size, 'cells', mode=mode)
            save_generated_img(cells_phantom, save_prefix+'_cells_'+str(i).zfill(n_digits), mode=mode, filetype=filetype)
    elif type == 'polygons':
        for i in tqdm(range(n_samples)):
            mixed_phantom = gen_3d_img(n_obj, d, w, h, size, 'polygons', mode=mode)
            save_generated_img(mixed_phantom, save_prefix+'_polygons_'+str(i).zfill(n_digits), mode=mode, filetype=filetype)
    elif type == 'spheres':
        for i in tqdm(range(n_samples)):
            sphere_phantom = gen_3d_img(n_obj, d, w, h, size, 'spheres', mode=mode)
            save_generated_img(sphere_phantom, save_prefix+'_spheres_'+str(i).zfill(n_digits), mode=mode, filetype=filetype)
    elif type == 'cubes':
        for i in tqdm(range(n_samples)):
            cube_phantom = gen_3d_img(n_obj, d, w, h, size, 'cubes', mode=mode)
            save_generated_img(cube_phantom, save_prefix+'_cubes_'+str(i).zfill(n_digits), mode=mode, filetype=filetype)


def create_csv(class_types, class_counts, prefixes, save_path):
    d = {}
    names = []
    classes = []
    class_nums = []
    for i in range(len(class_types)):
        c_t, count, prefix = class_types[i], class_counts[i], prefixes[i]
        n_digits = len(str(count))
        for j in range(count):
            name = prefix+'_'+c_t+'_'+str(j).zfill(n_digits)
            names.append(name)
            classes.append(c_t)
            class_nums.append(i)
    d['patient_id'] = names
    d['slide_id'] = [''] * sum(class_counts)
    d['type'] = classes
    d['label'] = class_nums
    df = pd.DataFrame(data=d)
    df.to_csv(save_path, header=True, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('save_dir_path', type=str, default='.',
                        help='Path to the directory in which to save the generated 3D images.')
    parser.add_argument('n_samples', type=int, default=1,
                        help='Number of 3D images to generate.')
    parser.add_argument('--prefix', type=str, default='phantom',
                        help='Prefix for saved images and image directory.')
    parser.add_argument('--n_obj', type=int, default=500,
                        help='Number of 3D images to generate.')
    parser.add_argument('--type', type=str, choices=['cells', 'spheres', 'cubes', 'polygons'], default='cells',
                        help='Chooses the type of objects to populate the images with.')
    parser.add_argument('--h', default=360, type=int,
                        help='Generated image height.')
    parser.add_argument('--w', default=360, type=int,
                        help='Generated image width.')
    parser.add_argument('--d', default=360, type=int,
                        help='Generated image depth.')
    parser.add_argument('--mode', default='RGB', type=str, choices=['L', 'RGB'],
                        help='Generated image mode (how colors are stored in data).')
    args = parser.parse_args()


    img_dir = args.save_dir_path
    if img_dir[-1] == '/':
        prefix = img_dir + args.prefix
    else:
        prefix = img_dir + '/' + args.prefix
    n_digits = len(str(int(args.n_samples)))

    gen_data(args.type, args.n_samples, prefix, args.mode, args.n_obj, args.d, args.w, args.h)

    print('done')