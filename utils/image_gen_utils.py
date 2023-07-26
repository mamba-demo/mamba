import math
import numpy as np
import os
import glob
import scipy.stats as stats
import datetime
import pydicom
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

########################################
##########      3D Utils      ##########
########################################

def dist_origin(point):
    return math.dist(point, (0.0, 0.0, 0.0))

# returns polar ([0]) and azimuth ([1]) angles (in radians) between two points in 3D
def get_points_orientation(a, b):
    xy_dist = math.dist(a[0:2], b[0:2])
    return (math.atan2(b[1] - a[1],  b[0] - a[0]), math.atan2(b[2] - a[2], xy_dist))

def get_avs_orientation(angle_vec):
    xy_dist = math.sqrt(angle_vec[0]**2 + angle_vec[1]**2)
    return (math.atan2(angle_vec[1],  angle_vec[0]), math.atan2(angle_vec[2], xy_dist))

# Returns the unit vector pointing from one point (a) towards another (b)
def get_points_angle_vector(a, b):
    length = math.dist(a, b)
    return ((b[0] - a[0])/length, (b[1] - a[1])/length, (b[2] - a[2])/length)

# Returns 3D unit vector representation of an orientation defined by azimuthal and polar angles
# TODO: make clear what the angles refer to
def get_orients_angle_vector(orient):
    return (math.cos(orient[0])*math.cos(orient[1]), 
            math.sin(orient[0])*math.cos(orient[1]), 
            math.sin(orient[1]))

# Returns a unit vector that is perpendicular to the inputted angle vector
def gen_perpendicular_angle_vector(angle_vec):
    x, y, z = angle_vec
    d1 = (y*z + x*z, y*z - x*z, -y*y - x*x)
    d1_mag = dist_origin(d1)
    d2 = (-y*z - x*y, z*z + x*x, x*y - y*z)
    d2_mag = dist_origin(d2)
    if d1_mag > d2_mag:
        return (d1[0]/d1_mag, d1[1]/d1_mag, d1[2]/d1_mag)
    return (d2[0]/d2_mag, d2[1]/d2_mag, d2[2]/d2_mag)


# def find_ellipsoid_vals(x_val, f1_x, f2_x, y_val, f1_y, f2_y):
# density is in # of points checked per unit length, removing duplicates
def get_line(a, b, density=3.0):
    length = math.dist(a, b)
    n_pts = math.ceil(length * density)
    prev_pt = [a[0], a[1], a[2]]
    points = [prev_pt]
    for d in range(1, n_pts + 1):
        new_pt = [np.round(a[0] + d * (b[0] - a[0])/n_pts).astype(int),
                    np.round(a[1] + d * (b[1] - a[1])/n_pts).astype(int),
                    np.round(a[2] + d * (b[2] - a[2])/n_pts).astype(int)]
        if new_pt != prev_pt:
            points.append(new_pt)
            prev_pt = new_pt
    return points

# Computes the Hamilton product of two inputted quaternions
def quaternion_prod(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 + c1*a2 - b1*d2 + d1*b2
    d = a1*d2 + d1*a2 + b1*c2 - c1*b2
    return [a, b, c, d]

# angle_vec is the angles of the vector perpendicular to the plane that the circle inhabits
# density is in # of points checked per unit length, removing duplicates
def get_circle(center, r, angle_vec=(0.0, 0.0, 0.0), density=9.0):
    n_pts = math.ceil(density * math.pi * r * r)
    delta_theta = 2*math.pi/n_pts
    perp_ang = gen_perpendicular_angle_vector(angle_vec)
    # print(f'Original angle: {angle_vec}')
    # print(f'Perpendicular angle: {perp_ang}')
    points = []
    rot_pt = [0.0, perp_ang[0]*r, perp_ang[1]*r, perp_ang[2]*r]
    prev_pt = [round(center[0] + rot_pt[1]), round(center[1] + rot_pt[2]), round(center[2] + rot_pt[3])]
    # print(f'Center: {center}')
    # print(f'Init point: {prev_pt}')
    points.append(prev_pt)
    for i in range(1, n_pts):
        theta = delta_theta * i
        theta_s = math.sin(theta/2)
        theta_c = math.cos(theta/2)
        rot_quat = (theta_c, theta_s * angle_vec[0], theta_s * angle_vec[1], theta_s * angle_vec[2])
        inv_rot_quat = (rot_quat[0], -rot_quat[1], -rot_quat[2], -rot_quat[3])
        new_rot_pt = quaternion_prod(quaternion_prod(rot_quat, rot_pt), inv_rot_quat)
        new_pt_rounded = [round(center[0] + new_rot_pt[1]), round(center[1] + new_rot_pt[2]), round(center[2] + new_rot_pt[3])]
        # print(f'New point: {new_pt_rounded}')
        if new_pt_rounded != prev_pt:
            # line_seg = get_line(prev_pt, new_pt_rounded)
            # for pt in line_seg:
            #     points.append(pt)
            points.append(new_pt_rounded)
            prev_pt = new_pt_rounded
    return points

########################################
######   Cell Types/Structures   #######
########################################

# thickness goes inward from specified length
class SpheroidStructure():
    def __init__(self, angle_vec_fun, len_fun, ecc_fun, thickness_fun, val_fun):
        self.angle_vec_fun = angle_vec_fun
        self.len_fun = len_fun
        self.ecc_fun = ecc_fun
        self.thickness_fun = thickness_fun
        self.val_fun = val_fun

    def add_struct(self, img, center):
        a = self.len_fun() / 2
        # print(f'a: {a}')
        c = self.ecc_fun() * a
        # print(f'c: {c}')
        thickness = round(self.thickness_fun())
        # print(f'thickness: {thickness}')
        val = self.val_fun()
        angle_vec = self.angle_vec_fun()
        l, w, h = img.shape[0], img.shape[1], img.shape[2]
        # print(f'Img shape: {img.shape}')
        # print(f'Center: {center}')
        x_adj, y_adj, z_adj = angle_vec
        v1 = [center[0] + a * x_adj, center[1] + a * y_adj, center[2] + a * z_adj]
        v1 = np.trunc(v1).astype(int)
        v2 = [center[0] - a * x_adj, center[1] - a * y_adj, center[2] - a * z_adj]
        v2 = np.trunc(v2).astype(int)
        f1 = [center[0] + c * x_adj, center[1] + c * y_adj, center[2] + c * z_adj]
        f1 = np.trunc(f1).astype(int)
        f2 = [center[0] - c * x_adj, center[1] - c * y_adj, center[2] - c * z_adj]
        f2 = np.trunc(f2).astype(int)
        # print(f'v1: {v1}')
        # print(f'v2: {v2}')
        total_dist = math.dist(v1, f1) + math.dist(v1, f2)
        prin_ax = get_line(v1, v2, density=9.0)
        # print(f'Principle axis: {prin_ax}')
        points = []
        r0_count = 0
        for pt in prin_ax:
            d1 = math.dist(pt, f1)
            d2 = math.dist(pt, f2)
            numer_sq = 0.0
            if d1+d2 < total_dist:
                numer_sq = (d1**4 + d2**4 + total_dist**4 
                            - 2*(d1**2)*(d2**2) - 2*(d1**2)*(total_dist**2) - 2*(d2**2)*(total_dist**2)
                                    )
            # if numer_sq < 0: # Due to rounding issues
            #     print(numer_sq)
            #     numer_sq = 0.0
            # print(f'{numer_sq}: {d1}, {d2}, {total_dist}')
            r_max = round(math.sqrt(numer_sq)/(2*total_dist))
            if r_max <= 0:
                r0_count += 1
            points.append(pt)
            for r in range(max(1, r_max - thickness), r_max+1):
                circ_pts = get_circle(pt, r, angle_vec, 9.0)
                # print(f'Circle points: {circ_pts}')
                for circ_pt in circ_pts:
                    points.append(circ_pt)
        # print(f'Total points: {len(points)}')
        if r0_count > 4:
            print(f'a: {a}, c: {c}, thickness: {thickness}, val: {val}, av: {angle_vec}, r0_count: {r0_count}')
        for pt in points:
            if pt[0] >= 0 and pt[0] < l and pt[1] >= 0 and pt[1] < w and pt[2] >= 0 and pt[2] < h:
                img[pt[0]][pt[1]][pt[2]] = val
        return img


class CellType():
    def __init__(self, structures):
        self.structs = structures

    def add(self, img, center):
        for struct in self.structs:
            img = struct.add_struct(img, center)
        return img


########################################
####  Image Manipulation Functions  ####
########################################

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def add_sphere(img, diameter, center=None, val_fun=None):
    l, w, h = img.shape[0], img.shape[1], img.shape[2]
    if not center:
        center = (int(np.random.random_sample()*l), 
                    int(np.random.random_sample()*w), 
                    int(np.random.random_sample()*h))
    radius = diameter / 2
    r2 = radius**2
    x_min = int(max(0, center[0]-radius))
    x_max = int(min(l, center[0]+radius+1))
    for x in range(x_min, x_max):
        x_dist = (center[0]-x)**2
        y_rad = int(np.floor(np.sqrt(r2 - x_dist))) # skip values of y which lie outside the sphere given x
        y_min = int(max(0, center[1]-y_rad))
        y_max = int(min(w, center[1]+y_rad+1))
        for y in range(y_min, y_max):
            y_dist = (center[1]-y)**2
            z_rad = int(np.floor(np.sqrt(r2 - x_dist - y_dist))) # skip values of z which lie outside the sphere given x & y
            z_min = int(max(0, center[2]-z_rad))
            z_max = int(min(h, center[2]+z_rad+1))
            for z in range(z_min, z_max):
                value = None
                if val_fun == None:
                    value = True
                else:
                    value = val_fun()
                img[x][y][z] = value
    return img

def add_cube(img, side_length, center=None, val_fun=None):
    l, w, h = img.shape[0], img.shape[1], img.shape[2]
    if not center:
        center = (int(np.random.random_sample()*l), 
                    int(np.random.random_sample()*w), 
                    int(np.random.random_sample()*h))
    x_max = int(min(l, center[0] + side_length/2))
    x_min = int(max(0, center[0] - side_length/2))
    y_max = int(min(w, center[1] + side_length/2))
    y_min = int(max(0, center[1] - side_length/2))
    z_max = int(min(h, center[2] + side_length/2))
    z_min = int(max(0, center[2] - side_length/2))
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                value = None
                if val_fun == None:
                    value = True
                else:
                    value = val_fun()
                img[x][y][z] = value

    return img


def add_cell(img, cell_probs, cells, center=None):
    l, w, h = img.shape[0], img.shape[1], img.shape[2]
    if not center:
        center = (int(np.random.random_sample()*l), 
                    int(np.random.random_sample()*w), 
                    int(np.random.random_sample()*h))
    cell = np.random.choice(cells, p=cell_probs)
    img = cell.add(img, center)
    return img


def gen_3d_img(n_obj, d, w, h, size=32, item_type='sphere', location=None, mode='RGB'):
    if mode=='RGB':
        shape = (d, w, h, 3)
    elif mode=='L':
        shape = (d, w, h)
    art_img = np.zeros(shape, dtype=np.uint8)

    if item_type == 'spheres':
        if mode=='RGB':
            color_dist = create_dist_fun(['normal', 'normal', 'normal'], [100, 200, 100], [3, 3, 3], 0, 255)
        elif mode=='L':  
            color_dist = create_dist_fun(['normal'], [100], [9], 0, 255)
        for i in range(n_obj):
            art_img = add_sphere(art_img, diameter=size, center=location, val_fun=color_dist)
            print(f'  Added {i+1}/{n_obj} spheres...', end='\r')

    elif item_type == 'cubes':
        if mode=='RGB':
            color_dist = create_dist_fun(['normal', 'normal', 'normal'], [100, 100, 200], [3, 3, 3], 0, 255)
        elif mode=='L':  
            color_dist = create_dist_fun(['normal'], [150], [5], 0, 255)
        for i in range(n_obj):
            art_img = add_cube(art_img, side_length=size, center=location, val_fun=color_dist)
            print(f'  Added {i+1}/{n_obj} cubes...', end='\r')

    elif item_type == 'polygons':
        if mode=='RGB':
            color_dist_1 = create_dist_fun(['normal', 'normal', 'normal'], [200, 100, 100], [3, 3, 3], 0, 255)
            color_dist_2 = create_dist_fun(['normal', 'normal', 'normal'], [100, 200, 100], [3, 3, 3], 0, 255)
        elif mode=='L':  
            color_dist_1 = create_dist_fun(['normal'], [200], [9], 0, 255)
            color_dist_2 = create_dist_fun(['normal'], [100], [9], 0, 255)
        l, w, h = shape[0], shape[1], shape[2]
        off_region = (int(np.random.random_sample()*(l * 1/3)), 
                int(np.random.random_sample()*(w * 1/3)), 
                int(np.random.random_sample()*(h * 1/3)))
        for i in range(n_obj):
            location = (int(np.random.random_sample()*l), 
                        int(np.random.random_sample()*w), 
                        int(np.random.random_sample()*h))
            if (location[0] >= off_region[0] and location[0] < off_region[0]+l*2/3
                and location[1] >= off_region[1] and location[1] < off_region[1]+w*2/3
                and location[2] >= off_region[2] and location[2] < off_region[2]+h*2/3):
                art_img = add_cube(art_img, side_length=size, center=location, val_fun=color_dist_1)
            else:
                art_img = add_sphere(art_img, diameter=size, center=location, val_fun=color_dist_2)
            print(f'  Added {i+1}/{n_obj} polygons...', end='\r')

    elif item_type == 'cells':
        uniform_angle_dist = empty_comp_fun(get_orients_angle_vector, 
                                create_dist_fun(['uniform', 'uniform'], [0, 0], [2*math.pi, 2*math.pi]))
        if mode=='RGB':
            normal_gray = create_dist_fun(['normal', 'normal', 'normal'], [126, 126, 126], [3, 3, 3], 0, 255)
            high_var_gray = create_dist_fun(['normal', 'normal', 'normal'], [126, 126, 126], [9, 9, 9], 0, 255)
        elif mode=='L':
            normal_gray = create_dist_fun(['normal'], [126], [9], 0, 255)
            high_var_gray = create_dist_fun(['normal'], [126], [27], 0, 255)
        normal_length = create_dist_fun(['normal'], [size], [round(size/4)], 8, 40)
        abnormal_length = create_dist_fun(['normal'], [round(size*1.5)], [round(size/6)], 8, 40)
        normal_ecc = create_dist_fun(['normal'], [0.25], [0.2], 0, 0.9)
        abnormal_ecc = create_dist_fun(['normal'], [0.55], [0.2], 0, 0.9)
        uniform_thickness = create_dist_fun(['uniform'], [3], [0])
        reg_cell_membrane = SpheroidStructure(uniform_angle_dist, normal_length, normal_ecc, uniform_thickness, normal_gray)
        healthy_cell = CellType([reg_cell_membrane])
        irreg_cell_membrane = SpheroidStructure(uniform_angle_dist, abnormal_length, abnormal_ecc, uniform_thickness, high_var_gray)
        unhealthy_cell = CellType([irreg_cell_membrane])
        cells = [healthy_cell, unhealthy_cell]
        normal_region_probs = [0.9, 0.1]
        abnormal_region_probs = [0.65, 0.35]
        for i in range(n_obj):
            l, w, h = shape[0], shape[1], shape[2]
            location = (int(np.random.random_sample()*l), 
                        int(np.random.random_sample()*w), 
                        int(np.random.random_sample()*h))
            if i < n_obj/2:
                off_region = (int(np.random.random_sample()*(l * 1/3)), 
                    int(np.random.random_sample()*(w * 1/3)), 
                    int(np.random.random_sample()*(h * 1/3)))
                if (location[0] >= off_region[0] and location[0] < off_region[0]+l*2/3
                    and location[1] >= off_region[1] and location[1] < off_region[1]+w*2/3
                    and location[2] >= off_region[2] and location[2] < off_region[2]+h*2/3):
                    art_img = add_cell(art_img, abnormal_region_probs, cells, center=location)
                else:
                    art_img = add_cell(art_img, normal_region_probs, cells, center=location)
            else:
                art_img = add_cell(art_img, normal_region_probs, cells, center=location)
            print(f'  Added {i+1}/{n_obj} cells...', end='\r')

    else:
        print("Please specify whether to populate the image with spheres or cubes.")

    print('Phantom generated    '+'  '*len(str(n_obj)))
    return art_img

def save_generated_img(img, dirpath, mode='RGB', filetype='tiff'):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    prefix = os.path.basename(dirpath).split('_')[0]
    for x in range(img.shape[0]):
        img_slice = Image.fromarray(img[x], mode=mode)
        if filetype == 'dcm':
            filename = f'{dirpath}/{prefix}_{str(x).zfill(len(str(img.shape[0])))}.dcm'

            # Populate required values for file meta information
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
            file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
            file_meta.ImplementationClassUID = UID("1.2.3.4")

            ds = FileDataset(filename, {},
                            file_meta=file_meta, preamble=b"\0" * 128)
            ds.PixelData = img_slice.tobytes()

            # Add the data elements
            ds.PatientName = "Test^Firstname"
            ds.PatientID = "123456"

            # Set creation date/time
            dt = datetime.datetime.now()
            ds.ContentDate = dt.strftime('%Y%m%d')
            timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
            ds.ContentTime = timeStr

            # File information
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            ds.is_little_endian = True
            ds.is_implicit_VR = True

            # Pixel array information
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.Rows = img_slice.height
            ds.Columns = img_slice.width
            ds.SamplesPerPixel = 3 # RGB
            ds.PhotometricInterpretation = "RGB"
            ds.PixelRepresentation = 0 # unsigned ints
            ds.PlanarConfiguration = 0

            ds.save_as(filename)
        else:
            img_slice.save(f'{dirpath}/{prefix}_{str(x).zfill(len(str(img.shape[0])))}.{filetype}')
        write_metadata_template(dirpath+'/'+prefix+'_meta.dat')
    print(f'Saved {dirpath}')

def load_generated_img(dirpath):
    if not os.path.isdir(dirpath):
        print(dirpath+' does not exist!')
        return None
    sorted_filenames = sorted(glob.iglob(dirpath+'/*'))
    img = []
    for filename in sorted_filenames:
        filetype = filename.split('.')[-1]
        if filetype != 'dat':
            if filetype == 'dcm':
                img_slice = dcmread(filename).pixel_array
                img.append(img_slice)
            else:
                img_slice = Image.open(filename)
                img.append(np.asarray(img_slice))
    img = np.asarray(img)
    print(f'Loaded image shape: {img.shape}')
    return img

########################################
######   Other Helper Functions   ######
########################################

# Creates a function that returns the successive draws from distributions of specified type, location, and scale parameters.
def create_dist_fun(dist_types, locs, scales, min_val=None, max_val=None):
    dists = []
    for i in range(len(dist_types)):
        if dist_types[i]=='normal':
            dists.append(stats.norm(loc=locs[i], scale=scales[i]))
        elif dist_types[i]=='uniform':
            dists.append(stats.uniform(loc=locs[i], scale=scales[i]))
        else:
            print('Could not read distribution for azimuthal angle vector function! Defaulting to normal distribution.')
            dists.append(stats.norm(loc=locs[i], scale=scales[i]))

    def dist_fun():
        if len(dists) == 1:
            result = dists[0].rvs(1)[0]
            if min_val:
                result = max(result, min_val)
            if max_val:
                result = min(result, max_val)
            return result
        else:
            results = []
            for i in range(len(dists)):
                results.append(dists[i].rvs(1)[0])
            if min_val:
                results = np.fmax(results, np.repeat([min_val], len(results)))
            if max_val:
                results = np.fmin(results, np.repeat([max_val], len(results)))
            return results

    return dist_fun

def composite_fun(f, g):
    return lambda x : f(g(x))

def empty_comp_fun(f, g):
    return lambda : f(g())

def write_metadata_template(path):
    file = open(path, 'w')
    file.write('<?xml version="1.0"?>'
                '\n<RAWFileData>'
                '\n\t<Version>1</Version>'
                '\n\t<ObjectFileName>phantom_meta.raw</ObjectFileName>'
                '\n\t<Format>USHORT</Format>'
                '\n\t<DataSlope>1</DataSlope>'
                '\n\t<DataOffset>0</DataOffset>'
                '\n\t<Unit>Density</Unit>'
                '\n\t<Resolution X="3000" Y="3000" Z="3000" T="1" />'
                '\n\t<Spacing X="4.0e-06" Y="4.0e-06" Z="4.0e-06" />'
                '\n\t<Orientation X0="1" X1="0" X2="0" Y0="0" Y1="1" Y2="0" Z0="0" Z1="0" Z2="1" />'
                '\n\t<Position P1="0.0" P2="0.0" P3="0.0" />'
                '\n</RAWFileData>')
    file.close()