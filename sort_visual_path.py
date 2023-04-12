import sys
sys.path.append('..')
import os, shutil
import itertools
from PIL import Image
import torch
import numpy as np
import cv2
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

global device
device = "cuda" if torch.cuda.is_available() else "cpu"

## pip install einops
from einops import rearrange

## python -m pip install tsp_solver2
from tsp_solver.greedy import solve_tsp

## pip install lpips
import lpips
lpips_perceptor = lpips.LPIPS(net='alex').eval().to(device)    # lpips model options: 'squeeze', 'vgg', 'alex'

##### helper functions #####

def load_img(img_path, mode='RGB'):
    try:
        img = Image.open(img_path).convert(mode)
        return img
    except:
        print(f"Error loading image: {img_path}")
        return None

def get_centre_crop(img, aspect_ratio):
    h, w = np.array(img).shape[:2]
    if w/h > aspect_ratio:
        # crop width:
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        crop = img[:, left:right]
    else:
        # crop height:
        new_h = int(w / aspect_ratio)
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2
        crop = img[top:bottom, :]
    return crop


def resize(img, target_w, mode = "bilinear"):
    b,c,h,w = img.shape
    target_h = int(target_w * h / w)
    resized = torch.nn.functional.interpolate(img, size=(target_h, target_w), mode=mode)
    return resized

def prep_pt_img_for_clip(pt_img, clip_preprocessor):
    # This is a bit hacky and can be optimized, but turn the PyTorch img back into a PIL image, since that's what the preprocessor expects:
    pt_img = 255. * rearrange(pt_img, 'b c h w -> b h w c')
    pil_img = Image.fromarray(pt_img.squeeze().cpu().numpy().astype(np.uint8))

    # now, preprocess the image with the CLIP preprocessor:
    clip_img = clip_preprocessor(images=pil_img, return_tensors="pt")["pixel_values"].float().to(device)
    return clip_img

@torch.no_grad()
def perceptual_distance(img1, img2, resize_target_pixels_before_computing_lpips = 768):

    '''
    returns perceptual distance between img1 and img2
    This function assumes img1 and img2 are [0,1]
    By default, images are resized to a fixed resolution before computing the lpips score
    '''
    minv1, minv2 = img1.min().item(), img2.min().item()
    minv = min(minv1, minv2)
    if minv < 0:
        print("WARNING: perceptual_distance() assumes images are in [0,1] range.  minv1: %.3f | minv2: %.3f" %(minv1, minv2))

    if resize_target_pixels_before_computing_lpips > 0:
        img1, img2 = resize(img1, resize_target_pixels_before_computing_lpips), resize(img2, resize_target_pixels_before_computing_lpips)

    # lpips model requires image to be in range [-1,1]:
    perceptual_distance = lpips_perceptor((2*img1)-1, (2*img2)-1).mean().item()

    return perceptual_distance

def get_uniformly_sized_crops(imgs, target_n_pixels):
    """
    Given a list of images:
        - extract the best possible centre crop of same aspect ratio for all images
        - rescale these crops to have ~target_n_pixels
        - return resized images
    """

    # Load images
    imgs = [load_img(img_data, 'RGB') for img_data in imgs]
    assert all(imgs), 'Some images were not loaded successfully'
    imgs = [np.array(img) for img in imgs]
    
    # Get center crops at same aspect ratio
    aspect_ratios = [img.shape[1] / img.shape[0] for img in imgs]
    final_aspect_ratio = np.mean(aspect_ratios)
    crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs]

    # Compute final w,h using final_aspect_ratio and target_n_pixels:
    final_h = np.sqrt(target_n_pixels / final_aspect_ratio)
    final_w = final_h * final_aspect_ratio
    final_h, final_w = int(final_h), int(final_w)

    # Resize images
    resized_imgs = [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]
    resized_imgs = [Image.fromarray(img) for img in resized_imgs]
    
    return resized_imgs

################################################################################


def load_images(directory, target_size):
    images, image_paths = [], []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(directory, filename))
    
    images = get_uniformly_sized_crops(image_paths, target_size)

    # convert the images to tensors
    image_tensors = [ToTensor()(img).unsqueeze(0) for img in images]

    print(f"Loaded {len(images)} images from {directory}")
    return list(zip(image_paths, image_tensors))

def compute_pairwise_lpips(image_tensors):
    pairwise_distances = {}
    num_combinations = len(image_tensors) * (len(image_tensors) - 1) // 2
    progress_bar = tqdm(total=num_combinations, desc="Computing pairwise LPIPS")

    for img1, img2 in itertools.combinations(image_tensors, 2):
        dist = perceptual_distance(img1[1].to(device), img2[1].to(device))
        pairwise_distances[(img1[0], img2[0])] = dist
        pairwise_distances[(img2[0], img1[0])] = dist
        progress_bar.update(1)

    progress_bar.close()
    return pairwise_distances

def create_distance_matrix(pairwise_distances, filenames):
    num_images = len(filenames)
    distance_matrix = [[0 for _ in range(num_images)] for _ in range(num_images)]
    for i, img1 in enumerate(filenames):
        for j, img2 in enumerate(filenames):
            if i != j:
                distance_matrix[i][j] = pairwise_distances[(img1, img2)]
    return distance_matrix

def main(directory):
    paths_and_tensors = load_images(directory, args.target_n_pixels)
    filenames = [t[0] for t in paths_and_tensors]

    print(f"Computing {len(filenames)**2} pairwise perceptual distances. This may take a while..")
    pairwise_distances = compute_pairwise_lpips(paths_and_tensors)
    distance_matrix = create_distance_matrix(pairwise_distances, filenames)

    print("Solving traveling salesman problem...")
    path_indices = solve_tsp(distance_matrix, optim_steps=args.optim_steps, endpoints=None)
    path = [filenames[idx] for idx in path_indices]

    outdir = os.path.join(directory, "reordered")
    os.makedirs(outdir, exist_ok=True)

    print(f"Saving optimal visual walkthrough to {outdir}")
    for i, index in enumerate(path_indices):
        original_img_path = paths_and_tensors[index][0]
        image_pt_tensor = paths_and_tensors[index][1]
        new_name = f"{i:05d}_{os.path.basename(original_img_path)}.jpg"

        pil_image = ToPILImage()(image_pt_tensor.squeeze(0))
        pil_image.save(os.path.join(outdir, new_name))

        if args.copy_metadata_files:
            # replace extension with .json
            json_filepath = original_img_path
            for ext in args.image_extensions:
                json_filepath = json_filepath.replace(ext, ".json")

            if os.path.exists(json_filepath):
                print(f"Copying {json_filepath} to {outdir}")
                shutil.copy(json_filepath, os.path.join(outdir, new_name.replace(".jpg", ".json")))

    print(f"\nAll done! ---> {len(path_indices)} reordered images saved to {outdir}")

if __name__ == "__main__":
    '''
    This script takes a directory of images and computes the shortest visual path through them using Traveling Salesman Solver

    requires:
    pip install tsp_solver2 lpips

    example usage:
    python sort_visual_path.py /path/to/images

    This script computes all pairwise perceptual distances between images in the directory, and then solves the traveling salesman problem using the greedy algorithm.
    This means its runtime is quadratic with respect to the number of images in the directory. For large directories, this may be prohibitively slow.

    
    '''

    import argparse
    parser = argparse.ArgumentParser(description="Compute shortest visual path through images in a directory")
    parser.add_argument("directory", type=str, help="Directory containing images")
    parser.add_argument("--optim_steps", type=int, default=6, help="Number of tsp optimisation steps to run (will try to optimize the greedy tsp solution)")
    parser.add_argument("--image_extensions", type=str, default=".jpg,.png,.jpeg", help="Comma separated list of image extensions to consider")
    parser.add_argument("--copy_metadata_files", action="store_true", help="If set, will copy any metadata files (e.g. .json) to the output directory")
    parser.add_argument("--target_n_pixels", type=int, default=1024*1024, help="Target number of pixels for output images (script will resize and crop images)")
    args = parser.parse_args()
    main(args.directory)