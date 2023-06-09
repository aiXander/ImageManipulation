# ImageManipulation
A collection of various scripts for image manipulation
Required libraries: `torch, PIL, cv2, numpy`

## 1. sort_visual_path.py
  - grabs all images from a directory
  - applies the best fitting common crop to all the images
  - computes all pairwise perceptual distances using [lpips](https://github.com/richzhang/PerceptualSimilarity)
  - re-orders all the images using [tsp-solver](https://github.com/dmishin/tsp-solver) (Traveling Salesman)
  - saves resulting trajectory to the subdirectory input_dir/reordered
  
usage example:

  `python sort_visual_path.py /path/to/img_dir`
