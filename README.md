# ImageManipulation
A collection of various scripts for image manipulation

## 1. compute_shortest_visual_path.py
  - grabs all images from a directory
  - applies the best fitting common crop to all the images
  - computes all pairwise perceptual distances using lpips [https://github.com/richzhang/PerceptualSimilarity]
  - re-orders all the images using tsp-solver (Traveling Salesman)
  - saves resulting trajectory to subdir /reordered
  
usage example:
''python compute_shortest_visual_path.py /path/to/images''
