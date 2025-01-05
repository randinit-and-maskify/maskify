This pipeline takes an image, creates segmentation masks of specified objects, and outputs an image with the object overlayed on a random background. 
<center>
  <img src="https://github.com/user-attachments/assets/2aa7e6e9-50c1-45f9-adb1-4c476ab5703b" alt="original image" width="300" height="205">
  <img src="https://github.com/user-attachments/assets/ae5aaeae-0f61-4f57-9eb1-d5d7659a1496" alt="image with detections" width="305" height="210">
  <img src="https://github.com/user-attachments/assets/f6d3a5b1-8014-415f-b701-1154a5c8fd1c" alt="binary image" width="305" height="210">
  <img src="https://github.com/user-attachments/assets/154d671e-bbff-4d57-88b5-66631568bfaf" alt="overlay image 1" width="300" height="205">
  <img src="https://github.com/user-attachments/assets/14085928-f212-4fb7-aa1c-24248ac63008" alt="overlay image 2" width="300" height="205">
  <img src="https://github.com/user-attachments/assets/b85ab573-88d4-4efa-a368-9a33d703c429" alt="overlay image 3" width="300" height="205">  
</center>

# Overview
Given the following inputs from the user:
- the labels of objects to detect,
- image(s),
  
The pipeline creates segmentation masks based on the user's labels and outputs an image of specified objects overlayed on a randomized background. This action increases dataset size, enhancing a model's ability to generalize across diverse scenarios and improving its robustness to environmental variations.

# Installation
1. Clone the repository:
```
git clone https://github.com/hlenwng/maskify.git
```
2. Install dependencies with conda:
```
conda env create -f environment.yml
conda activate maskify
```

# Pipeline tutorial
1. Edit the example `labels.json` file with your list of object descriptions. 

   - For example, "small white ceramic mug."
2. Upload your image(s) into the `images` folder.
3. Run pipeline using:
```
python run_seg_to_binary.py
```
4. The pipeline will write 3 files:
```
/example_dir/output
    ├── output_images/img#_detections.png                     
    ├── output_binary/img#_mask.npy
    ├── output_overlay/img#_overlay.png
```
5. Optional: Use `check.ipynb` to visualize  `img#_mask.npy` binary file.

# Directory structure
```
/example_dir/
    ├── run_seg_to_binary.py        # Main script
    ├── labels.json                 # (Input) File containing object labels/descriptions
    ├── images/                     # (Input) Folder containing '.png' images
    ├── output/                     # (Output) Folder to store output files
        ├── output_images/          # (Output) Folder to store output images with seg masks
        ├── output_binary/          # (Output) Folder to store output '.npy' binary files of seg masks
        ├── output_overlay/         # (Output) Folder to store output images with random background
    ├── check.ipynb                 # (Output) File to visualize '.npy' files
```

# References
This project includes code adapted from Grounded SAM: Ren, Tianhe, et al. "Grounded sam: Assembling open-world models for diverse visual tasks." arXiv preprint arXiv:2401.14159 (2024). 
