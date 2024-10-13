# Maskify
This pipeline takes an image, creates segmentation masks of specified objects, and outputs the data as a binary .npy file.

# Overview
Given inputs from the user:
- the labels of objects to detect,
- image(s),

The pipeline creates segmentation masks based on the user's labels and outputs the data as a binary .npy file, where each pixel value represents either the background (0) or the object (1).

# Installation
1. Clone the repository
```
git clone https://github.com/hlenwng/maskify.git
```
2. Install dependencies with conda
```
conda env create -f conda/environment.yml
conda activate maskify
pip install -e .
```

# Pipeline tutorial
1. Edit the .json file with your list of object descriptions. 
- For example, "small white ceramic mug."
2. Upload your image(s) into the `images` folder
3. Run pipeline using:
```
python run_seg_to_binary.py
```
