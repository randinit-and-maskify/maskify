#Assumed folder structure
'''
/example_dir/
    ├── run_seg_to_binary.py        # Main script
    ├── images/                     # Folder containing .png images
    ├── labels.json                 # Labels file
    ├── output/                     # Folder to store output files
        ├── output_images/          # Folder to store output images with seg masks
        ├── output_binary/          # Folder to store output numpy binary files of seg masks
'''

#pip install --upgrade -q git+https://github.com/huggingface/transformers

#Import required libraries
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
from pathlib import Path

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import os
import json

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

torch.cuda.empty_cache()

#logger = get_logger(__name__)

#Code from Grounding DINO: https://github.com/IDEA-Research/Grounded-Segment-Anything

#Store detection results
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

#Draw detection results of Grounding DINO
def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    file_name: str,
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]

    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]

        shapes.append(shape)

    # Update layout
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
    )

    # Save the image to the output folder
    output_file_path = os.path.join("output_images", file_name)
    fig.write_image(output_file_path)

    print(f"Image saved to {output_file_path}")

#Utils
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

#SAM approach defined (Grounding DINO + SAM)
#Adjusted to detect batches
def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = "IDEA-Research/grounding-dino-tiny", 
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]
    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

#Added code by Helen Wang 

# Load pipelines/models if not set
detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"

class ImageDataset(Dataset):
    def __init__(self, image_folder: str, labels: List[str]):
        self.image_paths = sorted(list(Path(image_folder).glob("*.png")))
        self.labels = labels
        self.transform = transforms.ToTensor()
        # self.transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image(str(image_path))
        image_tensor = self.transform(image)
        image_name = image_path.stem
        return image_tensor, image_name
        #image_url = str(image_path)

        # with torch.no_grad():
        #     image_array, segmentation_mask, detections = grounded_segmentation(
        #         image=image_url,
        #         labels=self.labels,
        #         threshold=0.3,
        #         polygon_refinement=False,
        #         detector_id=detector_id,
        #         segmenter_id=segmenter_id
        #     )
        
        #     # Apply transformation to image_array
        #     if isinstance(image_array, np.ndarray):
        #         image_array = Image.fromarray(image_array)
        #     image_array = self.transform(image_array)
            
        #     return image_array, segmentation_mask, detections


def load_image(image_path: str) -> Image.Image:
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise ValueError(f"Error loading image: {image_path}")
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.open(image_path).convert("RGB")
    return image

#Process image with grounded segmentation
def grounded_segmentation(
    image: Image.Image,
    labels: List[str],
    threshold: float,
    polygon_refinement: bool,
    detector_id: Optional[str] = "IDEA-Research/grounding-dino-tiny",
    segmenter_id: Optional[str] = "facebook/sam-vit-base"
) -> Tuple[np.ndarray, np.ndarray, list]:
    print(f"Running grounded segmentation on image: {image}")
    # image_pil = load_image(image)

    #query_mask = query_mask.to(torch.float32)

    # Detect and segment
    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    # Create a binary segmentation mask
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for detection in detections:
        if detection.mask is not None:
            mask[detection.mask == 1] = 1  # Apply mask to binary mask

    return np.array(image), mask, detections

torch.set_default_dtype(torch.float32)

def run_batch_inference(
    image_folder: str, 
    current_dir: Path,
    labels: List[str], 
    batch_size: int, 
    threshold: float = 0.3, 
    polygon_refinement: bool = False, 
    detector_id: Optional[str] = "IDEA-Research/grounding-dino-tiny", 
    segmenter_id: Optional[str] = "facebook/sam-vit-base"
):
    
    # Set up output directories 
    output_dir = current_dir / "output"
    output_images_dir = output_dir / "output_images"
    output_binary_dir = output_dir / "output_binary"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_binary_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset and data loader
    dataset = ImageDataset(image_folder, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # # Load pipelines/models if not set
    # detector_id = detector_id if detector_id else "IDEA-Research/grounding-dino-tiny"
    # segmenter_id = segmenter_id if segmenter_id else "facebook/sam-vit-base"
    
    # for batch in dataloader:
    #     for image_array, segmentation_mask, detections, image_name in batch:
    #         if segmentation_mask is not None and segmentation_mask.size > 0:
    #             # Save binary mask as .npy file
    #             np.save(str(output_binary_dir / f"{image_name}_mask.npy"), segmentation_mask)
                
    #             # Optionally save annotated image
    #             annotated_image_path = output_images_dir / f"{image_name}_segmented.png"
    #             plot_detections(image_array, detections, save_name=str(annotated_image_path))

    #             del batch
    #             torch.cuda.empty_cache()

    for images, image_names in dataloader:
        for image_tensor, image_name in zip(images, image_names):
            image_pil = transforms.ToPILImage()(image_tensor)
            image_array, segmentation_mask, detections = grounded_segmentation(
                image=image_pil,
                labels=labels,
                threshold=threshold,
                polygon_refinement=polygon_refinement,
                detector_id=detector_id,
                segmenter_id=segmenter_id
            )
            np.save(output_binary_dir / f"{image_name}_mask.npy", segmentation_mask)
            annotated_image_path = output_images_dir / f"{image_name}_segmented.png"
            plot_detections(image_array, detections, save_name=str(annotated_image_path))
            torch.cuda.empty_cache()

    print("Batch inference completed.")

#Load labels from labels.json (provided by user)
def load_labels(labels_file: Path) -> List[str]:
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    return labels_data['labels']

def main() -> None:
    current_dir = Path().resolve() #Get user's current directory
    image_folder = current_dir / "images"
    labels_file = current_dir / "labels.json"
    batch_size = 4
    labels = load_labels(Path(labels_file))

    run_batch_inference(str(image_folder), current_dir, labels, batch_size=batch_size)

if __name__ == "__main__":
    main()