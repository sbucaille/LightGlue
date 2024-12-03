import torch
from transformers import AutoImageProcessor, AutoModelForKeypointDetection
from lightglue import LightGlue
import requests
from PIL import Image


detector = AutoModelForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
detector = detector.eval().to("cuda")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")

# Load and preprocess data
url_image1 = "https://i.etsystatic.com/8026687/r/il/e16641/1436085510/il_fullxfull.1436085510_8o7t.jpg"
url_image2 = "https://hips.hearstapps.com/hmg-prod/images/paris-skyline-with-eiffel-tower-on-a-sunny-day-wide-royalty-free-image-1722542465.jpg"

image1 = Image.open(requests.get(url_image1, stream=True).raw)
image2 = Image.open(requests.get(url_image2, stream=True).raw)

# pre process input pairs of images
images = [image1, image2]
inputs = processor(images, return_tensors="pt")
inputs = inputs.to("cuda")
outputs = detector(**inputs)
image_sizes = [(image.height, image.width) for image in images]
outputs = processor.post_process_keypoint_detection(outputs, image_sizes)

configs = {
        "depth_confidence": -1,
        "width_confidence": -1,
    }
inputs = {
    "image0": outputs[0],
    "image1": outputs[1],
}
inputs["image0"]["image_size"]
matcher = LightGlue(features="superpoint", flash=False, **configs)
pred = matcher(inputs)

print(pred)