import os
import shutil
from roboflow import Roboflow

# Read the Roboflow key from the environment; never commit it.
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first.")

print("1. Initializing Roboflow...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("keio-dba-team").project("crowdhuman-nur7g")

print("\n2. Downloading dataset...")
print("(This is an 8GB+ dataset. It will take a few minutes depending on your internet connection.)")
dataset = project.version(1).download("yolov5")

print(f"\n3. Download finished! Dataset located temporarily at: {dataset.location}")
print("4. Reorganizing folders to match our YOLOv5 structure...")

source_dir = dataset.location
dest_dir = os.path.join("datasets", "crowdhuman")

os.makedirs(dest_dir, exist_ok=True)

# Roboflow creates 'train', 'valid', and 'test' folders.
# Each contains 'images' and 'labels' folders. We will move them to our structure.
for split in ["train", "valid", "test"]:
    split_src = os.path.join(source_dir, split)
    
    if os.path.exists(split_src):
        # Map 'valid' from Roboflow to 'val' to match the standard YOLOv5 format
        split_dest_name = "val" if split == "valid" else split
        
        # move images
        src_images = os.path.join(split_src, "images")
        if os.path.exists(src_images):
            dest_images = os.path.join(dest_dir, "images", split_dest_name)
            os.makedirs(dest_images, exist_ok=True)
            for f in os.listdir(src_images):
                shutil.move(os.path.join(src_images, f), os.path.join(dest_images, f))
        
        # move labels
        src_labels = os.path.join(split_src, "labels")
        if os.path.exists(src_labels):
            dest_labels = os.path.join(dest_dir, "labels", split_dest_name)
            os.makedirs(dest_labels, exist_ok=True)
            for f in os.listdir(src_labels):
                shutil.move(os.path.join(src_labels, f), os.path.join(dest_labels, f))

# Copy the yaml file which contains the class names
yaml_src = os.path.join(source_dir, "data.yaml")
if os.path.exists(yaml_src):
    shutil.copy(yaml_src, os.path.join(dest_dir, "data.yaml"))

print("5. Cleaning up temporary files...")
shutil.rmtree(source_dir)

print("\n🎉 Success! CrowdHuman is completely downloaded and set up for your EDA Analysis and Training Pipeline!")
