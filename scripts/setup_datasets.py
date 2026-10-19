import os
import urllib.request
import zipfile
import shutil

# Root directory for datasets
DATASETS_DIR = "datasets"

# Define the folder structure
FOLDERS = [
    "coco/images/train2017", "coco/images/val2017", "coco/images/test2017",
    "coco/labels/train2017", "coco/labels/val2017", "coco/labels/test2017",
    "crowdhuman/images/train", "crowdhuman/images/val",
    "crowdhuman/labels/train", "crowdhuman/labels/val",
    "caltech/images/train", "caltech/images/val",
    "caltech/labels/train", "caltech/labels/val",
    "mot/images/train", "mot/images/val",
    "mot/labels/train", "mot/labels/val",
]

def create_structure():
    print("Creating YOLO-compatible folder structure...")
    for folder in FOLDERS:
        path = os.path.join(DATASETS_DIR, folder)
        os.makedirs(path, exist_ok=True)
    print("✅ Folder structure created.\n")

def download_and_extract(url, extract_to):
    file_name = url.split('/')[-1]
    file_path = os.path.join(DATASETS_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name} from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"✅ Downloaded {file_name}.")
    else:
        print(f"✅ {file_name} already exists.")
        
    print(f"Extracting {file_name}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}.\n")

def setup_mot():
    print("--- Setting up MOT17 and MOT20 ---")
    mot_dir = os.path.join(DATASETS_DIR, "mot_raw")
    os.makedirs(mot_dir, exist_ok=True)
    
    mot17_url = "https://motchallenge.net/data/MOT17.zip"
    mot20_url = "https://motchallenge.net/data/MOT20.zip"
    
    download_and_extract(mot17_url, mot_dir)
    download_and_extract(mot20_url, mot_dir)
    print("⚠️ MOT datasets downloaded. You will need to convert MOT format to YOLO format.\n")

def setup_coco():
    print("--- Setting up COCO 2017 ---")
    print("To download COCO (~20GB), it is best to use the ultralytics package.")
    print("Run the following Python code in your environment:")
    print('''
    from ultralytics.utils.downloads import download
    from ultralytics import YOLO
    
    # Auto-downloads COCO if you start a training run or use their download script:
    # model = YOLO('yolov5n.pt')
    # model.train(data='coco.yaml', epochs=1) 
    ''')

def setup_crowdhuman():
    print("--- Setting up CrowdHuman ---")
    print("CrowdHuman is best downloaded via Roboflow (already in YOLO format).")
    print("1. Create a free Roboflow account.")
    print("2. Run the following code (replace 'YOUR_API_KEY' with your key):")
    print('''
    !pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("keio-dba-team").project("crowdhuman-nur7g")
    dataset = project.version(1).download("yolov5")
    ''')
    print("3. Move the downloaded images and labels into datasets/crowdhuman/\n")

def setup_caltech():
    print("--- Setting up Caltech Pedestrian ---")
    print("Caltech provides .seq files. You need a Python converter to extract PNGs.")
    print("1. Clone the converter: git clone https://github.com/mitmul/caltech-pedestrian-dataset-converter.git")
    print("2. Download the annotations and .seq files from the official site.")
    print("3. Run the converter to extract images to datasets/caltech/images/\n")

if __name__ == "__main__":
    print("🚀 Starting Dataset Setup Script\n")
    create_structure()
    
    # Uncomment the below line to automatically download MOT datasets (approx 5GB total)
    # setup_mot()
    
    setup_coco()
    setup_crowdhuman()
    setup_caltech()
    
    print("🎉 Setup script completed. Check the instructions above to finish downloading COCO, CrowdHuman, and Caltech.")
