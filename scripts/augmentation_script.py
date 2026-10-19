import albumentations as A
import cv2
import os
import glob
import matplotlib.pyplot as plt

# 1. Define the Augmentation Pipeline (The magic ingredient for occlusion)
# CoarseDropout randomly drops out rectangular regions from the image,
# filling them with black pixels to simulate heavy occlusion.
def get_train_transforms():
    return A.Compose([
        # Randomly cut out 3 to 8 black boxes (size 16x16 to 64x64)
        A.CoarseDropout(
            max_holes=8, 
            max_height=64, 
            max_width=64, 
            min_holes=3, 
            min_height=16, 
            min_width=16, 
            fill_value=0, 
            p=0.8 # Apply this 80% of the time
        )
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def visualize_augmentation(image_path, label_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # Load YOLO labels [class_id, center_x, center_y, width, height]
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
                
    # Apply the Albumentations transformation
    transform = get_train_transforms()
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    
    # Draw the bounding boxes on the augmented image
    for bbox in transformed_bboxes:
        cx, cy, bw, bh = bbox
        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)
        cv2.rectangle(transformed_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # Show the original vs augmented side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(transformed_image)
    axes[1].set_title("Augmented (CoarseDropout - Fake Occlusion)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_sample.png", dpi=300)
    print("Saved sample to augmentation_sample.png")
    plt.show()

if __name__ == "__main__":
    print("Testing Albumentations CoarseDropout for LARD...")
    # Grab a random image from our dataset to test on
    images = glob.glob(os.path.join("datasets", "crowdhuman", "images", "train", "*.jpg"))
    if images:
        test_img = images[0]
        test_lbl = test_img.replace("images", "labels").replace(".jpg", ".txt")
        visualize_augmentation(test_img, test_lbl)
    else:
        print("No images found! Make sure CrowdHuman is downloaded first.")