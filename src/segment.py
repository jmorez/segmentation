import torchvision
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


fp = r"data/image.jpeg"

img = Image.open(fp=fp)

#segmentation_model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.fasterrcnn_resnet50_fpn_v2)
segmentation_model = models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights)
segmentation_model.eval()

preprocess = transforms.Compose(
                [
                    #transforms.Resize((1600, 1200)),  # Resize to 224x224
                    transforms.ToTensor(),  # Convert to tensor
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    #     std=[0.229, 0.224, 0.225],  # ImageNet std
                    # ),
                ]
            )

threshold = 0.0 #Set your desired score threshold here. It does not seem to be a good metric to filter out duplicates though.
similarity_threshold = 0.5 # RMSE threshold for mask similarity (adjust as needed)

y = segmentation_model([preprocess(img)])
num_masks = y[0]["masks"].shape[0]
scores = y[0]["scores"].detach().cpu().numpy()
labels = y[0]["labels"].detach().cpu().numpy()

# Filter masks by threshold
indices = [i for i, score in enumerate(scores) if score >= threshold]

# Similarity check: keep only unique masks
unique_indices = []
unique_masks = []

def mask_ncc(mask1, mask2):
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)
    mask1 = (mask1 - mask1.mean()) / (mask1.std() + 1e-8)
    mask2 = (mask2 - mask2.mean()) / (mask2.std() + 1e-8)
    return np.mean(mask1 * mask2)

for idx in indices:
    mask = y[0]["masks"][idx, 0].detach().cpu().numpy()
    is_unique = True
    for umask in unique_masks:
        if mask_ncc(mask > 0.5, umask > 0.5) > similarity_threshold:
            is_unique = False
            break
    if is_unique:
        unique_indices.append(idx)
        unique_masks.append(mask)

filtered_num_masks = len(unique_indices)

cols = 6
rows = (filtered_num_masks + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

img_np = np.array(img)

for ax_idx, mask_idx in enumerate(unique_indices):
    mask = y[0]["masks"][mask_idx, 0].detach().cpu().numpy()
    # Plot image
    axes[ax_idx].imshow(img_np)
    # Plot mask if available
    if "masks" in y[0] and y[0]["masks"] is not None and y[0]["masks"].shape[0] > mask_idx:
        mask = y[0]["masks"][mask_idx, 0].detach().cpu().numpy()
        axes[ax_idx].imshow(mask > 0.5, cmap='jet', alpha=0.5)
        
    # Plot bounding box (always available)
    box = y[0]["boxes"][mask_idx].detach().cpu().numpy()
    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=1)
    axes[ax_idx].add_patch(rect)
    label = labels[mask_idx]
    score = scores[mask_idx]
    axes[ax_idx].set_title(f"Mask {mask_idx}\nLabel: {label}\nScore: {score:.2f}")
    axes[ax_idx].axis('off')

for ax_idx in range(filtered_num_masks, len(axes)):
    axes[ax_idx].axis('off')

plt.tight_layout()
plt.show(block=True)