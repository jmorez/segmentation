import torchvision
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CONFIG = {"filepath": r"data/image.jpeg", "iou_threshold": 0.9}


img = Image.open(fp=CONFIG["filepath"])

# segmentation_model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.fasterrcnn_resnet50_fpn_v2)
segmentation_model = models.detection.maskrcnn_resnet50_fpn_v2(
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights
)
segmentation_model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize((1600, 1200)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
    ]
)

y = segmentation_model([preprocess(img)])

# Reformat the results for ease of use. Have to squeeze the mask because of some stupid singleton dimension...
segmentation_results = [
    {"mask": np.squeeze(mask), "box": box, "score": score, "label": label}
    for mask, box, score, label in zip(
        y[0]["masks"].detach().cpu().numpy(),
        y[0]["boxes"].detach().cpu().numpy(),
        y[0]["scores"].detach().cpu().numpy(),
        y[0]["labels"].detach().cpu().numpy(),
    )
]


# Metric to filter out duplicates.
def intersectionOverUnion(a: np.ndarray, b: np.ndarray) -> float:
    intersection = a.astype(bool) & b.astype(bool)
    union = a.astype(bool) | b.astype(bool)
    return np.sum(intersection) / np.sum(union)


# Remove duplicate segmentations based on similarity of the bounding box.
unique_segmentation_results = []
for seg_result in segmentation_results:
    is_unique = True
    for unique_seg in unique_segmentation_results:
        if (
            intersectionOverUnion(unique_seg["mask"], seg_result["mask"])
            > CONFIG["iou_threshold"]
        ):
            is_unique = False
            break

    if is_unique:
        unique_segmentation_results.append(seg_result)

cols = 6
rows = (len(unique_segmentation_results) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

img_np = np.array(img)

for ax_idx, seg_result in enumerate(unique_segmentation_results):
    mask = seg_result["mask"]
    box = seg_result["box"]
    label = seg_result["label"]
    score = seg_result["score"]

    # Plot image
    axes[ax_idx].imshow(img_np)

    # Plot mask if available
    axes[ax_idx].imshow(mask > 0.5, cmap="jet", alpha=0.5)
    # Plot bounding box
    x1, y1, x2, y2 = box
    rect = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1
    )
    axes[ax_idx].add_patch(rect)
    axes[ax_idx].set_title(f"Mask {ax_idx}\nLabel: {label}\nScore: {score:.2f}")
    axes[ax_idx].axis("off")

# Hide any unused axes
for ax in axes[len(unique_segmentation_results) :]:
    ax.axis("off")


plt.tight_layout()
plt.show(block=True)
