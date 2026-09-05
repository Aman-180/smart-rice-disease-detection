import os
import random
import shutil

# CONFIG
source_images = "Object-Detection-1/train/images"
source_labels = "Object-Detection-1/train/labels"
output_root = "dataset_yolo"

train_ratio = 0.70
valid_ratio = 0.20
test_ratio = 0.10

random.seed(42)

image_files = [f for f in os.listdir(source_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

total = len(image_files)
train_end = int(total * train_ratio)
valid_end = train_end + int(total * valid_ratio)

splits = {
    "train": image_files[:train_end],
    "valid": image_files[train_end:valid_end],
    "test": image_files[valid_end:]
}

for split_name, files in splits.items():
    img_out = os.path.join(output_root, split_name, "images")
    lbl_out = os.path.join(output_root, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_file in files:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        src_img = os.path.join(source_images, img_file)
        src_lbl = os.path.join(source_labels, label_file)

        dst_img = os.path.join(img_out, img_file)
        dst_lbl = os.path.join(lbl_out, label_file)

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"WARNING: no label found for {img_file}")

    print(f"{split_name}: {len(files)} images")

print("Split complete.")