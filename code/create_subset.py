import os
import shutil
import random

# CONFIGURATION
SOURCE_IMG_DIR = r"..\dataset\train\images"
SOURCE_LBL_DIR = r"..\dataset\train\labels"

DEST_IMG_DIR = r"..\dataset\train_small\images"
DEST_LBL_DIR = r"..\dataset\train_small\labels"

TARGET_COUNT = 1500 

def main():
    # 1. Creating new folders
    os.makedirs(DEST_IMG_DIR, exist_ok=True)
    os.makedirs(DEST_LBL_DIR, exist_ok=True)

    # 2. Getting list of all image files
    all_images = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(all_images)} total images.")

    if len(all_images) == 0:
        print("Error: No images found! Check paths.")
        return

    # 3. Shuffle and Pick
    if len(all_images) > TARGET_COUNT:
        selected_images = random.sample(all_images, TARGET_COUNT)
    else:
        selected_images = all_images

    print(f"Copying {len(selected_images)} images to 'train_small'...")

    # 4. Copy files
    count = 0
    for img_name in selected_images:
        src_img = os.path.join(SOURCE_IMG_DIR, img_name)
        dst_img = os.path.join(DEST_IMG_DIR, img_name)
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl = os.path.join(SOURCE_LBL_DIR, label_name)
        dst_lbl = os.path.join(DEST_LBL_DIR, label_name)

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            count += 1
        else:
            print(f"Warning: Label missing for {img_name}")

    print(f"Done! Created dataset with {count} pairs in '../dataset/train_small'")

if __name__ == "__main__":
    main()