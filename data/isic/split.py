import os
import shutil
import csv

# CSV file containing image IDs and their corresponding classes
csv_file = 'data/isic/class_id.csv'

# original image and mask directory
img_dir = 'data/isic/ISIC2018_Task1-2_Training_Input'  
mask_dir = 'data/isic/ISIC2018_Task1_Training_GroundTruth'  


output_base_dir = 'data/isic'


class_map = {
    'nevus': 'nevus',
    'melanoma': 'melanoma',
    'seborrheic_keratosis': 'seborrheic_keratosis'  
}


for class_name in class_map.values():
    os.makedirs(os.path.join(output_base_dir, 'images', class_name), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'masks', class_name), exist_ok=True)


with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    
    for row in reader:
        img_id, img_class = row[0], row[1]  


        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        mask_path = os.path.join(mask_dir, f'{img_id}_segmentation.png')

        target_img_dir = os.path.join(output_base_dir, 'images', class_map[img_class])
        target_mask_dir = os.path.join(output_base_dir, 'masks', class_map[img_class])

        if os.path.exists(img_path):
            shutil.copy(img_path, target_img_dir)  
        else:
            print(f"Image {img_path} not found.")

        if os.path.exists(mask_path):
            shutil.copy(mask_path, target_mask_dir) 
        else:
            print(f"Mask {mask_path} not found.")
