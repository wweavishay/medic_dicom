import os
import shutil
import random

def split_images(input_folder, output_folder):
    # Create train, test, and val directories
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    val_folder = os.path.join(output_folder, 'val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # List class folders
    classes_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    for class_folder in classes_folders:
        class_images = os.listdir(os.path.join(input_folder, class_folder))
        random.shuffle(class_images)  # Shuffle images

        num_images = len(class_images)
        num_train = int(0.7 * num_images)  # 70% for train
        num_test = int(0.15 * num_images)  # 15% for test
        num_val = num_images - num_train - num_test  # Remaining for validation

        train_images = class_images[:num_train]
        test_images = class_images[num_train:num_train + num_test]
        val_images = class_images[num_train + num_test:]

        # Copy images to respective folders
        for image in train_images:
            src = os.path.join(input_folder, class_folder, image)
            dest = os.path.join(train_folder, class_folder, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)

        for image in test_images:
            src = os.path.join(input_folder, class_folder, image)
            dest = os.path.join(test_folder, class_folder, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)

        for image in val_images:
            src = os.path.join(input_folder, class_folder, image)
            dest = os.path.join(val_folder, class_folder, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)

# Example usage
input_folder = 'images'  # Update with your input folder containing class folders
output_folder = 'images'  # Update with desired output folder
split_images(input_folder, output_folder)