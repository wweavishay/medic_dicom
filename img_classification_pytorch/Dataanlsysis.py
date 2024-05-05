import os
import matplotlib.pyplot as plt
from PIL import Image

# Define directories
IMG_DIR = 'images'
TRAIN_DIR = os.path.join(IMG_DIR, 'train')
VAL_DIR = os.path.join(IMG_DIR, 'val')
TEST_DIR = os.path.join(IMG_DIR, 'test')

# Function to count the number of images in each class
def count_images_per_class(directory):
    classes = os.listdir(directory)
    class_counts = {}
    for cls in classes:
        class_dir = os.path.join(directory, cls)
        num_images = len(os.listdir(class_dir))
        class_counts[cls] = num_images
    return class_counts

# Function to plot class distribution
def plot_class_distribution(class_counts, title):
    labels = class_counts.keys()
    values = class_counts.values()

    plt.bar(labels, values)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Function to display sample images from each category
def display_sample_images_per_category(directory):
    classes = os.listdir(directory)
    for cls in classes:
        class_dir = os.path.join(directory, cls)
        image_files = os.listdir(class_dir)
        num_samples = min(25, len(image_files))
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        fig.suptitle(cls)
        for i, img_file in enumerate(image_files[:num_samples]):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path)
            ax = axes[i // 5, i % 5]
            ax.imshow(img)
            ax.axis('off')
        plt.show()

# Function to display the ratio of images between train and test sets
def plot_train_test_ratio(train_counts, test_counts):
    labels = ['Train Set', 'Test Set']
    sizes = [sum(train_counts.values()), sum(test_counts.values())]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Ratio of Images between Train and Test Sets')
    plt.show()

# Function to display menu options
def display_menu():
    print("Menu:")
    print("1. Plot class distribution for train set")
    print("2. Plot the ratio of images between train and test sets")
    print("3. Plot class distribution for test set")
    print("4. Display sample images for each category in train set")
    print("5. Display sample images for each category in test set")
    print("6. Exit")

# Main function
if __name__ == "__main__":
    train_counts = count_images_per_class(TRAIN_DIR)
    test_counts = count_images_per_class(TEST_DIR)
    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            plot_class_distribution(train_counts, "Class Distribution for Train Set")
        elif choice == "2":
            plot_train_test_ratio(train_counts, test_counts)
        elif choice == "3":
            plot_class_distribution(test_counts, "Class Distribution for Test Set")
        elif choice == "4":
            display_sample_images_per_category(TRAIN_DIR)
        elif choice == "5":
            display_sample_images_per_category(TEST_DIR)
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
