import os
import cv2

def extract_file_name(filename):
    return os.path.splitext(filename)[0]

def add_title_to_image(image, title):
    title_org = (30, 50)  # Position of the title
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 1
    title_color = (255, 255, 255)  # White color
    title_thickness = 2
    cv2.putText(image, title, title_org, title_font, title_font_scale, title_color, title_thickness, cv2.LINE_AA)

def create_video(images_folder, output_video_path, fps=2):
    # Create a list of image files in the folder
    images = [img for img in sorted(os.listdir(images_folder)) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]

    # Sort the list of images alphabetically
    images.sort()

    # Specify video properties (resolution and frame rate)
    frame_width, frame_height = cv2.imread(os.path.join(images_folder, images[0])).shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Iterate through the list of images, add title, and write them to the video
    for image in images:
        image_path = os.path.join(images_folder, image)
        img = cv2.imread(image_path)
        title = extract_file_name(image)
        add_title_to_image(img, title)
        out.write(img)

    # Release the VideoWriter object
    out.release()

    print(output_video_path, " Video created successfully.")

if __name__ == "__main__":
    # Path to the folder containing images
    images_folder = 'images/miximages'

    # Specify the output video path
    output_video_path = "output_video.mp4"

    # Create the video
    create_video(images_folder, output_video_path)
