import os
import cv2
import numpy as np


def detect_contours(frames_folder, output_video_path, fps=2):
    image_files = []

    # Traverse through all subdirectories in frames_folder
    for root, dirs, files in os.walk(frames_folder):
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No PNG image files found in {frames_folder}")
        return

    # Sort the image files
    image_files.sort()

    # Read the first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Failed to read {image_files[0]}. Check if the file exists and is accessible.")
        return
    height, width, _ = first_img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as per your requirement
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    total_frames = len(image_files)
    frame_count = 0
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to read {image_file}. Skipping...")
            continue

        # Calculate absolute difference between the current frame and the first frame
        frame_diff = cv2.absdiff(first_img, img)

        # Convert difference image to grayscale
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

        # Apply threshold to create binary image
        _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the binary difference image
        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours onto the original image
        contour_img = img.copy()
        for contour in contours:
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)  # Draw green contour

        # Overlay frame counter onto the image
        frame_count += 1
        text = f'Frame: {frame_count} / {total_frames}'
        cv2.putText(contour_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Text in green

        # Write each frame multiple times to make it appear for 0.5 seconds
        for _ in range(int(fps * 0.5)):
            out.write(contour_img)

    # Release the VideoWriter object
    out.release()


def highlight_amoebic_shape(frames_folder, output_video_path, fps=2):
    image_files = []

    # Traverse through all subdirectories in frames_folder
    for root, dirs, files in os.walk(frames_folder):
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No PNG image files found in {frames_folder}")
        return

    # Sort the image files
    image_files.sort()

    # Read the first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Failed to read {image_files[0]}. Check if the file exists and is accessible.")
        return
    height, width, _ = first_img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as per your requirement
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    total_frames = len(image_files)
    frame_count = 0
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to read {image_file}. Skipping...")
            continue

        # Calculate absolute difference between the current frame and the first frame
        frame_diff = cv2.absdiff(first_img, img)

        # Convert difference image to grayscale
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

        # Apply threshold to create binary image
        _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the binary difference image
        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask to fill the detected contours
        mask = np.zeros_like(img)

        # Draw contours onto the mask
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Dilate the mask to expand the filled area
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=3)

        # Apply the dilated mask to the original image
        result = cv2.bitwise_and(img, dilated_mask)

        # Overlay frame counter onto the image
        frame_count += 1
        text = f'Frame: {frame_count} / {total_frames}'
        cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Text in green

        # Write each frame multiple times to make it appear for 0.5 seconds
        for _ in range(int(fps * 0.5)):
            out.write(result)

    # Release the VideoWriter object
    out.release()


if __name__ == "__main__":
    frames_parent_folder = "images"  # Assuming the images are in the "images" folder

    for root, dirs, files in os.walk(frames_parent_folder):
        for folder in dirs:
            frames_folder = os.path.join(root, folder)
            contour_video_path = os.path.join(frames_folder, f"{folder}_contour_detection_video.mp4")
            amoebic_video_path = os.path.join(frames_folder, f"{folder}_amoebic_shape_highlight_video.mp4")
            fps = 2  # Frames per second (each frame will be displayed for 0.5 seconds)

            try:
                detect_contours(frames_folder, contour_video_path, fps)
                highlight_amoebic_shape(frames_folder, amoebic_video_path, fps)
                print(f"Videos created for folder {frames_folder}")
            except Exception as e:
                print(f"Error processing {frames_folder}: {e}")
