import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np


def read_dicom_files(root_folder, subfolders):
    dicom_files = []
    for folder in subfolders:
        folder_path = os.path.join(root_folder, folder)
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('.dcm'):
                    file_path = os.path.join(dirpath, filename)
                    # Add force=True to force reading even if the file doesn't strictly adhere to the DICOM standard
                    dicom_data = pydicom.dcmread(file_path, force=True)
                    dicom_files.append((folder, filename, dicom_data))
    return dicom_files




def apply_windowing(pixel_array, window_center, window_width):
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    return np.clip(pixel_array, min_value, max_value)


def display_and_save_dicom_images(dicom_files, window_center, window_width, output_folder):
    for filename, dicom_data in dicom_files:
        pixel_array = dicom_data.pixel_array

        if 'NumberOfFrames' in dicom_data:
            # Dynamic series (video or time-series data)
            number_of_frames = int(dicom_data.NumberOfFrames)
            for i in range(number_of_frames):
                frame = dicom_data.get_frame(i)
                windowed_image = apply_windowing(frame, window_center, window_width)

                # Create a folder for each DICOM file
                output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
                os.makedirs(output_subfolder, exist_ok=True)

                # Save the image plot as PNG
                output_file_path = os.path.join(output_subfolder, f'image_plot_{i}.png')
                plt.imshow(windowed_image, cmap=plt.cm.bone)
                plt.title(f'DICOM Image (Frame {i + 1}/{number_of_frames})')
                plt.axis('off')
                plt.savefig(output_file_path, bbox_inches='tight')
                plt.close()  # Close the plot to avoid displaying it
        else:
            # Static image
            windowed_image = apply_windowing(pixel_array, window_center, window_width)

            # Create a folder for each DICOM file
            output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(output_subfolder, exist_ok=True)

            # Save the image plot as PNG
            output_file_path = os.path.join(output_subfolder, 'image_plot.png')
            plt.imshow(windowed_image, cmap=plt.cm.bone)
            plt.title('DICOM Image')
            plt.axis('off')
            plt.savefig(output_file_path, bbox_inches='tight')
            plt.close()  # Close the plot to avoid displaying it


def main():
    root_folder = "datapatient"
    subfolders = ["A2", "A3", "N1", "N2", "N3", "N5", "N6", "PD1", "PD3", "PD4", "PD5", "PD6"]

    for folder in subfolders:
        output_folder = os.path.join("images", f"dicomfiles_{folder}")

        dicom_files = read_dicom_files(root_folder, [folder])

        # Adjust these parameters for windowing
        window_center = 50  # Example value, adjust as needed
        window_width = 350  # Example value, adjust as needed

        choice = input(f"Do you want to update the data to put the DICOM files in folder '{folder}'? (y/n): ")
        if choice.lower() == 'y':
            for subfolder, filename, dicom_data in dicom_files:
                pixel_array = dicom_data.pixel_array
                output_subfolder = os.path.join(output_folder, subfolder)
                os.makedirs(output_subfolder, exist_ok=True)
                if 'NumberOfFrames' in dicom_data:
                    number_of_frames = int(dicom_data.NumberOfFrames)
                    for i in range(number_of_frames):
                        frame = dicom_data.get_frame(i)
                        windowed_image = apply_windowing(frame, window_center, window_width)
                        output_file_path = os.path.join(output_subfolder, f'{os.path.splitext(filename)[0]}_{i}.png')
                        plt.imsave(output_file_path, windowed_image, cmap=plt.cm.bone)
                else:
                    windowed_image = apply_windowing(pixel_array, window_center, window_width)
                    output_file_path = os.path.join(output_subfolder, f'{os.path.splitext(filename)[0]}.png')
                    plt.imsave(output_file_path, windowed_image, cmap=plt.cm.bone)
            print(f"DICOM files for folder '{folder}' have been organized into folders.")
        elif choice.lower() == 'n':
            print(f"No action taken for folder '{folder}'. Proceeding to the next folder.")
        else:
            print("Invalid choice. Proceeding to the next folder.")

if __name__ == "__main__":
    main()
