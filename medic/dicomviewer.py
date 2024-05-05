import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import pydicom

class DICOMViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM-CT Image Viewer")
        self.current_slice = 0
        self.dicom_files = []
        self.folder_path = None
        self.video_playing = False
        self.color_display = True
        self.brightness = 0
        self.red_value = 0
        self.green_value = 0
        self.blue_value = 0

        # Create Menu
        self.menu_bar = Menu(root)
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Patient", command=self.open_patient)
        self.file_menu.add_command(label="Exit", command=root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.display_menu = Menu(self.menu_bar, tearoff=0)
        self.display_menu.add_command(label="Toggle Color Display", command=self.toggle_color_display)
        self.menu_bar.add_cascade(label="Display", menu=self.display_menu)
        self.root.config(menu=self.menu_bar)

        # Frames
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(pady=10)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(pady=10)

        # Previous Button
        self.previous_button = tk.Button(self.top_frame, text="Previous Slice", command=self.previous_slice)
        self.previous_button.grid(row=0, column=0, padx=10)

        # Next Button
        self.next_button = tk.Button(self.top_frame, text="Next Slice", command=self.next_slice)
        self.next_button.grid(row=0, column=1, padx=10)

        # Restart Button
        self.restart_button = tk.Button(self.top_frame, text="Restart", command=self.restart_video)
        self.restart_button.grid(row=0, column=2, padx=10)

        # Go Label
        self.go_label = tk.Label(self.top_frame, text="Go to Slice:")
        self.go_label.grid(row=0, column=3, padx=10)

        # Slice Entry
        self.slice_entry = tk.Entry(self.top_frame)
        self.slice_entry.grid(row=0, column=4, padx=10)

        # Go Button
        self.go_button = tk.Button(self.top_frame, text="Go", command=self.go_to_slice)
        self.go_button.grid(row=0, column=5, padx=10)

        # Play/Pause Button
        self.play_pause_button = tk.Button(self.top_frame, text="Play", command=self.play_pause_video)
        self.play_pause_button.grid(row=0, column=6, padx=10)

        # Video Speed Label
        self.video_speed_label = tk.Label(self.top_frame, text="Video Speed:")
        self.video_speed_label.grid(row=0, column=7, padx=10)

        # Video Speed Scale
        self.video_speed_scale = tk.Scale(self.top_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_video_speed)
        self.video_speed_scale.grid(row=0, column=8, padx=10)
        self.video_speed = 1

        # Frame Counter Label
        self.frame_counter_label = tk.Label(self.top_frame, text="Frame: 0")
        self.frame_counter_label.grid(row=0, column=9, padx=10)

        # Canvas for Image Display
        self.canvas = tk.Canvas(self.root, width=512, height=512)
        self.canvas.pack()

        self.image = None

        # Color Adjustment Sliders
        self.red_scale = tk.Scale(self.bottom_frame, label="Red", from_=-255, to=255, orient=tk.HORIZONTAL, command=self.update_color)
        self.red_scale.grid(row=0, column=0, padx=10)
        self.green_scale = tk.Scale(self.bottom_frame, label="Green", from_=-255, to=255, orient=tk.HORIZONTAL, command=self.update_color)
        self.green_scale.grid(row=0, column=1, padx=10)
        self.blue_scale = tk.Scale(self.bottom_frame, label="Blue", from_=-255, to=255, orient=tk.HORIZONTAL, command=self.update_color)
        self.blue_scale.grid(row=0, column=2, padx=10)

    def open_patient(self):
        self.folder_path = filedialog.askdirectory(title="Select Patient Folder")
        if self.folder_path:
            self.dicom_files = [file for file in os.listdir(self.folder_path) if file.lower().endswith('.dcm')]
            self.dicom_files.sort()
            self.current_slice = 0
            self.show_slice()

    def previous_slice(self):
        if self.current_slice > 0:
            self.current_slice -= 1
            self.show_slice()

    def next_slice(self):
        if self.current_slice < len(self.dicom_files) - 1:
            self.current_slice += 1
            self.show_slice()

    def restart_video(self):
        self.current_slice = 0
        self.show_slice()

    def go_to_slice(self):
        try:
            slice_num = int(self.slice_entry.get())
            if 0 <= slice_num < len(self.dicom_files):
                self.current_slice = slice_num
                self.show_slice()
            else:
                messagebox.showerror("Error", "Invalid slice number")
        except ValueError:
            messagebox.showerror("Error", "Slice number must be an integer")

    def show_slice(self):
        if self.folder_path is None:
            messagebox.showerror("Error", "No patient folder selected")
            return

        file_path = os.path.join(self.folder_path, self.dicom_files[self.current_slice])
        ds = pydicom.dcmread(file_path)
        slice_data = ds.pixel_array

        # Apply brightness adjustment
        slice_data += self.brightness

        # Convert grayscale to RGB
        slice_data_rgb = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2RGB)

        # Apply color transformation
        if self.color_display:
            slice_data_rgb[:, :, 0] += self.red_value
            slice_data_rgb[:, :, 1] += self.green_value
            slice_data_rgb[:, :, 2] += self.blue_value

        # Convert to uint8
        slice_data_rgb = np.clip(slice_data_rgb, 0, 255).astype(np.uint8)

        # Convert to RGB image
        img = Image.fromarray(slice_data_rgb)

        # Display image
        img = ImageTk.PhotoImage(image=img)
        if self.image is not None:
            self.canvas.delete(self.image)
        self.image = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        # Update frame counter label
        self.frame_counter_label.config(text=f"Frame: {self.current_slice + 1}")

    def toggle_color_display(self):
        self.color_display = not self.color_display
        self.show_slice()

    def play_pause_video(self):
        if not self.video_playing:
            self.video_playing = True
            self.play_pause_button.config(text="Pause")
            self.play_video()
        else:
            self.video_playing = False
            self.play_pause_button.config(text="Play")

    def play_video(self):
        if self.video_playing:
            self.next_slice()
            self.root.after(int(1000 / self.video_speed), self.play_video)

    def update_video_speed(self, value):
        self.video_speed = int(value)

    def update_color(self, event=None):
        self.red_value = self.red_scale.get()
        self.green_value = self.green_scale.get()
        self.blue_value = self.blue_scale.get()
        self.show_slice()


if __name__ == "__main__":
    root = tk.Tk()
    app = DICOMViewer(root)
    root.mainloop()
