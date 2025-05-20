import cv2
import dlib
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import pickle
import threading
from PIL import Image, ImageTk  # For image preview

# Load YOLOv8 model for human detection
yolo_model = YOLO('my_model.pt')  # Use 'cpu' if you don't have a GPU

# Load dlib's face detector and embedding model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_embedder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Global variables
registered_embeddings = []
registered_labels = []
save_folder = "registered_faces"
pickle_file = "registered_records.pkl"

# Create the folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load registered records from pickle file if it exists
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        registered_embeddings, registered_labels = pickle.load(f)

# Function to extract face embeddings using dlib
def extract_embedding(face_image):
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    faces = face_detector(face_image_rgb)
    if len(faces) == 0:
        return None
    face = faces[0]
    shape = shape_predictor(face_image_rgb, face)
    embedding = face_embedder.compute_face_descriptor(face_image_rgb, shape)
    return np.array(embedding)

# Function to detect humans and extract faces
def detect_and_extract_faces(frame):
    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))
    results = yolo_model(frame_resized)
    faces = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 is 'person' in YOLOv8
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                faces.append((face, (x1, y1, x2, y2)))
    return faces

# Function to register a face using OpenCV
def register_face():
    def registration_thread():
        global registered_embeddings, registered_labels
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame")
                break

            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame
                continue

            faces = detect_and_extract_faces(frame)

            for face, bbox in faces:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Press 's' to capture", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Register Face", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(faces) == 0:
                    messagebox.showwarning("Warning", "No face detected")
                else:
                    face, bbox = faces[0]
                    embedding = extract_embedding(face)
                    if embedding is not None:
                        # Schedule the dialog to run in the main thread
                        root.after(0, lambda: prompt_for_name_and_id(face, embedding))
                        break
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=registration_thread, daemon=True).start()

# Function to prompt for name and ID (runs in the main thread)
def prompt_for_name_and_id(face, embedding):
    # Show preview of the captured face
    preview_window = tk.Toplevel(root)
    preview_window.title("Preview")
    preview_label = tk.Label(preview_window)
    preview_label.pack()

    # Convert the face image to a format Tkinter can display
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tk = ImageTk.PhotoImage(face_pil)
    preview_label.config(image=face_tk)
    preview_label.image = face_tk

    # Prompt for name and ID
    name = simpledialog.askstring("Input", "Enter Name:", parent=preview_window)
    id_number = simpledialog.askstring("Input", "Enter ID:", parent=preview_window)

    if name and id_number:
        registered_embeddings.append(embedding)
        registered_labels.append(f"{name}_{id_number}")

        # Save the captured face image
        image_filename = os.path.join(save_folder, f"{name}_{id_number}.jpg")
        cv2.imwrite(image_filename, face)

        # Save the registered records to pickle file
        with open(pickle_file, "wb") as f:
            pickle.dump((registered_embeddings, registered_labels), f)

        messagebox.showinfo("Success", "Face registered successfully")
    else:
        messagebox.showwarning("Warning", "Name and ID are required")

    preview_window.destroy()

# Function to recognize a face
def recognize_face():
    def recognition_thread():
        global registered_embeddings, registered_labels
        if len(registered_embeddings) == 0:
            messagebox.showwarning("Warning", "No faces registered")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame")
                break

            faces = detect_and_extract_faces(frame)

            for face, bbox in faces:
                x1, y1, x2, y2 = bbox
                embedding = extract_embedding(face)
                if embedding is not None:
                    distances = [np.linalg.norm(embedding - reg_embedding) for reg_embedding in registered_embeddings]
                    min_distance = min(distances)
                    if min_distance < 0.4: #0.4 is better in our case
                        label = registered_labels[distances.index(min_distance)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f" {label}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown face", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Recognize Face", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=recognition_thread, daemon=True).start()

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Face Registration and Recognition")

# Create a status bar
status_bar = tk.Label(root, text="Welcome to Face Registration and Recognition", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

register_button = tk.Button(root, text="Register Face", command=register_face, width=20, height=2)
register_button.pack(pady=10)

recognize_button = tk.Button(root, text="Recognize Face", command=recognize_face, width=20, height=2)
recognize_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_app, width=20, height=2)
exit_button.pack(pady=10)

root.mainloop()