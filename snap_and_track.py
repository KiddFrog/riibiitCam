import os
import time
from PIL import Image
import cv2

# Set the output directory for the photos
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiit/PICTURES")

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

# Load the cascade for facial tracking
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def capture_photo():
    # Generate a unique filename based on the current date and time
    filename = time.strftime("%Y%m%d-%H%M%S")

    # Use libcamera-jpeg to capture a photo and save it to the output directory
    os.system(f"libcamera-jpeg -o {os.path.join(OUTPUT_DIR, filename + '.jpg')}")

    return filename

def crop_and_track(filename):
    # Load the captured photo
    image_path = os.path.join(OUTPUT_DIR, filename + ".jpg")
    image = Image.open(image_path)

    # Initialize a list to store tracked images
    tracked_images = []

    for i in range(4):
        # Crop the image based on the camera's position
        x = WIDTH * (i % 2)
        y = HEIGHT * (i // 2)
        cropped_image = image.crop((x, y, x + WIDTH, y + HEIGHT))

        # Convert to OpenCV format for facial tracking
        cv_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

        # Detect faces in the cropped image
        faces = face_cascade.detectMultiScale(cv_image, 1.1, 4)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert back to PIL format
        tracked_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Save the tracked image with a new filename
        tracked_filename = f"Tracked_{filename}_{i}.jpg"
        tracked_image_path = os.path.join(OUTPUT_DIR, tracked_filename)
        tracked_image.save(tracked_image_path)
        
        tracked_images.append(tracked_image_path)

    return tracked_images

def save_bounding_box_positions(filename, bounding_box_positions):
    # Save bounding box positions to a text file
    positions_file_path = os.path.join(OUTPUT_DIR, f"{filename}_bounding_box_positions.txt")
    with open(positions_file_path, 'w') as positions_file:
        for key, value in bounding_box_positions.items():
            positions_file.write(f"{key} = {value}\n")

def place_images(filename, tracked_images, bounding_box_positions):
    # Load the position of the bounding box in image_1
    x_1 = bounding_box_positions.get("image_1 x_0", 0)
    y_1 = bounding_box_positions.get("image_1 y_0", 0)

    # Create a directory for the placed images
    placed_images_dir = os.path.join(OUTPUT_DIR, "PLACED_IMAGES")
    os.makedirs(placed_images_dir, exist_ok=True)

    # Initialize a list to store placed images
    placed_images = []

    for i, tracked_image_path in enumerate(tracked_images):
        # Load the tracked image
        tracked_image = Image.open(tracked_image_path)

        # Calculate the adjustment for x, y positions
        x_adjustment = x_1 - bounding_box_positions.get(f"image_1 x_0", 0)
        y_adjustment = y_1 - bounding_box_positions.get(f"image_1 y_0", 0)

        # Adjust the position of the bounding box in the image
        adjusted_image = tracked_image.crop((x_adjustment, y_adjustment, x_adjustment + WIDTH, y_adjustment + HEIGHT))

        # Save the adjusted image with a new filename
        placed_filename = f"PLACED_{filename}_{i}.jpg"
        placed_image_path = os.path.join(placed_images_dir, placed_filename)
        adjusted_image.save(placed_image_path)

        placed_images.append(placed_image_path)

    return placed_images

def create_gif(image_paths, gif_path):
    # Create a GIF from the images
    with Image.open(image_paths[0]) as gif_image:
        gif_image.save(gif_path, save_all=True, append_images=[Image.open(path) for path in image_paths[1:]], loop=0, duration=100)

# Step 1: Capture Photo
filename = capture_photo()

# Step 2: Crop and Track Faces
tracked_images = crop_and_track(filename)

# Step 3: Save Bounding Box Positions
bounding_box_positions = {
    "image_0 x_0": 0,
    "image_0 y_0": 0,
    "image_1 x_0": 0,  # Adjust as needed
    "image_1 y_0": 0,  # Adjust as needed
    "image_2 x_0": 0,
    "image_2 y_0": 0,
    "image_3 x_0": 0,
    "image_3 y_0": 0,
}
save_bounding_box_positions(filename, bounding_box_positions)

# Step 4: Create GIF from Tracked Images
tracked_gif_path = os.path.join(OUTPUT_DIR, f"tracked_faces_{filename}.gif")
create_gif(tracked_images, tracked_gif_path)

# Step 5: Place Images
placed_images = place_images(filename, tracked_images, bounding_box_positions)

# Step 6: Create GIF from Placed Images
placed_gif_path = os.path.join(OUTPUT_DIR, f"placed_faces_{filename}.gif")
create_gif(placed_images, placed_gif_path)
