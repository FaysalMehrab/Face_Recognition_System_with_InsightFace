import insightface
import cv2
import numpy as np
from pathlib import Path
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize InsightFace FaceAnalysis
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320))  # Use CPU; set ctx_id=0 for GPU

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for face detection."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def detect_and_extract_embedding(image):
    """Detect faces and extract embedding for the first detected face."""
    try:
        faces = app.get(image)
        if len(faces) == 0:
            raise ValueError("No faces detected")
        return faces[0].embedding
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def process_person_folder(person_folder, output_folder):
    """Process all images in a person's folder and save embeddings."""
    person_name = person_folder.name
    output_person_folder = output_folder / person_name
    output_person_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing person: {person_name}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(person_folder.glob(ext))
    
    if not image_paths:
        logger.warning(f"No images found in {person_folder}")
        return

    for image_path in image_paths:
        logger.info(f"Processing image: {image_path}")
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_path)
        if image is None:
            continue

        # Detect face and extract embedding
        embedding = detect_and_extract_embedding(image)
        if embedding is None:
            continue

        # Save embedding
        embedding_filename = output_person_folder / f"{image_path.stem}.npy"
        np.save(embedding_filename, embedding)
        logger.info(f"Saved embedding to: {embedding_filename}")

def main(data_folder="Data", output_folder="Embeddings"):
    """Main function to extract and save embeddings for all persons."""
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not data_folder.exists():
        logger.error(f"Data folder {data_folder} does not exist")
        return

    person_folders = [f for f in data_folder.iterdir() if f.is_dir()]
    if not person_folders:
        logger.error(f"No person folders found in {data_folder}")
        return

    for person_folder in person_folders:
        process_person_folder(person_folder, output_folder)

    logger.info("Embedding extraction completed")

if __name__ == "__main__":
    main()