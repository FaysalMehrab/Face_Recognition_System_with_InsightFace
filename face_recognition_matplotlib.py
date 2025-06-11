import insightface
import cv2
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize InsightFace FaceAnalysis
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU; set ctx_id=0 for GPU

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for face detection."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Return BGR for display, RGB for processing
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None, None

def extract_embedding_and_bbox(image):
    """Extract embedding and bounding box for the first detected face."""
    try:
        faces = app.get(image)
        if len(faces) == 0:
            raise ValueError("No faces detected")
        return faces[0].embedding, faces[0].bbox
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None

def load_saved_embeddings(embeddings_folder):
    """Load all saved embeddings from the Embeddings folder."""
    embeddings_folder = Path(embeddings_folder)
    if not embeddings_folder.exists():
        logger.error(f"Embeddings folder {embeddings_folder} does not exist")
        return np.array([]), []
    
    embeddings = []
    labels = []
    for person_folder in embeddings_folder.iterdir():
        if person_folder.is_dir():
            person_name = person_folder.name
            for embedding_file in person_folder.glob("*.npy"):
                embedding = np.load(embedding_file)
                embeddings.append(embedding)
                labels.append(person_name)
    
    return np.array(embeddings), labels

def recognize_face(image_path, embeddings_folder, threshold=0.4):
    """Recognize the person in the input image and return details."""
    # Check if image exists
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image {image_path} does not exist")
        return None, None, None, None

    # Load and preprocess image
    img_bgr, img_rgb = load_and_preprocess_image(image_path)
    if img_rgb is None:
        return None, None, None, None

    # Extract embedding and bounding box
    query_embedding, bbox = extract_embedding_and_bbox(img_rgb)
    if query_embedding is None:
        return None, None, img_bgr, None

    # Load saved embeddings
    saved_embeddings, labels = load_saved_embeddings(embeddings_folder)
    if len(saved_embeddings) == 0:
        logger.error("No saved embeddings found")
        return None, None, img_bgr, None

    # Compute similarities
    similarities = cosine_similarity([query_embedding], saved_embeddings)[0]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)

    if max_similarity >= threshold:
        return labels[max_index], max_similarity, img_bgr, bbox
    else:
        return "Unknown", max_similarity, img_bgr, bbox

def save_image_with_details(img_bgr, person_name, similarity, bbox, output_path):
    """Save the image with bounding box and recognition details and show in a popup window."""
    if img_bgr is None:
        logger.error("No image to save")
        return

    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Prepare text
    text = f"Name: {person_name}"
    similarity_text = f"Similarity: {similarity*100:.2f}%"
    
    # Draw text background for readability with better spacing
    text_x, text_y = 10, 40
    line_gap = 15  # Gap between lines

    # Name background
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(
        img_bgr,
        (text_x - 5, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 5, text_y + 5),
        (0, 0, 0), -1
    )

    # Similarity background
    similarity_size, _ = cv2.getTextSize(similarity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    sim_y = text_y + text_size[1] + line_gap
    cv2.rectangle(
        img_bgr,
        (text_x - 5, sim_y - similarity_size[1] - 10),
        (text_x + similarity_size[0] + 5, sim_y + 5),
        (0, 0, 0), -1
    )

    # Draw text with better spacing
    cv2.putText(img_bgr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img_bgr, similarity_text, (text_x, sim_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save image
    cv2.imwrite(str(output_path), img_bgr)
    logger.info(f"Saved annotated image to: {output_path}")

    # Show image in a popup window using matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Recognition Result")
    plt.axis('off')
    plt.show()

def recognize_face_from_frame(frame, embeddings_folder, threshold=0.4):
    """Recognize face from a webcam frame."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    query_embedding, bbox = extract_embedding_and_bbox(img_rgb)
    if query_embedding is None:
        return "No face", 0.0, frame, None

    saved_embeddings, labels = load_saved_embeddings(embeddings_folder)
    if len(saved_embeddings) == 0:
        return "No embeddings", 0.0, frame, bbox

    similarities = cosine_similarity([query_embedding], saved_embeddings)[0]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)

    if max_similarity >= threshold:
        return labels[max_index], max_similarity, frame, bbox
    else:
        return "Unknown", max_similarity, frame, bbox

def recognize_faces_from_frame(frame, embeddings_folder, threshold=0.4):
    """Recognize all faces from a webcam frame."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    results = []
    saved_embeddings, labels = load_saved_embeddings(embeddings_folder)
    if len(saved_embeddings) == 0:
        return []

    for face in faces:
        query_embedding = face.embedding
        bbox = face.bbox
        similarities = cosine_similarity([query_embedding], saved_embeddings)[0]
        max_similarity = np.max(similarities)
        max_index = np.argmax(similarities)
        if max_similarity >= threshold:
            name = labels[max_index]
        else:
            name = "Unknown"
        results.append((name, max_similarity, bbox))
    return results

def run_webcam_recognition(embeddings_folder="Embeddings", threshold=0.4):
    """Run real-time face recognition using webcam (multi-face)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    plt.ion()
    fig, ax = plt.subplots()
    im = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break

        results = recognize_faces_from_frame(frame, embeddings_folder, threshold)
        display_frame = frame.copy()

        for name, similarity, bbox in results:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{name} ({similarity*100:.1f}%)"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                display_frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 5, y1),
                (0, 0, 0), -1
            )
            cv2.putText(display_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        if im is None:
            im = ax.imshow(img_rgb)
            plt.axis('off')
        else:
            im.set_data(img_rgb)
        plt.title("Face Recognition System")
        plt.pause(0.001)

    cap.release()
    plt.ioff()
    plt.close()

def main(image_path, embeddings_folder="Embeddings"):
    """Main function to recognize a face and save the image with details."""
    image_path = Path(image_path)
    output_path = image_path.parent / f"output_{image_path.name}"
    
    person_name, similarity, img_bgr, bbox = recognize_face(image_path, embeddings_folder)
    if person_name is None:
        logger.error("Recognition failed")
        if img_bgr is not None:
            cv2.imwrite(str(output_path), img_bgr)
            logger.info(f"Saved original image to: {output_path}")
    else:
        logger.info(f"Recognized: {person_name} (Similarity: {similarity:.4f})")
        save_image_with_details(img_bgr, person_name, similarity, bbox, output_path)

if __name__ == "__main__":
    # To test with an image, uncomment the next two lines:
    # test_image = "path/to/test/file"  # Ensure this file exists
    # main(test_image)

    # To use webcam for real-time recognition, call:
    run_webcam_recognition("Embeddings")
