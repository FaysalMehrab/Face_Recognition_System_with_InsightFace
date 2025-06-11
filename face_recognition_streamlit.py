import streamlit as st
import cv2
import numpy as np
import insightface
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import time

# Initialize InsightFace
@st.cache_resource
def load_model():
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app

app = load_model()

def load_saved_embeddings(embeddings_folder):
    embeddings_folder = Path(embeddings_folder)
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

def recognize_faces_from_frame(frame, embeddings_folder, threshold=0.4):
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

def draw_door_light_img(open_door=False, status_light=None, width=120, height=160):
    """Draw a modern door and status indicator with crimson door color."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    door_color = (60, 20, 220)  # Crimson in BGR (approx: #DC143C)
    frame_color = (107, 114, 128)  # Gray-500
    thickness = 4

    # Draw status light
    if status_light == "green":
        cv2.circle(img, (width // 2, int(height * 0.15)), 14, (34, 197, 94), -1)  # Green-500
    elif status_light == "red":
        cv2.circle(img, (width // 2, int(height * 0.15)), 14, (239, 68, 68), -1)  # Red-500

    # Draw door frame
    cv2.rectangle(img, (20, 30), (width-20, height-10), frame_color, thickness)
    
    # Draw door
    if open_door:
        # Draw open door (perspective)
        pts = np.array([
            [int(width*0.3), int(height*0.35)],
            [int(width*0.8), int(height*0.3)],
            [int(width*0.8), int(height*0.85)],
            [int(width*0.3), int(height*0.8)]
        ], np.int32)
        cv2.fillPoly(img, [pts], door_color)
    else:
        # Draw closed door
        cv2.rectangle(img, (int(width*0.3), int(height*0.35)), 
                     (int(width*0.8), int(height*0.85)), door_color, -1)
        # Door handle
        cv2.circle(img, (int(width*0.72), int(height*0.6)), 6, (250, 204, 21), -1)  # Yellow-400
    
    return img

# --- Professional UI Design ---
st.set_page_config(
    page_title="SecureFace Access",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.image("https://via.placeholder.com/150x40?text=SecureFace", use_container_width=True)
    st.subheader("System Configuration")
    
    # System settings
    threshold = st.slider("Recognition Threshold", 0.3, 0.9, 0.5, 0.05,
                         help="Higher values increase security but may cause false negatives")
    door_open_duration = st.slider("Door Open Duration (seconds)", 1, 10, 3, 1,
                                  help="How long the door stays open after recognition")
    
    st.divider()
    st.subheader("System Indicators")
    st.markdown("""
        - **Green**: Access granted  
        - **Red**: Unrecognized person  
        - **No light**: Standby mode
    """)

# Main Content Area
st.title("SecureFace Access Control")
st.caption("AI-powered facial recognition for secure physical access management")

# Status indicator
status_container = st.container()
status_msg = status_container.empty()

# Video and door display
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Live Camera Feed")
    camera_placeholder = st.empty()

with col2:
    st.subheader("Access Status")
    door_placeholder = st.empty()
    with st.expander("System Indicators"):
        st.markdown("""
        - **Green**: Access granted
        - **Red**: Unrecognized person
        - **No light**: Standby mode
        """)

# Control buttons
controls = st.columns(3)
with controls[0]:
    start_btn = st.button("▶️ Start Recognition", type="primary", use_container_width=True)
with controls[1]:
    stop_btn = st.button("⏹️ Stop System", use_container_width=True)
with controls[2]:
    if st.button("ℹ️ System Info", use_container_width=True):
        st.info("""
        **SecureFace Access Control v1.2**  
        - Powered by InsightFace Recognition
        - Real-time facial authentication
        - Enterprise-grade security
        """)

# Initialize session state
if 'run' not in st.session_state:
    st.session_state.run = False
if 'door_opened' not in st.session_state:
    st.session_state.door_opened = False
if 'door_opened_time' not in st.session_state:
    st.session_state.door_opened_time = 0
if 'last_status' not in st.session_state:
    st.session_state.last_status = None
if 'last_recognized' not in st.session_state:
    st.session_state.last_recognized = "None"
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'door_status' not in st.session_state:
    st.session_state.door_status = "Closed"

# Button handlers
if start_btn:
    st.session_state.run = True
if stop_btn:
    st.session_state.run = False

# Camera processing loop
if st.session_state.run:
    camera = cv2.VideoCapture(0)
    embeddings_folder = "Embeddings"

    while st.session_state.run and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            status_msg.error("Camera feed unavailable. Please check connection.")
            break

        # Process frame
        results = recognize_faces_from_frame(frame, embeddings_folder, threshold)
        open_door = False
        unknown_detected = False

        # --- Track if any face is detected ---
        face_detected = len(results) > 0

        # Reset for each frame
        recognized_name = "None"
        max_confidence = 0.0

        for name, similarity, bbox in results:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            if name != "Unknown":
                color = (59, 246, 175)  # Green for recognized
                text = f"{name} ({similarity*100:.1f}%)"
                text_color = color
                if similarity > max_confidence:
                    recognized_name = name
                    max_confidence = similarity
            else:
                color = (0, 0, 255)  # Red for unknown
                text = f"UNKNOWN ({similarity*100:.1f}%)"
                text_color = color

            # Draw bounding box and label with accuracy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA
            )

            if name != "Unknown":
                open_door = True
            else:
                unknown_detected = True

        # Update recognition info and session state for Access Log
        if face_detected:
            st.session_state.last_recognized = recognized_name
            st.session_state.confidence = max_confidence
        else:
            st.session_state.last_recognized = "None"
            st.session_state.confidence = 0.0

        # Door control logic
        current_time = time.time()
        status = None
        status_light = None
        status_text = ""

        if not face_detected:
            status_light = None
            status = "no_person"
            status_text = """
            <div style='
                background: linear-gradient(to right, #f3f4f6, #e5e7eb);
                padding: 1.2rem;
                border-radius: 12px;
                border-left: 5px solid #64748b;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
                font-family: sans-serif;
                margin: 1rem 0;
            '>
                <h3 style='color: #334155; margin:0;'>No Person Detected</h3>
                <p style='color: #334155; margin:0.5rem 0;'>No person is detected. Door remains closed.</p>
            </div>
            """
            st.session_state.door_opened = False
            st.session_state.door_status = "Closed"
        elif open_door:
            st.session_state.door_opened = True
            st.session_state.door_opened_time = current_time
            status = f"open:{recognized_name}"

        if face_detected and st.session_state.door_opened and (current_time - st.session_state.door_opened_time < door_open_duration):
            status_light = "green"
            status_text = f"""
            <div style='
                background: linear-gradient(to right, #ecfdf5, #d1fae5);
                padding: 1.2rem;
                border-radius: 12px;
                border-left: 5px solid #10b981;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                font-family: sans-serif;
                margin: 1rem 0;
            '>
                <h3 style='color: #065f46; margin:0;'>Access Granted</h3>
                <p style='color: #047857; margin:0.5rem 0;'>Welcome <b>{recognized_name}</b>, door is open</p>
            </div>
            """
            st.session_state.door_status = "Open"
        elif face_detected and unknown_detected and not open_door:
            status_light = "red"
            status = "closed:unknown"
            status_text = """
            <div style='
                background: linear-gradient(to right, #fef2f2, #fee2e2);
                padding: 1.2rem;
                border-radius: 12px;
                border-left: 5px solid #ef4444;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                font-family: sans-serif;
                margin: 1rem 0;
            '>
                <h3 style='color: #b91c1c; margin:0;'>Access Denied</h3>
                <p style='color: #b91c1c; margin:0.5rem 0;'>Unauthorized personnel detected</p>
            </div>
            """
            st.session_state.door_opened = False
            st.session_state.door_status = "Closed"

        # Update UI components
        door_img = draw_door_light_img(
            open_door=(st.session_state.door_status == "Open"),
            status_light=status_light
        )
        door_placeholder.image(
            cv2.cvtColor(door_img, cv2.COLOR_BGR2RGB),
            use_container_width=True  # <-- use_container_width instead of use_column_width
        )

        camera_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True  # <-- use_container_width instead of use_column_width
        )

        if status != st.session_state.last_status:
            status_msg.markdown(status_text, unsafe_allow_html=True)
            st.session_state.last_status = status

        # Add slight delay to reduce CPU usage
        time.sleep(0.05)

    # Cleanup when stopped
    if camera:
        camera.release()
        st.session_state.run = False
        status_msg.info("System stopped. Click 'Start Recognition' to reactivate")
else:
    if stop_btn:
        status_msg.info("System is currently inactive")
    else:
        status_msg.info("Click 'Start Recognition' to begin facial authentication")