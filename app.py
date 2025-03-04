import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

# Constants
CONFIDENCE_THRESHOLD = 0.1  # Reduced threshold to allow more detections
NMS_THRESHOLD = 0.3
CLASSES = ["Mature Cotton", "Immature Cotton"]  # Update with actual class names

# Load ONNX Model
@st.cache_resource()
def load_model():
    session = ort.InferenceSession("best.onnx")
    return session

# Preprocess Image
def preprocess_image(image):
    image = image.resize((640, 640))
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    image_array = image_array.transpose(2, 0, 1)  # Convert to (C, H, W)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return image_array

# Run Model
def run_inference(session, image_array):
    inputs = {"images": image_array}
    outputs = session.run(None, inputs)
    return outputs

# Process Detections
def process_detections(outputs):
    detections = np.squeeze(outputs[0])  # Remove batch dimension
    if detections.shape[0] < 9:
        st.error("⚠️ Unexpected model output shape. Check the ONNX model.")
        return [], [], []
    
    boxes = detections[0:4, :].T  # Extract bounding boxes
    scores = detections[4, :]  # Extract confidence scores
    class_ids = np.argmax(detections[5:, :], axis=0)  # Extract class IDs

    # Apply confidence threshold
    valid_indices = np.where(scores > CONFIDENCE_THRESHOLD)[0]
    boxes, scores, class_ids = boxes[valid_indices], scores[valid_indices], class_ids[valid_indices]
    
    return boxes, scores, class_ids

# Draw Bounding Boxes
def draw_boxes(image, boxes, scores, class_ids):
    if len(class_ids) == 0:
        st.error("⚠️ No objects detected. Try another image.")
        return image
    
    image = np.array(image)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(image)

# Streamlit UI
st.title("Cotton Maturity Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    session = load_model()
    image_array = preprocess_image(image)
    outputs = run_inference(session, image_array)
    
    # Debugging output
    st.write(f"Model Output Shape: {outputs[0].shape}")
    st.write("Raw Model Output (Debugging):")
    for i, row in enumerate(outputs[0][:9]):
        st.write(f"Detection {i}: {row}")
    
    boxes, scores, class_ids = process_detections(outputs)
    detected_image = draw_boxes(image, boxes, scores, class_ids)
    st.image(detected_image, caption="Detected Objects", use_column_width=True)
