import streamlit as st
import subprocess
import os
from PIL import Image

# Get the absolute path of the script's directory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def text_block():
    st.title("Road Sign Detection App")
    st.markdown("### This app detects 4 distinct classes of road signs:")
    st.markdown("- Traffic Light")
    st.markdown("- Stop Sign")
    st.markdown("- Speed Limit Sign")
    st.markdown("- Crosswalk Sign")

@st.cache_data(show_spinner=False)
def load_local_image(image_path):
    """
    Load an image from the local file system.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image.Image or None: The loaded image as a PIL Image object, or None if there was an error reading the image.
    """
    try:
        image = Image.open(image_path)
        return image
    except IOError:
        return None

def predict(uploaded, image_placeholder):
    """
    This function classifies the image and shows the result in Streamlit.
    """
    command = [
        "python", "yolov5/detect.py",
        "--weights", "yolov5/runs/train/exp/weights/best.pt",
        "--img", "640",
        "--conf", "0.4",
        "--iou-thres", "0.45",
        "--source", "uploaded_images/image_to_predict.png",
        "--save-txt",
        "--save-conf"
    ]
    if uploaded:
        st.image(uploaded, caption="Original Image")
        subprocess.run(command)
        image_path = "yolov5/runs/detect/exp/image_to_predict.png"
        image = load_local_image(image_path)
        if image is not None:
            # Display the image using Streamlit
            image_placeholder.image(image, caption="Detected objects", use_column_width=True)
        else:
            st.error("Failed to load the image from the local repository.")
    else:
        st.error("No image has been uploaded")

def save_uploaded_image(uploaded):
    """
    Save the uploaded image to the local file system.

    Parameters:
        uploaded (BytesIO): The uploaded image as a BytesIO object.

    Description:
        This function saves the uploaded image to the 'uploaded_images' directory in the local file system.
        If the directory does not exist, it creates one.
        The image is saved with a unique filename 'image_to_predict.png'.

    Example:
        save_uploaded_image(uploaded_image)
    """

    # Specify the directory where you want to save the image
    save_dir = "uploaded_images"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the uploaded image with a unique filename
    with open(os.path.join(save_dir, 'image_to_predict.png'), "wb") as f:
        f.write(uploaded.getvalue())

def main():
    text_block()
    uploaded = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    
    # Create a placeholder for the detected image
    image_placeholder = st.empty()
    
    if uploaded:
        # Save the uploaded image to a directory
        save_uploaded_image(uploaded)
    
    if st.button("Detect Sign"):
        # Call the predict function and pass the image placeholder
        predict(uploaded, image_placeholder)

if __name__ == '__main__':
    main()
# https://dl2trafficsigns-yggakjlmx2.streamlit.app/
