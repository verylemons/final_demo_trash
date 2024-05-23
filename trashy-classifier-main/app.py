import threading
from ultralytics import YOLO
import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from streamlit_webrtc import webrtc_streamer
import av

#####################################################################################################################
# DEPLOY MODEL!!!!!!

# Load model
model = YOLO("best.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# function to detect


def detect_objects(image):
    # image = Image.fromarray(image).convert('RGB')
    image = image.to_image()
    result = model.predict(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=image,
        detections=detections
    )

    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )

    return av.VideoFrame.from_image(annotated_image)
#####################################################################################################################

#####################################################################################################################
# OUTLINE OF WEBSITE !!!!!!!!

# INDIVIDUAL TABS


def infoTAB():
    st.markdown(":blue-background[**ABOUT OUR PROJECT**]")
    st.markdown("*Group Members: Kaleb Ugalde, Parth Sandeep, Vidhi Oswal, Shashwat Dudeja, Kwae Htoo*")
    st.markdown("")
    st.markdown("""
                We decided to do a computer vision project and create multi-object detector that is able to identify trash (e.g platic, bottles, etc).
                After finilzing on our idea, we researched online on datasets and already existing models that we could use.
                The end result was using a trash dataset from Kaggle and using the YOLO model for multi-object detection.
                To put everything together, we use PyTorch to implement our model and train on model on the dataset from Kaggle.
                Finally, we made a simple website using Streamlit to deploy our model.
                """)
    st.markdown("")

    st.markdown(":blue-background[**RESULTS**]")
    st.image('results/F1_curve.png')
    st.image('results/labels.jpg')
    st.image('results/P_curve.png')
    st.image('results/val_batch1_labels.jpg')
    st.image('results/val_batch1_pred.jpg')
    


def demoTAB():
    st.title("Trashy Classifier Model")
    option = st.radio("Choose Input Method:", ('Upload Image', 'Use Camera'))
    if option == 'Upload Image':
        uploaded_image = st.file_uploader(
            'Choose an image...', type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            results = detect_objects(image)
            st.image(results)
    else:
        st.title("Webcam Live Feed")
        # run = st.button('Run')
        # end = st.button('End Feed')
        # FRAME_WINDOW = st.image([])
        webrtc_streamer(key="sample", video_frame_callback=detect_objects)
        # camera = cv2.VideoCapture(0)
        # while run:
        #     _, frame = camera.read()
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     results = detect_objects(frame)
        #     FRAME_WINDOW.image(results)
        # if end:
        #     st.write('Webcam Stopped')
        #     camera.release()
        #     cv2.destroyAllWindows()


#####################################################################################################################
#  RUN STREAMLIT !!!!!!!!!
def main():
    tab_select = st.sidebar.radio("Welcome! Where would you like to explore?", (
        "About Our Project", "Try Out Our Model"))
    if tab_select == "About Our Project":
        infoTAB()
    elif tab_select == "Try Out Our Model":
        demoTAB()


if __name__ == '__main__':
    main()
#####################################################################################################################
