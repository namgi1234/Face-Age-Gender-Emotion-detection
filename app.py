import PIL.Image
import streamlit as st
import PIL
import cv2
import numpy
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_web(image):
    faces = face_cascade.detectMultiScale(
    image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(21,149,21), thickness=2)
        result=DeepFace.analyze(image, enforce_detection=False)
        sex = 'Female'
        if result[0]['dominant_gender'] == 'Man':
            sex = 'male'
        output = f'{sex} {str(result[0]['age'])} {result[0]['dominant_emotion']}'
        cv2.putText(image,output,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(21,149,21),1,cv2.LINE_AA)
    return image, faces

st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon="😁",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Age/Gender/Emotion : 😁")

st.header("Type")
source_radio = st.sidebar.radio("Select Source",["image","webcam"])

input = None
if source_radio == "image":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image",type=("jpg","png"))

    # if input is not None:
    #     upload_image = PIL.Image.open(input)
    #     visualized_image = img.image_detection(upload_image)
    #     st.image('1.png')
    #     st.image(upload_image)

elif source_radio =="webcam":
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while run:
        # Reading image from video stream
        _, img = camera.read()
        # Call method we defined above
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, a = detect_web(img)
        # st.image(img, use_column_width=True)
        FRAME_WINDOW.image(img)