import easyocr
import cv2
import cvzone
import ultralytics
import streamlit as st
import os
import moviepy.editor as mpe

st.title("Number plate recognition 🚗🏍️")
st.write(":orange[Note:] This model works better with car pictures, because of the dataset mainly contained car pictures.")

# Setting up the easyocr module
text_reader = easyocr.Reader(['en'])

os.makedirs("Detections", exist_ok=True)

file_type = st.radio("Select file type", ["Image", "Video"])
file = st.file_uploader("Choose your file")

if file_type == "Image":
    if file:
        # img = Image.open(file)
        with open("saved_img.jpg", "wb") as f:
            f.write(file.getvalue())
        st.image(file, width=500)
else:
    if file:
        with open("saved_video.mp4", "wb") as f:
            f.write(file.getvalue())
        st.video("saved_video.mp4")


@st.cache_resource
def load_model():
    return ultralytics.YOLO("best.pt")


# Prediction/Detection function

def get_prediction(model, img=None, video=None):
    number_plates = set()
    if img:
        img = cv2.imread(img)
        results = model.predict(source=img)

        for result in results:
            for box in result.boxes:
                # finding x, y, w, h of bounding boxes
                bbox = box.xyxy[0]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                # Cropping the number plate portion
                no_plate = img[y1:y2, x1: x2]
                # Convert to grayscale
                no_plate = cv2.cvtColor(no_plate, cv2.COLOR_BGR2GRAY)
                no_plate = 255 - no_plate  # Inverting the image

                # Using OCR to get the Vehicle Number Plate
                result = text_reader.readtext(no_plate)
                txt = ""
                for _, text, _ in result:
                    txt += text
                txt = [i for i in list(txt) if i.isalnum()]
                txt = "".join(txt).upper()

                scaling_factor = (img.shape[1] // 1000) + 1

                # Plotting bounding box and number_plate
                if len(txt) >= 5:
                    number_plates = number_plates.union({txt})
                    img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                        color=(252, 82, 3), thickness=scaling_factor)
                    cvzone.putTextRect(img, txt, pos=(x1+5, y1-10),
                                       scale=scaling_factor, thickness=scaling_factor, colorR=(252, 82, 3))
                else:
                    img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                        color=(252, 215, 3), thickness=scaling_factor)
                    cvzone.putTextRect(img, "number_plate", pos=(x1+5, y1-10),
                                       scale=scaling_factor, thickness=scaling_factor, colorR=(252, 215, 3), colorT=(0, 0, 0))

        cv2.imwrite("Detections/pred.jpg", img)   # Saving the image

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, number_plates
    else:
        vid = cv2.VideoCapture(video)
        width = int(vid.get(3))
        height = int(vid.get(4))

        size = (width, height)
        ouput_video = cv2.VideoWriter('Detections/pred.avi',
                                      cv2.VideoWriter_fourcc(*'MJPG'),
                                      20.0, size)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                results = model.predict(source=frame)
                for result in results:
                    for box in result.boxes:
                        # finding x, y, w, h of bounding boxes
                        bbox = box.xyxy[0]

                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])

                        # Cropping the number plate portion
                        no_plate = frame[y1:y2, x1: x2]
                        no_plate = cv2.cvtColor(no_plate, cv2.COLOR_BGR2GRAY)
                        no_plate = 255 - no_plate
                        # Plotting bounding box

                        # Using OCR to get the Vehicle Number Plate
                        result = text_reader.readtext(no_plate)
                        txt = ""
                        for _, text, _ in result:
                            txt += text
                        txt = [i for i in list(txt) if i.isalnum()]
                        txt = "".join(txt).upper()
                        scaling_factor = (frame.shape[1] // 1000) + 1

                        # Plotting bounding box and number_plate
                        if len(txt) >= 5:
                            number_plates = number_plates.union({txt})
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2),
                                                  color=(252, 82, 3), thickness=scaling_factor)
                            cvzone.putTextRect(frame, txt, pos=(x1+5, y1-10),
                                               scale=scaling_factor, thickness=scaling_factor, colorR=(252, 82, 3))
                        else:
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2),
                                                  color=(252, 215, 3), thickness=scaling_factor)
                            cvzone.putTextRect(frame, "number_plate", pos=(x1+5, y1-10),
                                               scale=scaling_factor, thickness=scaling_factor, colorR=(252, 215, 3), colorT=(0, 0, 0))
            else:
                vid.release()
                ouput_video.release()

            ouput_video.write(frame)   # Write the frame on the output_video
        return number_plates


if file:
    if st.button("Detect"):
        model = load_model()
        if file_type == "Image":
            with st.spinner("Detecting"):
                pred_img, number_plates = get_prediction(
                    model, img="saved_img.jpg")
                st.image(pred_img)
        else:
            with st.spinner("Detecting"):
                number_plates = get_prediction(
                    model, video="saved_video.mp4")

                # Converting '.avi' file to '.mp4'
                clip = mpe.VideoFileClip("Detections/pred.avi")
                clip.write_videofile("Detections/pred.mp4")
                st.video("Detections/pred.mp4")

        # Writing the detected number plates on the file
        with open("Detections/number_plates.txt", "w") as f:
            for i, no_plate in enumerate(number_plates):
                f.write(f"{i+1} {no_plate}\n")

        st.write("Results saved in dir named 'Detections'")
