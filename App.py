import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
from PIL import Image
import pandas as pd
import datetime

from uniform_detection import detect_uniform


# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="AI Student Attendance System",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------
# Load Models
# ------------------------------

@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    svm_model = joblib.load("models/svm_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return detector, embedder, svm_model, encoder

detector, embedder, svm_model, encoder = load_models()

threshold = 0.60


# ------------------------------
# CSV Paths
# ------------------------------

person_csv = "Person_info.csv"
mapping_csv = "models/student_mapping.csv"


# ------------------------------
# Load CSV Files
# ------------------------------

if os.path.exists(person_csv):
    students = pd.read_csv(person_csv)
else:
    students = pd.DataFrame(columns=["Student_ID","Person_name","Image_path"])

if os.path.exists(mapping_csv):
    student_map = pd.read_csv(mapping_csv)
else:
    student_map = pd.DataFrame(columns=["Student_ID","Name"])


# ------------------------------
# Tabs
# ------------------------------

tab1,tab2,tab3 = st.tabs([
    "📷 Capture Attendance",
    "🧑 Register New Student",
    "📊 Attendance Report"
])

# =====================================================
# TAB 1 : ATTENDANCE
# =====================================================

with tab1:

    st.subheader("Capture Photo for Attendance")

    camera_image = st.camera_input("Take Photo")

    if camera_image is not None:

        image = Image.open(camera_image)
        image = np.array(image)

        img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        faces = []

        try:
            if image is not None:
                faces = detector.detect_faces(image)
        except:
            st.warning("Face detection failed")

        detected_students = []

        for face in faces:

            x,y,w,h = face['box']

            x = max(0,x)
            y = max(0,y)

            face_img = image[y:y+h,x:x+w]

            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img,(160,160))

            embedding = embedder.embeddings([face_img])

            preds = svm_model.predict(embedding)
            prob = svm_model.predict_proba(embedding)

            confidence = np.max(prob)

            name = encoder.inverse_transform(preds)[0]

            # ------------------------------
            # SAME FACE + UNIFORM + ID LOGIC
            # ------------------------------

            if confidence >= threshold:

                row = student_map[student_map["Name"]==name]

                if not row.empty:

                    student_id = row.iloc[0]["Student_ID"]

                    uniform,id_card,img = detect_uniform(img)

                    if uniform and id_card:

                        detected_students.append(student_id)

                        label = f"{name} ({confidence*100:.2f}%)"
                        color = (0,255,0)

                    else:

                        label = f"{name} - Uniform/ID Missing"
                        color = (0,0,255)

                else:

                    label = "Match Not Found"
                    color = (0,0,255)

            else:

                label = "Match Not Found"
                color = (0,0,255)

            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

            cv2.putText(img,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,color,2)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        st.image(img)

        # ------------------------------
        # Attendance Marking
        # ------------------------------

        today = datetime.date.today()

        attendance = []

        for _,row in student_map.iterrows():

            sid = row["Student_ID"]
            name = row["Name"]

            status = "Present" if sid in detected_students else "Absent"

            attendance.append({
                "Student_ID":sid,
                "Name":name,
                "Date":today,
                "Status":status
            })

        df = pd.DataFrame(attendance)

        df.to_excel("attendance.xlsx",index=False)

        st.success("Attendance Updated")


# =====================================================
# TAB 2 : REGISTER NEW STUDENT
# =====================================================

with tab2:

    st.subheader("Register New Student")

    student_id = st.text_input("Student ID")
    name = st.text_input("Student Name")

    if st.button("Start Image Capture"):

        if student_id and name:

            dataset_path = "dataset/" + name
            os.makedirs(dataset_path,exist_ok=True)

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Camera could not be opened")
            else:

                count = 0
                max_images = 20

                st.info("Press 'c' to capture image. Capture at least 20 images.")

                while True:

                    ret, frame = cap.read()

                    if not ret or frame is None:
                        break

                    faces = []

                    try:
                        faces = detector.detect_faces(frame)
                    except:
                        faces = []

                    face_img = None

                    for face in faces:

                        x,y,w,h = face['box']

                        x = max(0,x)
                        y = max(0,y)

                        face_img = frame[y:y+h,x:x+w]

                        if face_img.size == 0:
                            continue

                        face_img = cv2.resize(face_img,(160,160))

                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                        cv2.imshow("Face Preview",face_img)

                    cv2.putText(frame,"Look Straight / Left / Right / Up / Down",
                                (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,255,255),
                                2)

                    cv2.putText(frame,f"Images Captured: {count}/{max_images}",
                                (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255,0,0),
                                2)

                    cv2.imshow("Dataset Creator",frame)

                    key = cv2.waitKey(1)

                    if key == ord('c') and len(faces) > 0 and face_img is not None:

                        img_path = dataset_path + "/" + str(count) + ".jpg"

                        cv2.imwrite(img_path,face_img)

                        count += 1

                    if count >= max_images:
                        break

                    if key == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

                st.success("Image Capture Completed")


    # ------------------------------
    # SUBMIT REGISTRATION
    # ------------------------------

    if st.button("Submit Registration"):

        if student_id and name:

            dataset_path = "dataset/" + name

            if not os.path.exists(dataset_path):
                st.error("Capture images first")
            else:

                images = os.listdir(dataset_path)

                new_rows = []

                for img in images:

                    path = dataset_path + "/" + img

                    new_rows.append({
                        "Student_ID":student_id,
                        "Name":name,
                        "Image_path":path
                    })

                new_df = pd.DataFrame(new_rows)

                students_updated = pd.concat([students,new_df],ignore_index=True)

                students_updated.to_csv(person_csv,index=False)

                new_map = pd.DataFrame([{
                    "Student_ID":student_id,
                    "Name":name
                }])

                student_map_updated = pd.concat([student_map,new_map],ignore_index=True)

                student_map_updated.to_csv(mapping_csv,index=False)

                st.success("Student Registered Successfully")


# ------------------------------
# Attendance Marking (ONLY PRESENT)
# ------------------------------

today = datetime.date.today()
now = datetime.datetime.now().strftime("%H:%M:%S")

attendance_rows = []

# Remove duplicate student IDs from same frame
detected_students = list(set(detected_students))

for sid in detected_students:

    row = student_map[student_map["Student_ID"] == sid]

    if not row.empty:

        name = row.iloc[0]["Name"]

        attendance_rows.append({
            "Student_ID": sid,
            "Name": name,
            "Date": today,
            "Time": now,
            "Status": "Present"
        })

# ------------------------------
# Only proceed if students detected
# ------------------------------

if len(attendance_rows) > 0:

    df_new = pd.DataFrame(attendance_rows)

    file_path = "attendance.xlsx"

    # Append to existing file
    if os.path.exists(file_path):

        try:
            df_old = pd.read_excel(file_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df_final = df_new

    else:
        df_final = df_new

    # Remove duplicates (same student same day)
    df_final.drop_duplicates(
        subset=["Student_ID", "Date"],
        keep="last",
        inplace=True
    )

    df_final.to_excel(file_path, index=False)

    st.success(f"{len(attendance_rows)} Student(s) Marked Present")

else:

    st.warning("No valid students detected. Attendance not updated.")