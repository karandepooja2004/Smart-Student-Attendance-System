# 🎓 Smart Attendance System

An AI-powered attendance system that uses **Face Recognition + Uniform Detection + ID Detection** to automatically mark student attendance.

---

## 🚀 Features

### ✅ Face Recognition

* Uses **MTCNN** for face detection
* Uses **FaceNet** for embedding generation
* Uses **SVM model** for classification

### ✅ Uniform Detection

* Detects whether student is wearing proper uniform

### ✅ ID Card Detection

* Verifies if student is wearing ID card

### ✅ Smart Attendance System

* Marks attendance **only when all conditions are satisfied**:

  * Face recognized
  * Uniform detected
  * ID card detected

### ✅ Automatic Excel Report

* Saves attendance in `attendance.xlsx`
* Stores:

  * Student_ID
  * Name
  * Date
  * Time
  * Status (Present)

### ✅ Register New Student

* Capture **multiple face angles (20+ images)**
* Automatically:

  * Saves images in dataset folder
  * Updates CSV files

### ✅ Streamlit GUI

* Clean and interactive UI
* Tabs:

  * 📷 Capture Attendance
  * 🧑 Register Student
  * 📊 Attendance Report

---

## 🧠 Technologies Used

* Python
* Streamlit
* OpenCV
* MTCNN (Face Detection)
* FaceNet (Embeddings)
* Scikit-learn (SVM)
* Pandas
* NumPy

---

## 📂 Dataset & CSV Creation
### 📸 Dataset Creation

* Run FaceCapture.py file
* Create a folder: dataset/
* Inside it, create subfolders for each student:
  
    dataset/Rahul/

    dataset/Priya/

* Capture 20–50 face images per student using camera
* Images are automatically saved in respective folders

### 📄 CSV Files
1. Person_info.csv

* Stores training data (image paths with labels)

  EX -
  
    Student_ID,Person_name,Image_path

    101,Rahul,dataset/Rahul/0.jpg
  
    101,Rahul,dataset/Rahul/1.jpg
