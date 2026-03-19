import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIG =================

SHIRT_THRESHOLD = 35
PANT_THRESHOLD = 35

UNIFORM_REF_IMAGE = "n.jpg"

# ================= LOAD MODEL =================

model = YOLO("yolov8n.pt")

# ================= COLOR FUNCTIONS =================

def dominant(img):

    if img.size == 0:
        return np.array([0,0,0],dtype=np.float32)

    lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    return np.median(lab.reshape(-1,3),axis=0).astype(np.float32)


def get_distance(c1,c2):

    weights = np.array([0.5,1.2,1.2])
    return np.sqrt(np.sum(((c1-c2)*weights)**2))


# ================= LOAD UNIFORM REFERENCE =================

def load_uniform_reference():

    sample = cv2.imread(UNIFORM_REF_IMAGE)

    if sample is None:
        raise Exception("Uniform reference image not found")

    h,w,_ = sample.shape

    shirt = dominant(sample[int(h*0.2):int(h*0.5),int(w*0.3):int(w*0.7)])
    pant  = dominant(sample[int(h*0.55):int(h*0.9),int(w*0.3):int(w*0.7)])

    return shirt,pant


REF_SHIRT,REF_PANT = load_uniform_reference()

# ================= ID CARD DETECTION =================

def detect_id(crop):

    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,2
    )

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)

        aspect_ratio = float(w)/h if h>0 else 0
        area_ratio = (w*h)/(crop.shape[0]*crop.shape[1])

        if 0.5 < aspect_ratio < 2.0 and 0.01 < area_ratio < 0.15:
            return True

    return False


# ================= MAIN DETECTION FUNCTION =================

def detect_uniform(frame):

    """
    Input:
        frame (camera image)

    Returns:
        uniform_detected (bool)
        id_detected (bool)
        frame_with_boxes
    """

    uniform_detected = False
    id_detected = False

    results = model(frame,conf=0.5,classes=[0])

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            person = frame[y1:y2,x1:x2]

            if person.size == 0:
                continue

            ph,pw = person.shape[:2]

            shirt_roi = person[int(ph*0.2):int(ph*0.5),int(pw*0.25):int(pw*0.75)]
            pant_roi  = person[int(ph*0.55):int(ph*0.9),int(pw*0.25):int(pw*0.75)]

            shirt_c = dominant(shirt_roi)
            pant_c  = dominant(pant_roi)

            sd = get_distance(shirt_c,REF_SHIRT)
            pd = get_distance(pant_c,REF_PANT)

            uniform_detected = sd < SHIRT_THRESHOLD and pd < PANT_THRESHOLD

            id_roi = person[int(ph*0.2):int(ph*0.5),int(pw*0.2):int(pw*0.8)]
            id_detected = detect_id(id_roi)

            color = (0,255,0) if uniform_detected else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    return uniform_detected,id_detected,frame