import cv2
import numpy as np
import os
from PIL import Image


def create_user(f_id, name):
    web = cv2.VideoCapture(0)
    web.set(3, 640)
    web.set(4, 480)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    f_dir = 'dataset'
    path = os.path.join(f_dir, str(f_id))  # store by ID, not name
    os.makedirs(path, exist_ok=True)

    counter = 0
    while True:
        ret, img = web.read()
        if not ret:
            print("Error: Could not access webcam.")
            return
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        multi_face = faces.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in multi_face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            counter += 1

            filename = f"user.{f_id}.{counter}.jpg"
            cv2.imwrite(os.path.join(path, filename), gray[y:y+h, x:x+w])
            cv2.imshow("Image", img)

        k = cv2.waitKey(100) & 0xFF
        if k == 27 or counter >= 100:
            break

    web.release()
    cv2.destroyAllWindows()


create_user(1,"roy")

def train():
    database = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faceSamples, ids = [], []

    for root, dirs, files in os.walk(database):
        for file in files:
            if file.startswith("user.") and file.endswith(".jpg"):
                path = os.path.join(root, file)
                PIL_image = Image.open(path).convert('L')
                img_numpy = np.array(PIL_image, 'uint8')

                # Extract ID from filename
                id = int(file.split('.')[1])  

                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id)

    recognizer.train(faceSamples, np.array(ids))
    recognizer.write('trainer.yml')
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program")
    return len(np.unique(ids))
train()

def recognize(names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    cascadepath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadepath)

    font = cv2.FONT_HERSHEY_SIMPLEX
    name = ""
    face_count = 0

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Could not access webcam.")
            break

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Lower confidence = better match
            if confidence < 70:
                label = names.get(id, "unknown")
            else:
                label = "unknown"

            # Track consecutive recognitions
            if name == label:
                face_count += 1
            else:
                name = label
                face_count = 0

            cv2.putText(img, str(label), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, f"{round(100-confidence)}%", (x+5, y+h-5), font, 1, (255, 255, 255), 1)

        cv2.imshow("Camera", img)
        k = cv2.waitKey(27) & 0xFF
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
            
recognize({1:'roy'})