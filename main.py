import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

def augment_image(img):
    flipped = cv2.flip(img, 1)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return [img, flipped, rotated]

# Очистка логов
with open("predictions.log", "w") as log_file:
    log_file.write("---- NEW SESSION ----\n")

# Загрузка и аугментация данных
data_dir = 'data'
images, labels = [], []

for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    if not os.path.isdir(class_dir): continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        for aug in augment_image(img):
            resized = cv2.resize(aug, (64, 64))
            images.append(resized.flatten())
            labels.append(label)

X = np.array(images)
le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2f}")

# Видеопоток и лог
cap = cv2.VideoCapture(2)
log_file = open("predictions.log", "a")
mode = "recognition"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f'snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, frame)
    elif key == ord('m'):
        mode = "camera" if mode == "recognition" else "recognition"

    display_frame = frame.copy()

    if mode == "recognition":
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > 50 and h > 50:
                roi = frame[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (64, 64))
                gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                input_data = gray.flatten().reshape(1, -1)

                neighbors = model.kneighbors(input_data, return_distance=True)
                dists = neighbors[0][0]
                idxs = neighbors[1][0]
                votes = y_train[idxs]
                counts = np.bincount(votes)
                pred_class = np.argmax(counts)
                confidence = counts[pred_class] / 3.0

                label = le.inverse_transform([pred_class])[0]
                if confidence > 0.6:
                    # Рамка
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Текст в верхнем левом углу
                    cv2.putText(display_frame, f'{label} ({confidence*100:.1f}%)', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f'{timestamp} | {label} | Confidence: {confidence:.2f}\n')
                    log_file.flush()

    # GUI-информация
    cv2.putText(display_frame, f'Mode: {mode}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, 'Keys: q=Quit, s=Save, m=Mode', (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Camera', display_frame)

cap.release()
log_file.close()
cv2.destroyAllWindows()
