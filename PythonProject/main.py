import numpy as np
import cv2

# Dosya yolları
video_path = '3.mp4'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.99

# Sınıf isimleri ve renkler
classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Modeli yükleme
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Video dosyasını açma
cap = cv2.VideoCapture(video_path)

# İlk kareyi alarak başlatıyoruz
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Geçerli kareyi griye çeviriyoruz
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # İki kare arasındaki farkı alarak hareketi belirliyoruz
    frame_diff = cv2.absdiff(prev_frame_gray, curr_frame_gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Eşik değerinin üzerindeki pikselleri sayarak hareket miktarını buluyoruz
    movement_amount = np.sum(thresh) / 255
    movement_threshold = 5000  # Hareket algılama eşiği

    if movement_amount > movement_threshold:
        # Hareket varsa nesne tespiti yap
        height, width = curr_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(curr_frame, (300, 300)), 0.007, (300, 300), 130)
        net.setInput(blob)
        detected_objects = net.forward()

        for i in range(detected_objects.shape[2]):
            confidence = detected_objects[0, 0, i, 2]
            if confidence > min_confidence:
                class_index = int(detected_objects[0, 0, i, 1])

                upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
                cv2.rectangle(curr_frame, (upper_left_x, upper_left_y),
                              (lower_right_x, lower_right_y),
                              colors[class_index], 3)
                cv2.putText(
                    curr_frame,
                    prediction_text,
                    (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2
                )

        cv2.putText(curr_frame, "Hareket Bulundu!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(curr_frame, "Hareket Yok", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Obje Bulundu", curr_frame)

    # ESC (27) tuşu ile çıkış
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Geçerli kareyi bir sonraki döngü için önceki kare olarak ayarla
    prev_frame_gray = curr_frame_gray

cap.release()
cv2.destroyAllWindows()
