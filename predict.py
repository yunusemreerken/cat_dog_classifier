import cv2 
import numpy as np
import pickle

# Modeli yükle
with open("cat_dog_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_image(image):
    """ Görüntüyü modelin anlayacağı formata çevirip tahmin yapar """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Gri tona çevir
    image = cv2.resize(image, (64, 64))  # Yeniden boyutlandır
    image = image.flatten().reshape(1, -1)  # Tek satırlık vektöre çevir
    
    prediction = model.predict(image)
    return "Cat" if prediction[0] == 0 else "Dog"

# Kamerayı aç
cap = cv2.VideoCapture(0)
prediction_text = "Press 'S' to predict"  # Başlangıç mesajı

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Kameradan görüntü alınamıyor.")
        break

    # Tahmin sonucunu ekrana yazdır
    cv2.putText(frame, prediction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Canlı görüntüyü göster
    cv2.imshow("Kamera", frame)

    key = cv2.waitKey(1) & 0xFF

    # "S" tuşuna basılırsa görüntüyü kaydet ve tahmin yap
    if key == ord("s"):
        cv2.imwrite("dataset/Prediction/cat_or_dog.jpg", frame)
        prediction_text = f"Prediction: {predict_image(frame)}"  # Ekrana yazdırılacak tahmin sonucu
        print(prediction_text)  # Konsola da yazdır

    # "Q" tuşuna basılırsa çık
    elif key == ord("q"):
        print("Çıkılıyor...")
        break

# Kamera ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
