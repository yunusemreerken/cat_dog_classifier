import cv2 
import numpy as np
import pickle
import time

with open("cat_dog_model.pkl","rb") as f:
    model = pickle.load(f)

def predict_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Image not uploaded -> {image_path}")
    else:
        image = cv2.resize(image,(64, 64))# Resize image (64 * 64)
        image = image.flatten().reshape(1, -1)  # 2D -> 1D

    prediction = model.predict(image)
    return "Cat" if prediction[0] == 0 else "Dog"


test_image = 'dataset/Prediction/cat_or_dog.jpg'
print(f"Tahmin: {predict_image(test_image)}")
                
# Tüm pencereleri kapat
cv2.destroyAllWindows()



