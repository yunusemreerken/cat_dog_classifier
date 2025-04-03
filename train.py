import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score

# set the directory dataset
dataset_path = "dataset/"

#Select images and labels
X=[] #Features( Images )
y=[] #Labels (0 = Cat, 1 = Dog)

for category in ["Cat", "Dog"]:
    if category == "Cat":
        label = 0  # Kedi için 0 kullanıyoruz
    else:
        label = 1  # Köpek için 1 kullanıyoruz
    folder_path = os.path.join(dataset_path, category)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image_name)
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)#Image Gray Scale

        if image is None:
            print(f"Error: Image not uploaded -> {image_path}")
        else:
            image = cv2.resize(image,(64, 64))# Resize image (64 * 64)
            X.append(image.flatten()) # 2D -> 1D
            y.append(label)
            

#Convert array to numpy array
X = np.array(X)
y = np.array(y)


#data seperate train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Eğer X_train düz bir listeyse, reshape yap
if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, 1)



print(np.unique(y_train))  # Yalnızca 1 sınıf var mı kontrol edin
#create model and train model
model = lr(max_iter=1000)
model.fit(X_train,y_train)

#modeli değerlendir
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")

#save the model
with open("cat_dog_model.pkl","wb") as f:
    pickle.dump(model,f)


print("Model başarıyla kaydedildi! Tebrikler")
