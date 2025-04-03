import sys
import cv2
import pickle
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class CatDogTrainer(QWidget):
    def __init__(self):
        super().__init__()

        # Model dosyası kontrolü
        self.model_path = "cat_dog_model.pkl"
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = LogisticRegression()

        # Veri seti
        self.dataset_path = "dataset.pkl"
        self.X_data, self.y_data = self.load_dataset()

        # Kamera başlat
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # UI Bileşenleri
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 300)

        self.label_combo = QComboBox(self)
        self.label_combo.addItems(["Kedi", "Köpek"])

        self.save_button = QPushButton("Görüntüyü Kaydet", self)
        self.save_button.clicked.connect(self.save_image)

        self.train_button = QPushButton("Modeli Eğit", self)
        self.train_button.clicked.connect(self.train_model)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.label_combo)
        layout.addWidget(self.save_button)
        layout.addWidget(self.train_button)
        self.setLayout(layout)

        self.setWindowTitle("Kedi-Köpek Model Eğitme Paneli")
        self.setGeometry(100, 100, 500, 500)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def save_image(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            flattened = resized.flatten()

            label = 0 if self.label_combo.currentText() == "Kedi" else 1
            self.X_data.append(flattened)
            self.y_data.append(label)

            print("✅ Görüntü kaydedildi!")

            # Güncellenmiş veriyi kaydet
            with open(self.dataset_path, "wb") as f:
                pickle.dump((self.X_data, self.y_data), f)

    def train_model(self):
        if len(self.X_data) > 1:
            X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)

            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            print("✅ Model başarıyla eğitildi!")
        else:
            print("⚠️ Yeterli veri yok!")

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "rb") as f:
                return pickle.load(f)
        return [], []

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    trainer = CatDogTrainer()
    trainer.show()
    sys.exit(app.exec_())
