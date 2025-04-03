import os
import cv2

dataset_path = "dataset"  # Veri kümenizin ana dizini

def check_and_remove_corrupt_images(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        # Eğer dosya bir resim değilse (örneğin Thumbs.db), atla
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Siliniyor: {file_path} (Geçersiz dosya türü)")
            os.remove(file_path)
            continue

        # Resmi yükle ve kontrol et
        image = cv2.imread(file_path)
        if image is None:
            print(f"Siliniyor: {file_path} (Bozuk resim)")
            os.remove(file_path)

# Kedi ve köpek klasörlerini kontrol et
check_and_remove_corrupt_images(os.path.join(dataset_path, "Cat"))
check_and_remove_corrupt_images(os.path.join(dataset_path, "Dog"))

print("✅ Temizleme işlemi tamamlandı!")
