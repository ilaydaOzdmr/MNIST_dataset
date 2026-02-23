import os
import numpy as np
from PIL import Image

class SVMDataStore:
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir (string): 'mnist' klasörünün yolu.
            split (string): 'train' veya 'test' seçimi.
        """
        self.root_dir = root_dir
        self.split = split
        self.data_path = os.path.join(root_dir, split)
        
        # Verileri yükle
        self.images, self.labels = self._load_data()

    def _load_data(self):
        images = []
        labels = []
        
        # Eğer klasör yoksa hata ver
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Klasör bulunamadı: {self.data_path}")
        
        # Mevcut dosya ve klasörleri listele
        contents = os.listdir(self.data_path)
        
        # Dosya adlarından (örn. 0_000222.png) class subdirectory var mı kontrol et
        has_class_dirs = any(os.path.isdir(os.path.join(self.data_path, item)) for item in contents)
        
        if has_class_dirs:
            # Eski format: class subdirectories (örn. mnist/train/0/, mnist/train/1/)
            classes = sorted(os.listdir(self.data_path))
            
            for class_name in classes:
                class_path = os.path.join(self.data_path, class_name)
                
                # Sadece klasörleri işleme al (dosyaları atla)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        # Dosya yolunu oluştur
                        img_path = os.path.join(class_path, img_name)
                        
                        # Sadece resim dosyalarını ekle
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            # Resmi aç ve numpy array'e çevir
                            image = Image.open(img_path).convert('L')
                            image_array = np.array(image).flatten() / 255.0  # Normalize to [0,1]
                            
                            images.append(image_array)
                            labels.append(int(class_name))  # Klasör adını integer etikete çevir
        else:
            # Yeni format: dosyalar doğrudan train klasöründe (örn. 0_000222.png, 1_000104.png)
            for img_name in contents:
                img_path = os.path.join(self.data_path, img_name)
                
                # Sadece resim dosyalarını işleme al
                if os.path.isfile(img_path) and img_name.endswith(('.png', '.jpg', '.jpeg')):
                    # Resmi aç ve numpy array'e çevir
                    image = Image.open(img_path).convert('L')
                    image_array = np.array(image).flatten() / 255.0  # Normalize to [0,1]
                    
                    # Etiketi çek
                    label = int(img_name.split('_')[0])
                    
                    images.append(image_array)
                    labels.append(label)
        
        return np.array(images), np.array(labels)

    def get_data(self):
        return self.images, self.labels

    def save_data(self, output_dir='output'):
        """
        Verileri numpy dosyaları olarak kaydet.
        Args:
            output_dir (string): Kaydetme klasörü.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, f'X_{self.split}.npy'), self.images)
        np.save(os.path.join(output_dir, f'y_{self.split}.npy'), self.labels)
        
        print(f"Veriler {output_dir} klasörüne kaydedildi.")

# --- TEST KISMI ---
if __name__ == "__main__":
    # Senin klasör yapına göre yol
    dataset_path = "mnist" 
    
    try:
        # SVM Datastore'u oluştur (Train için)
        svm_train = SVMDataStore(root_dir=dataset_path, split='train')
        X_train, y_train = svm_train.get_data()
        
        print(f"SVM Train veri şekli: {X_train.shape}")
        print(f"SVM Train etiket şekli: {y_train.shape}")
        print(f"Örnek etiketler: {y_train[:10]}")
        
        # Verileri kaydet
        svm_train.save_data()
        
        # Test için de yükle
        svm_test = SVMDataStore(root_dir=dataset_path, split='test')
        X_test, y_test = svm_test.get_data()
        
        print(f"SVM Test veri şekli: {X_test.shape}")
        print(f"SVM Test etiket şekli: {y_test.shape}")
        
        # Test verilerini de kaydet
        svm_test.save_data()
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
    