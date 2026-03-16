import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Kaydedilmiş numpy dosyalarından verileri yükler.
    """
    X_train = np.load('output/X_train.npy')
    y_train = np.load('output/y_train.npy')
    X_test = np.load('output/X_test.npy')
    y_test = np.load('output/y_test.npy')
    return X_train, y_train, X_test, y_test

def train_svm(X_train, y_train, kernel='linear', C=100.0, gamma='scale'):
    """
    SVM modelini eğitir.
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        kernel: SVM çekirdeği ('linear', 'poly', 'rbf', 'sigmoid')
        C: Düzenleme parametresi
        gamma: RBF çekirdeği için gamma
    Returns:
        Eğitilmiş SVM modeli
    """
    print(f"SVM modeli eğitiliyor... Kernel: {kernel}, C: {C}, Gamma: {gamma}")
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm_model.fit(X_train, y_train) #Modelin eğitildiği kısım
    print("Eğitim tamamlandı.")
    return svm_model

def evaluate_model(model, X_test, y_test):
    """
    Modeli test verisiyle değerlendirir ve metrikleri hesaplar.
    """
    print("Model değerlendiriliyor...")
    y_pred = model.predict(X_test) #tahmin işlemi yapılır
    
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nBaşarı Metrikleri:")
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall): {recall:.4f}")
    print(f"F1-Skor: {f1:.4f}")
    
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.savefig('output/confusion_matrix.png')
    plt.show()
    
    return accuracy, precision, recall, f1

def main():
    """
    Ana fonksiyon: Verileri yükler, modeli eğitir ve değerlendirir.
    """
    try:
        # Verileri yükle
        print("Veriler yükleniyor...")
        X_train, y_train, X_test, y_test = load_data()
        print(f"Eğitim verisi: {X_train.shape}, Test verisi: {X_test.shape}")
        
        # SVM modelini eğit
        svm_model = train_svm(X_train, y_train, kernel='linear', C=100.0, gamma='scale')
        
        # Modeli değerlendir
        evaluate_model(svm_model, X_test, y_test)
        
        print("\nSVM eğitimi ve değerlendirmesi tamamlandı!")
        print("Confusion matrix 'output/confusion_matrix.png' dosyasına kaydedildi.")
        
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        print("Lütfen önce 'python3 svm_datastore.py' çalıştırarak verileri hazırlayın.")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()