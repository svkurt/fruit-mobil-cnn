from flask import Flask, jsonify, send_from_directory, request, render_template
from flask_cors import CORS
import os
import numpy as np
import cv2  # Görüntü işleme kütüphanesi (OpenCV)
import joblib  # .pkl dosyalarını (küçük veri yapıları) kaydetmek/okumak için
import threading  # Aynı anda birden fazla işlem yapabilmek için (Async)
import gc  # Garbage Collector: Bellek temizliği için
import requests  # İnternetten dosya indirmek için
from datetime import datetime
import tensorflow as tf

# --- MATPLOTLIB AYARI (KRİTİK) ---
# Sunucularda (Railway, Heroku vb.) ekran kartı arayüzü (GUI) yoktur.
# 'Agg' backend'i, grafikleri ekrana basmak yerine dosyaya yazmaya yarar.
# Bu satır olmazsa sunucu "TclError: no display name" hatası verip çöker.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Makine Öğrenmesi Metrikleri
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential, load_model

# ============================================================
# 1. BAŞLANGIÇ AYARLARI VE DONANIM KONFİGÜRASYONU
# ============================================================
print("\n" + "="*50)
print(f"🔧 SİSTEM BAŞLATILIYOR (Final Full Fix)...")

# --- GPU DEVRE DIŞI BIRAKMA (CLOUD İÇİN) ---
# Railway gibi bulut platformlarının ücretsiz/başlangıç paketlerinde GPU yoktur.
# TensorFlow GPU aramaya çalışıp bulamazsa bazen hata verir veya RAM'i şişirir.
# Bu kod ile "Sadece CPU kullan" emrini veriyoruz.
try:
    tf.config.set_visible_devices([], 'GPU')
except: pass

app = Flask(__name__)
# CORS (Cross-Origin Resource Sharing): Farklı bir kaynaktan (örneğin mobil uygulama
# veya farklı bir web sitesi) gelen isteklerin engellenmemesini sağlar.
CORS(app)

# --- THREAD KİLİDİ (PLOT LOCK) ---
# Matplotlib "thread-safe" değildir. Yani aynı anda iki kullanıcı grafik isterse
# sunucu karışır ve çöker. Bu kilit, işlemleri sıraya koyar.
plot_lock = threading.Lock()

# ============================================================
# 2. DOSYA YOLLARI (PATH CONFIGURATION)
# ============================================================
# Projenin çalıştığı ana dizini buluyoruz.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Klasör yapılarını tanımlıyoruz:
STATIC_DIR = os.path.join(BASE_DIR, "static")  # Resim, CSS vb. statik dosyalar
MODELS_DIR = os.path.join(BASE_DIR, "models")  # .h5 ve .pkl dosyaları
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")  # Üretilen grafiklerin kaydedileceği yer

# Klasörler yoksa oluşturuyoruz (Hata almamak için)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dosya yolları
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_fruit_best_model.h5")
CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")
CACHE_PATH = os.path.join(MODELS_DIR, "evaluation_cache.pkl")
TRAINING_PLOT_PATH = os.path.join(PLOTS_DIR, "training_curve.png")

# --- GITHUB OTOMATİK İNDİRME LİNKLERİ ---
# Sunucu her yeniden başladığında dosyaları GitHub'dan çeker.
# LFS (Large File Storage) dosyaları için 'media.githubusercontent.com' kullanılır.
MODEL_URL = 'https://media.githubusercontent.com/media/svkurt/fruit-mobil-cnn/main/models/cnn_fruit_best_model.h5'
# Küçük dosyalar için 'raw.githubusercontent.com' kullanılır.
CACHE_URL = 'https://raw.githubusercontent.com/svkurt/fruit-mobil-cnn/main/models/evaluation_cache.pkl'
TRAINING_PLOT_URL = 'https://raw.githubusercontent.com/svkurt55/fruit-mobil-cnn/main/static/plots/training_curve.png'

# Modelin eğitildiği resim boyutu (Değiştirilmemeli, model buna göre eğitildi)
IMG_SIZE = 100

# Global değişkenler (RAM'de tutulacak veriler)
global_model = None
global_class_names = []

# Analiz sonuçlarını tutan önbellek (Cache)
# Her defasında hesaplama yapmamak için sonuçları burada saklıyoruz.
cached_results = {
    "y_true": None,
    "y_pred": None,
    "y_probs": None,
    "class_names": [],
    "accuracy": 0,
    "report": {}
}

# Eğitim durumu (Şu an pasif, ileride kullanılabilir)
training_state = {
    "is_training": False,
    "status": "Idle",
    "progress": 0,
    "message": "Hazır.",
    "last_updated": None
}

# ============================================================
# 3. YARDIMCI FONKSİYONLAR (UTILS)
# ============================================================
def download_file(filepath, url, description):
    """
    Verilen URL'den dosyayı indirir ve bozuk olup olmadığını kontrol eder.
    Özellikle GitHub LFS (Large File Storage) hatalarını yakalamak için kritiktir.
    """
    if os.path.exists(filepath):
        # KONTROL 1: Model dosyası 5MB'dan küçükse, yanlış inmiştir (LFS Pointer hatası).
        # Bu durumda dosyayı silip tekrar indirmemiz gerekir.
        if "model.h5" in filepath and os.path.getsize(filepath) < 5 * 1024 * 1024:
            print(f"⚠️ {description} hatalı (Boyut çok küçük), siliniyor...")
            os.remove(filepath)
        # KONTROL 2: Resim dosyası 0 byte ise (boşsa) sil.
        elif "png" in filepath and os.path.getsize(filepath) == 0:
            os.remove(filepath)
        else:
            return # Dosya sağlam, indirmeye gerek yok.

    print(f"📥 İndiriliyor: {filepath} ...")
    try:
        # stream=True: Dosyayı parça parça indirir (RAM'i şişirmemek için)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ İndirme tamamlandı: {description}")
        else:
            print(f"❌ İndirme başarısız ({description}). Kod: {response.status_code}")
    except Exception as e:
        print(f"❌ Hata ({description}): {e}")

def load_resources():
    """
    Sistemi ayağa kaldırırken veya ihtiyaç anında gerekli dosyaları (Model, Cache)
    indirip belleğe yükleyen 'Lazy Loading' fonksiyonu.
    """
    global global_model, global_class_names, cached_results
    
    # 1. Modeli İndir ve Yükle
    download_file(MODEL_PATH, MODEL_URL, "Model")
    if global_model is None and os.path.exists(MODEL_PATH):
        try:
            print("🧠 Model yükleniyor...")
            # compile=False: Modeli sadece tahmin (predict) için kullanacağız, eğitim yapmayacağız.
            # Bu sayede optimizer yüklenmez ve bellek tasarrufu sağlanır.
            global_model = load_model(MODEL_PATH, compile=False) 
            print("✅ Model Hazır.")
        except Exception as e:
            print(f"⚠️ Model hatası: {e}")
            try: os.remove(MODEL_PATH) # Bozuksa sil
            except: pass

    # 2. Sınıf İsimlerini Yükle (.pkl dosyasından)
    if os.path.exists(CLASSES_PATH):
        try: global_class_names = joblib.load(CLASSES_PATH)
        except: pass

    # 3. Analiz Verilerini (Cache) İndir ve Yükle
    download_file(CACHE_PATH, CACHE_URL, "Cache")
    if cached_results["y_true"] is None and os.path.exists(CACHE_PATH):
        try:
            data = joblib.load(CACHE_PATH)
            cached_results.update(data) # Cache sözlüğünü güncelle
            print("📊 Analiz verileri yüklendi.")
            if not global_class_names:
                global_class_names = data.get("class_names", [])
        except Exception as e:
            print(f"⚠️ Cache hatası: {e}")

    # 4. Eğitim Grafiğini İndir (Sunucuda çizilemediği için hazır indiriyoruz)
    download_file(TRAINING_PLOT_PATH, TRAINING_PLOT_URL, "Eğitim Grafiği")

# Uygulama başlarken kaynakları yüklemeyi dene
load_resources()

# ============================================================
# 4. ROUTES (API UÇ NOKTALARI)
# ============================================================
@app.route('/')
def index():
    """Ana sayfa. Sunucunun ayakta olup olmadığını kontrol etmek için."""
    return "Meyve AI Backend Aktif"

@app.route("/predict", methods=["POST"])
def predict_single_image():
    """
    MOBİL UYGULAMADAN GELEN FOTOĞRAFI TAHMİN EDEN FONKSİYON.
    """
    # Model yüklü değilse yüklemeyi dene
    if global_model is None: 
        load_resources()
        if global_model is None:
            return jsonify({"error": "Model yüklenemedi."}), 500
    
    # Dosya kontrolü
    if 'file' not in request.files: return jsonify({"error": "Dosya yok."}), 400
    
    file = request.files['file']
    try:
        # Resmi RAM üzerinden oku (Diske kaydetmeden işleme - Hız kazandırır)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- ÖN İŞLEME (PREPROCESSING) ---
        # 1. Renk dönüşümü: OpenCV BGR okur, Model RGB ister.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 2. Boyutlandırma: Model 100x100 ile eğitildiği için.
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # 3. Normalizasyon: Piksel değerlerini 0-255 arasından 0-1 arasına çek.
        img = img.astype("float32") / 255.0
        # 4. Batch boyutu ekleme: (100, 100, 3) -> (1, 100, 100, 3)
        img = np.expand_dims(img, axis=0)
        
        # --- TAHMİN ---
        probs = global_model.predict(img, verbose=0)
        pred_idx = np.argmax(probs) # En yüksek olasılıklı indeks
        confidence = float(np.max(probs)) # Güven oranı (Örn: 0.95)
        
        # İndeksi sınıf ismine çevir (0 -> 'Elma')
        if len(global_class_names) > 0:
            pred_class = global_class_names[pred_idx]
        else:
            pred_class = f"Class {pred_idx}"
        
        # Bellek Temizliği (Memory Leak önlemek için)
        del img, probs
        gc.collect()

        return jsonify({"class": pred_class, "confidence": f"%{confidence * 100:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate")
def evaluate():
    """
    'Test Et' butonuna basıldığında çağrılır.
    Hesaplama yapmaz, 'load_resources' ile indirilen hazır cache verisini döner.
    """
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
             return jsonify({"error": "Analiz verisi yok."}), 500

    return jsonify({
        "accuracy": f"{cached_results['accuracy'] * 100:.2f}%",
        "model_type": "CNN (Offline)",
        "class_report": cached_results['report']
    })

@app.route("/get-plot/<plot_type>")
def get_plot(plot_type):
    """
    İstenilen grafiği (Confusion Matrix, ROC vb.) o an oluşturup resim olarak döner.
    """
    filename = f"{plot_type}.png"
    save_path = os.path.join(PLOTS_DIR, filename)

    # --- EĞİTİM GRAFİĞİ İÇİN ÖZEL KONTROL ---
    # Bu grafik hesaplanarak çizilemez (geçmiş veridir), indirilmesi gerekir.
    if plot_type == "training_curve":
        if not os.path.exists(save_path):
            download_file(save_path, TRAINING_PLOT_URL, "Eğitim Grafiği")
        
        if os.path.exists(save_path):
            return send_from_directory(PLOTS_DIR, filename)
        else:
            return jsonify({"error": "Grafik GitHub'da bulunamadı."}), 404
    # ------------------------------------------

    # Diğer grafikler için veri kontrolü
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
            return jsonify({"error": "Veri yok."}), 400
    
    # Verileri Numpy formatına çevir (Hata almamak için)
    y_true = np.array(cached_results["y_true"])
    y_pred = np.array(cached_results["y_pred"])
    y_probs = np.array(cached_results["y_probs"])
    class_names = cached_results["class_names"]

    try:
        # THREAD KİLİDİ: Aynı anda tek çizim yapılsın
        with plot_lock:
            plt.close('all') # Eski çizimleri temizle
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 1. Confusion Matrix (Karmaşıklık Matrisi)
            if plot_type == "confusion_matrix":
                cm = confusion_matrix(y_true, y_pred)
                unique_indices = sorted(list(set(y_true) | set(y_pred)))
                labels = [class_names[i] if i < len(class_names) else f"{i}" for i in unique_indices]
                sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
            # 2. Hatalı Tahminler Grafiği
            elif plot_type == "top10_wrong":
                cm = confusion_matrix(y_true, y_pred)
                # Köşegen (Doğru bilenler) dışındaki toplam hataları hesapla
                wrong_preds = cm.sum(axis=1) - np.diag(cm)
                top_k = min(10, len(wrong_preds))
                if top_k > 0:
                    top_idx = np.argsort(wrong_preds)[-top_k:][::-1] # En çok hata yapılanları sırala
                    top_names = [class_names[i] if i < len(class_names) else f"{i}" for i in top_idx]
                    top_vals = [wrong_preds[i] for i in top_idx]
                    ax.bar(top_names, top_vals, color="salmon")
                    ax.set_xticklabels(top_names, rotation=45, ha='right')
                    ax.set_title("Hatalı Tahminler")
                else:
                    ax.text(0.5, 0.5, "Hata Yok", ha='center')
                
            # 3. ROC Eğrisi (Performans Analizi)
            elif plot_type == "roc_curve":
                if y_probs is None:
                    ax.text(0.5, 0.5, "Olasılık verisi yok", ha='center')
                else:
                    n_classes = y_probs.shape[1] 
                    # Sınıfları One-Hot formatına çevir (Binary'ye dönüştür)
                    y_test_bin = label_binarize(y_true, classes=range(n_classes))
                    if n_classes == 2 and y_test_bin.shape[1] == 1:
                        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

                    lines_drawn = 0
                    present_classes = np.unique(y_true)
                    
                    # Sadece test setinde var olan sınıfları çiz
                    for i in present_classes:
                        if i < n_classes:
                            try:
                                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                                roc_auc = auc(fpr, tpr)
                                label_name = class_names[i] if i < len(class_names) else f"Class {i}"
                                ax.plot(fpr, tpr, lw=2, label=f'{label_name} ({roc_auc:.2f})')
                                lines_drawn += 1
                            except: pass

                    if lines_drawn > 0:
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.legend(loc="lower right", fontsize='small')
                        ax.set_title("ROC Curve")
                    else:
                        ax.text(0.5, 0.5, "Grafik Çizilemedi", ha='center')

            plt.tight_layout()
            fig.savefig(save_path) # Resmi kaydet
            plt.close(fig) # Bellekten temizle
            gc.collect()
        
        return send_from_directory(PLOTS_DIR, filename)

    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return jsonify({"error": f"{e}"}), 500

# Statik dosyaları (resimler vb.) sunmak için gerekli rota
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# Eğitim rotası (Şu an devre dışı bırakılmış)
@app.route("/train", methods=["GET", "POST"])
def trigger_training():
    return jsonify({"status": "error", "message": "Devre dışı."})

# Eğitim durumu sorgulama
@app.route("/train-status")
def get_training_status():
    return jsonify(training_state)

if __name__ == "__main__":
    # Railway'in atadığı PORT'u al, yoksa 5000 kullan
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
