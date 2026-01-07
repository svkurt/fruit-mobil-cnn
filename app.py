from flask import Flask, jsonify, send_from_directory, request, render_template
from flask_cors import CORS
import os
import sys
import numpy as np
import cv2
import joblib
import threading
import time
import gc
import requests 
from datetime import datetime
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, label_binarize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# 1. BAŞLANGIÇ AYARLARI
# ============================================================
print("\n" + "="*50)
print(f"🔧 SİSTEM BAŞLATILIYOR (Fixed Hybrid Mode)...")

try:
    tf.config.set_visible_devices([], 'GPU')
except: pass

app = Flask(__name__)
CORS(app)
plot_lock = threading.Lock()

# ============================================================
# 2. DOSYA YOLLARI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
plots_dir = os.path.join(STATIC_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "cnn_fruit_best_model.h5")
CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")
CACHE_PATH = os.path.join(MODELS_DIR, "evaluation_cache.pkl")

# --- KRİTİK DEĞİŞİKLİK ---
# LFS dosyaları için 'raw' yerine 'media' subdomain'i kullanılır.
# Bu link doğrudan 100MB'lık binary dosyayı verir.
MODEL_URL = 'https://media.githubusercontent.com/media/alifuatkurt55/fruit-cnn/main/models/cnn_fruit_best_model.h5'
CACHE_URL = 'https://raw.githubusercontent.com/alifuatkurt55/fruit-cnn/main/models/evaluation_cache.pkl'

IMG_SIZE = 100
global_model = None
global_class_names = []

cached_results = {
    "y_true": None,
    "y_pred": None,
    "y_probs": None,
    "class_names": [],
    "accuracy": 0,
    "report": {}
}

training_state = {
    "is_training": False,
    "status": "Disabled",
    "progress": 0,
    "message": "Eğitim devre dışı.",
    "last_updated": None
}

# ============================================================
# 3. YARDIMCI FONKSİYONLAR
# ============================================================
def download_file(filepath, url, description):
    """Dosyayı indirir ve boyut kontrolü yapar"""
    # Dosya var mı?
    if os.path.exists(filepath):
        # Eğer model dosyası 5MB'dan küçükse kesin yanlıştır (LFS pointer'dır), sil.
        if "model.h5" in filepath and os.path.getsize(filepath) < 5 * 1024 * 1024:
            print(f"⚠️ {description} boyutu çok küçük (Hatalı LFS dosyası). Siliniyor...")
            os.remove(filepath)
        else:
            # Dosya sağlam görünüyor
            return

    print(f"📥 İndiriliyor: {filepath} ...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ {description} indirildi. Boyut: {os.path.getsize(filepath) // 1024} KB")
        else:
            print(f"❌ İndirme başarısız ({description}). Kod: {response.status_code}")
    except Exception as e:
        print(f"❌ İndirme hatası ({description}): {e}")

def load_resources():
    global global_model, global_class_names, cached_results
    
    # 1. Modeli İndir ve Yükle
    download_file(MODEL_PATH, MODEL_URL, "Model Dosyası")
    
    if global_model is None:
        if os.path.exists(MODEL_PATH):
            try:
                print("🧠 Model hafızaya yükleniyor...")
                global_model = load_model(MODEL_PATH, compile=False) 
                print("✅ Model Hazır.")
            except Exception as e:
                print(f"🔥 Model bozuk veya okunamadı: {e}")
                # Bozuk dosyayı sil ki sonraki sefer tekrar indirsin
                try: os.remove(MODEL_PATH)
                except: pass
        else:
            print("❌ Model dosyası bulunamadı.")

    # 2. Sınıf İsimlerini Yükle
    if os.path.exists(CLASSES_PATH):
        try:
            global_class_names = joblib.load(CLASSES_PATH)
        except: pass

    # 3. Hazır Analiz Verilerini Yükle
    download_file(CACHE_PATH, CACHE_URL, "Cache Dosyası")
    
    # Cache yükleme mantığı (Hata düzeltildi)
    if cached_results["y_true"] is None and os.path.exists(CACHE_PATH):
        try:
            data = joblib.load(CACHE_PATH)
            cached_results.update(data)
            print("📊 Hazır analiz verileri yüklendi.")
            
            # NumPy array hatasını önlemek için len() kontrolü
            if len(global_class_names) == 0:
                global_class_names = data.get("class_names", [])
        except Exception as e:
            print(f"⚠️ Cache okuma hatası: {e}")

# Başlangıçta yükle
load_resources()

# ============================================================
# 4. ROUTES
# ============================================================
@app.route('/')
def index():
    status = "Aktif" if global_model else "Model Yüklenemedi"
    return f"Meyve AI Sunucusu: {status}"

@app.route("/predict", methods=["POST"])
def predict_single_image():
    if global_model is None: 
        load_resources()
        if global_model is None:
            return jsonify({"error": "Sunucu Hatası: Model yüklenemedi."}), 500
    
    if 'file' not in request.files: return jsonify({"error": "Dosya yok."}), 400
    
    file = request.files['file']
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None: return jsonify({"error": "Resim okunamadı."}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        probs = global_model.predict(img, verbose=0)
        pred_idx = np.argmax(probs)
        confidence = float(np.max(probs))
        
        if len(global_class_names) > 0:
            pred_class = global_class_names[pred_idx]
        else:
            pred_class = f"Class {pred_idx}"
        
        del img, probs
        gc.collect()

        return jsonify({"class": pred_class, "confidence": f"%{confidence * 100:.2f}"})
    except Exception as e:
        print(f"Predict Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate")
def evaluate():
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
             return jsonify({"error": "Hazır test verisi bulunamadı."}), 500

    return jsonify({
        "accuracy": f"{cached_results['accuracy'] * 100:.2f}%",
        "model_type": "CNN (Offline Cache)",
        "class_report": cached_results['report']
    })

@app.route("/get-plot/<plot_type>")
def get_plot(plot_type):
    # Hazır veriyi kullanarak grafik çiz (Hızlı)
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
            return jsonify({"error": "Veri yok."}), 400

    filename = f"{plot_type}.png"
    save_path = os.path.join(plots_dir, filename)
    
    y_true = cached_results["y_true"]
    y_pred = cached_results["y_pred"]
    y_probs = cached_results["y_probs"]
    class_names = cached_results["class_names"]

    try:
        with plot_lock:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if plot_type == "confusion_matrix":
                cm = confusion_matrix(y_true, y_pred)
                unique_indices = sorted(list(set(y_true) | set(y_pred)))
                # İndekslerin sınıf isimlerine karşılık geldiğinden emin ol
                labels = []
                for i in unique_indices:
                    if i < len(class_names):
                        labels.append(class_names[i])
                    else:
                        labels.append(f"Class {i}")
                        
                sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
            elif plot_type == "top10_wrong":
                cm = confusion_matrix(y_true, y_pred)
                wrong_preds = cm.sum(axis=1) - np.diag(cm)
                top_k = min(10, len(wrong_preds))
                if top_k > 0:
                    top_idx = np.argsort(wrong_preds)[-top_k:][::-1]
                    
                    top_names = []
                    top_vals = []
                    for i in top_idx:
                        if i < len(class_names):
                            top_names.append(class_names[i])
                        else:
                            top_names.append(f"Class {i}")
                        top_vals.append(wrong_preds[i])

                    ax.bar(top_names, top_vals, color="salmon")
                    ax.set_xticklabels(top_names, rotation=45, ha='right')
                    ax.set_title("En Çok Hata Yapılanlar")
                else:
                    ax.text(0.5, 0.5, "Hata Yok! (Mükemmel Sonuç)", ha='center')
                
            elif plot_type == "roc_curve":
                # --- GÜNCELLENMİŞ ROC KODU ---
                if y_probs is None:
                    ax.text(0.5, 0.5, "Olasılık verisi (y_probs) bulunamadı", ha='center')
                else:
                    # Modelin çıktı boyutunu (sınıf sayısını) al
                    n_classes = y_probs.shape[1]
                    
                    # Gerçek değerleri binarize et (One-Hot Encoding)
                    # classes parametresi 0'dan n_classes'a kadar olmalı
                    y_test_bin = label_binarize(y_true, classes=range(n_classes))
                    
                    # Eğer test setinde sadece 2 sınıf varsa label_binarize bazen tek sütun döner
                    if n_classes == 2 and y_test_bin.shape[1] == 1:
                        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

                    lines_drawn = 0
                    # En çok 20 sınıfı çiz (Grafik okunabilir olsun diye)
                    for i in range(min(n_classes, 20)):
                        # Sadece test setinde örneği bulunan sınıflar için çizim yap
                        # (Hiç örneği olmayan bir sınıfın AUC'si hesaplanamaz)
                        if np.sum(y_test_bin[:, i]) > 0:
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                            roc_auc = auc(fpr, tpr)
                            
                            # İsimlendirme güvenliği
                            label = class_names[i] if i < len(class_names) else f"Class {i}"
                            ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.2f})')
                            lines_drawn += 1
                    
                    if lines_drawn > 0:
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.legend(loc="lower right", fontsize='small')
                        ax.set_title("ROC Curve")
                    else:
                        ax.text(0.5, 0.5, "Yeterli veri çeşitliliği yok\n(Tek tip sınıf var)", ha='center')

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
            gc.collect()
        
        return send_from_directory(plots_dir, filename)

    except Exception as e:
        print(f"Grafik hatası ({plot_type}): {e}")
        return jsonify({"error": f"Grafik oluşturulamadı: {e}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# Eğitim endpointleri (Boş)
@app.route("/train", methods=["GET", "POST"])
def trigger_training():
    return jsonify({"status": "error", "message": "Devre dışı."})

@app.route("/train-status")
def get_training_status():
    return jsonify(training_state)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)