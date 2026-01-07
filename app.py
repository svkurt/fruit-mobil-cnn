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
matplotlib.use('Agg') # GUI hatası almamak için
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
print(f"🔧 SİSTEM BAŞLATILIYOR (Hybrid Mode)...")

# Railway CPU olduğu için GPU'yu kapatalım (Hata önleyici)
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
CACHE_PATH = os.path.join(MODELS_DIR, "evaluation_cache.pkl") # Hazır analiz dosyası

# GitHub URL'leri
MODEL_URL = 'https://raw.githubusercontent.com/alifuatkurt55/fruit-cnn/main/models/cnn_fruit_best_model.h5'
CACHE_URL = 'https://raw.githubusercontent.com/alifuatkurt55/fruit-cnn/main/models/evaluation_cache.pkl'

IMG_SIZE = 100
global_model = None
global_class_names = []

# Analiz sonuçlarını tutacak değişken
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
    "status": "Idle",
    "progress": 0,
    "message": "Model eğitimi bekleniyor.",
    "last_updated": None
}

# ============================================================
# 3. YARDIMCI FONKSİYONLAR (İNDİRME VE YÜKLEME)
# ============================================================
def download_file(filepath, url):
    """Dosya yoksa veya boyutu çok küçükse indirir"""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        print(f"📥 İndiriliyor: {filepath} ...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ İndirme tamamlandı.")
            else:
                print(f"❌ İndirme başarısız. Kod: {response.status_code}")
        except Exception as e:
            print(f"❌ Hata: {e}")

def load_resources():
    global global_model, global_class_names, cached_results
    
    # 1. Modeli Hazırla
    download_file(MODEL_PATH, MODEL_URL)
    
    if global_model is None:
        try:
            print("🧠 Model yükleniyor...")
            # compile=False RAM kullanımını azaltır, tahmin için yeterlidir.
            global_model = load_model(MODEL_PATH, compile=False) 
            print("✅ Model Hazır.")
        except Exception as e:
            print(f"⚠️ Model yükleme hatası: {e}")

    # 2. Sınıf İsimlerini Hazırla
    if os.path.exists(CLASSES_PATH):
        try:
            global_class_names = joblib.load(CLASSES_PATH)
        except: pass

    # 3. Hazır Analiz Verilerini (Cache) Hazırla
    download_file(CACHE_PATH, CACHE_URL)
    
    if cached_results["y_true"] is None and os.path.exists(CACHE_PATH):
        try:
            data = joblib.load(CACHE_PATH)
            cached_results.update(data)
            print("📊 Hazır analiz verileri yüklendi.")
            
            # Eğer sınıf isimleri pkl dosyasından gelmediyse buradan al
            if not global_class_names:
                global_class_names = data.get("class_names", [])
        except Exception as e:
            print(f"⚠️ Cache okuma hatası: {e}")

# Uygulama başlarken kaynakları yükle
load_resources()

# ============================================================
# 4. ROUTES
# ============================================================
@app.route('/')
def index():
    return "Meyve AI Sunucusu Aktif"

@app.route("/predict", methods=["POST"])
def predict_single_image():
    # --- BURASI ESKİ KODUNUZLA AYNI MANTIKTA ---
    if global_model is None: 
        load_resources()
        if global_model is None:
            return jsonify({"error": "Model yüklenemedi."}), 500
    
    if 'file' not in request.files: return jsonify({"error": "Dosya yok."}), 400
    
    file = request.files['file']
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
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
        
        # RAM temizliği
        del img, probs
        gc.collect()

        return jsonify({"class": pred_class, "confidence": f"%{confidence * 100:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate")
def evaluate():
    # --- BURASI DEĞİŞTİ: ARTIK HESAPLAMA YAPMIYOR, HAZIR VERİYİ VERİYOR ---
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
             return jsonify({"error": "Hazır test verisi (cache) bulunamadı."}), 500

    return jsonify({
        "accuracy": f"{cached_results['accuracy'] * 100:.2f}%",
        "model_type": "CNN (Offline)",
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
                labels = [class_names[i] for i in unique_indices]
                sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix")
                
            elif plot_type == "top10_wrong":
                cm = confusion_matrix(y_true, y_pred)
                wrong_preds = cm.sum(axis=1) - np.diag(cm)
                top_k = min(10, len(wrong_preds))
                top_idx = np.argsort(wrong_preds)[-top_k:][::-1]
                top_names = [class_names[i] for i in top_idx if i < len(class_names)]
                top_vals = [wrong_preds[i] for i in top_idx]
                ax.bar(top_names, top_vals, color="salmon")
                ax.set_xticklabels(top_names, rotation=45)
                ax.set_title("En Çok Hata Yapılanlar")
                
            elif plot_type == "roc_curve":
                y_test_bin = label_binarize(y_true, classes=range(len(class_names)))
                n_classes = y_test_bin.shape[1]
                for i in range(min(n_classes, 20)):
                    if i < y_probs.shape[1]:
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
            gc.collect()
        
        return send_from_directory(plots_dir, filename)

    except Exception as e:
        return jsonify({"error": f"Grafik hatası: {e}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# Eğitim endpointleri (Gereksiz ama hata vermemesi için boş bırakıldı)
@app.route("/train", methods=["GET", "POST"])
def trigger_training():
    return jsonify({"status": "error", "message": "Railway üzerinde eğitim devre dışı bırakıldı."})

@app.route("/train-status")
def get_training_status():
    return jsonify(training_state)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)