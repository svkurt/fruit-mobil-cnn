from flask import Flask, jsonify, send_from_directory, request, render_template
from flask_cors import CORS
import os
import numpy as np
import cv2
import joblib
import threading
import gc
import requests 
from datetime import datetime
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential, load_model

# ============================================================
# 1. BAŞLANGIÇ AYARLARI
# ============================================================
print("\n" + "="*50)
print(f"🔧 SİSTEM BAŞLATILIYOR (Final ROC Fix)...")

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

# LFS için 'media' subdomain kullanımı
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
    "status": "Idle",
    "progress": 0,
    "message": "Hazır.",
    "last_updated": None
}

# ============================================================
# 3. YARDIMCI FONKSİYONLAR
# ============================================================
def download_file(filepath, url, description):
    if os.path.exists(filepath):
        # Model dosyası çok küçükse sil (LFS hatası önlemi)
        if "model.h5" in filepath and os.path.getsize(filepath) < 5 * 1024 * 1024:
            print(f"⚠️ {description} boyutu hatalı, siliniyor...")
            os.remove(filepath)
        else:
            return

    print(f"📥 İndiriliyor: {filepath} ...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ İndirme tamamlandı.")
        else:
            print(f"❌ İndirme başarısız: {response.status_code}")
    except Exception as e:
        print(f"❌ Hata: {e}")

def load_resources():
    global global_model, global_class_names, cached_results
    
    # Modeli İndir
    download_file(MODEL_PATH, MODEL_URL, "Model")
    
    # Modeli Yükle
    if global_model is None and os.path.exists(MODEL_PATH):
        try:
            print("🧠 Model yükleniyor...")
            global_model = load_model(MODEL_PATH, compile=False) 
            print("✅ Model Hazır.")
        except Exception as e:
            print(f"⚠️ Model hatası: {e}")
            try: os.remove(MODEL_PATH) 
            except: pass

    # Sınıf İsimleri
    if os.path.exists(CLASSES_PATH):
        try: global_class_names = joblib.load(CLASSES_PATH)
        except: pass

    # Analiz Verileri (Cache)
    download_file(CACHE_PATH, CACHE_URL, "Cache")
    
    if cached_results["y_true"] is None and os.path.exists(CACHE_PATH):
        try:
            data = joblib.load(CACHE_PATH)
            cached_results.update(data)
            print("📊 Analiz verileri yüklendi.")
            if not global_class_names:
                global_class_names = data.get("class_names", [])
        except Exception as e:
            print(f"⚠️ Cache hatası: {e}")

load_resources()

# ============================================================
# 4. ROUTES
# ============================================================
@app.route('/')
def index():
    return "Meyve AI Backend"

@app.route("/predict", methods=["POST"])
def predict_single_image():
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
        
        del img, probs
        gc.collect()

        return jsonify({"class": pred_class, "confidence": f"%{confidence * 100:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate")
def evaluate():
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
             return jsonify({"error": "Analiz verisi yok."}), 500

    return jsonify({
        "accuracy": f"{cached_results['accuracy'] * 100:.2f}%",
        "model_type": "CNN (Cache)",
        "class_report": cached_results['report']
    })

@app.route("/get-plot/<plot_type>")
def get_plot(plot_type):
    # Veri kontrolü
    if cached_results["y_true"] is None:
        load_resources()
        if cached_results["y_true"] is None:
            return jsonify({"error": "Veri yok."}), 400

    filename = f"{plot_type}.png"
    save_path = os.path.join(plots_dir, filename)
    
    y_true = np.array(cached_results["y_true"])
    y_pred = np.array(cached_results["y_pred"])
    y_probs = np.array(cached_results["y_probs"])
    class_names = cached_results["class_names"]

    try:
        with plot_lock:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if plot_type == "confusion_matrix":
                cm = confusion_matrix(y_true, y_pred)
                unique_indices = sorted(list(set(y_true) | set(y_pred)))
                labels = [class_names[i] if i < len(class_names) else f"{i}" for i in unique_indices]
                sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
            elif plot_type == "top10_wrong":
                cm = confusion_matrix(y_true, y_pred)
                wrong_preds = cm.sum(axis=1) - np.diag(cm)
                top_k = min(10, len(wrong_preds))
                if top_k > 0:
                    top_idx = np.argsort(wrong_preds)[-top_k:][::-1]
                    top_names = [class_names[i] if i < len(class_names) else f"{i}" for i in top_idx]
                    top_vals = [wrong_preds[i] for i in top_idx]
                    ax.bar(top_names, top_vals, color="salmon")
                    ax.set_xticklabels(top_names, rotation=45, ha='right')
                    ax.set_title("Hatalı Tahminler")
                else:
                    ax.text(0.5, 0.5, "Hata Yok", ha='center')
                
            elif plot_type == "roc_curve":
                # --- ESKİ KODUN MANTIĞININ DÜZELTİLMİŞ HALİ ---
                if y_probs is None:
                    ax.text(0.5, 0.5, "Olasılık verisi yok", ha='center')
                else:
                    # Model kaç sınıf çıktısı veriyor?
                    n_classes = y_probs.shape[1] 
                    
                    # Gerçek değerleri (y_true) modelin sınıf sayısına göre binary yap
                    # Bu sayede y_test_bin ile y_probs aynı boyutta (Sütun sayısı) olur.
                    y_test_bin = label_binarize(y_true, classes=range(n_classes))
                    
                    # Eğer sadece 2 sınıf varsa label_binarize tek sütun döner, onu düzeltelim
                    if n_classes == 2 and y_test_bin.shape[1] == 1:
                        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

                    lines_drawn = 0
                    
                    # Sadece test verisinde mevcut olan (Unique) sınıfları çiz
                    present_classes = np.unique(y_true)
                    
                    for i in present_classes:
                        # Eğer indeks geçerliyse
                        if i < n_classes:
                            try:
                                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                                roc_auc = auc(fpr, tpr)
                                
                                label_name = class_names[i] if i < len(class_names) else f"Class {i}"
                                ax.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC={roc_auc:.2f})')
                                lines_drawn += 1
                            except: pass # Hesaplama hatası olursa geç

                    if lines_drawn > 0:
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.legend(loc="lower right", fontsize='small')
                        ax.set_title("ROC Curve")
                    else:
                        ax.text(0.5, 0.5, "Grafik Çizilemedi", ha='center')
                # -----------------------------------

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
            gc.collect()
        
        return send_from_directory(plots_dir, filename)

    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return jsonify({"error": f"{e}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/train", methods=["GET", "POST"])
def trigger_training():
    return jsonify({"status": "error", "message": "Devre dışı."})

@app.route("/train-status")
def get_training_status():
    return jsonify(training_state)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)