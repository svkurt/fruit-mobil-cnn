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
from datetime import datetime

# ============================================================
# 1. SİSTEM VE GPU YAPILANDIRMASI
# ============================================================
current_python_dir = os.path.dirname(sys.executable)
conda_env_path = current_python_dir
conda_lib_path = os.path.join(conda_env_path, "Library", "bin")

print("\n" + "="*50)
print(f"🔧 SİSTEM BAŞLATILIYOR...")

os.environ['PATH'] = conda_lib_path + os.pathsep + os.environ['PATH']
os.environ['PATH'] = os.path.join(conda_env_path, "Library", "bin") + os.pathsep + os.environ['PATH']

if hasattr(os, 'add_dll_directory') and os.path.exists(conda_lib_path):
    try:
        os.add_dll_directory(conda_lib_path)
    except Exception as e:
        pass

import tensorflow as tf

print("DONANIM KONTROLÜ...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU BULUNDU: {len(gpus)} adet aktif.")
    except RuntimeError as e:
        print(f"❌ GPU Ayar Hatası: {e}")
else:
    print("⚠️ GPU BULUNAMADI! CPU Modu.")
print("="*50 + "\n")

import matplotlib
matplotlib.use('Agg') # GUI hatası almamak için
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, label_binarize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

app = Flask(__name__)
CORS(app)

plot_lock = threading.Lock()

# ============================================================
# 2. DOSYA YOLLARI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "fruits-360_100x100_mini/train")
TEST_DIR = os.path.join(BASE_DIR, "fruits-360_100x100_mini/test")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_fruit_best_model.h5")
CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 25

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
plots_dir = os.path.join(STATIC_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ============================================================
# 3. GLOBAL DEĞİŞKENLER
# ============================================================
training_state = {
    "is_training": False,
    "status": "Idle",
    "progress": 0,
    "message": "Model eğitimi bekleniyor.",
    "last_updated": None
}

global_model = None
global_class_names = []

# --- CACHE (ÖNBELLEK) ---
cached_results = {
    "y_true": None,
    "y_pred": None,
    "y_probs": None,
    "class_names": []
}

def load_global_model():
    global global_model, global_class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        try:
            print("Model yükleniyor...")
            global_model = load_model(MODEL_PATH)
            global_class_names = joblib.load(CLASSES_PATH)
            print("Model hazır.")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")

load_global_model()

# ============================================================
# 4. EĞİTİM & CALLBACK
# ============================================================
class FlaskStatusCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        training_state["status"] = "Training"
        training_state["progress"] = epoch + 1
        training_state["message"] = f"Epoch {epoch + 1}/{EPOCHS} Bitti. Başarı: %{acc*100:.1f} (Val: %{val_acc*100:.1f})"
        training_state["last_updated"] = datetime.now().strftime("%H:%M:%S")
        print(f"Training: {training_state['message']}")

def plot_training_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    save_path = os.path.join(plots_dir, "training_curve.png")

    with plot_lock:
        # OO Style (Daha güvenli)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(epochs_range, acc, label='Train Accuracy')
        ax1.plot(epochs_range, val_acc, label='Val Accuracy')
        ax1.legend(loc='lower right')
        ax1.set_title('Accuracy')
        ax1.grid(True)

        ax2.plot(epochs_range, loss, label='Train Loss')
        ax2.plot(epochs_range, val_loss, label='Val Loss')
        ax2.legend(loc='upper right')
        ax2.set_title('Loss')
        ax2.grid(True)

        fig.savefig(save_path)
        plt.close(fig) # Figürü bellekten sil

def train_model_background():
    global global_model, global_class_names
    try:
        training_state["is_training"] = True
        training_state["status"] = "Loading Data"
        training_state["message"] = "Veriler hazırlanıyor..."
        training_state["progress"] = 0
        
        X, y = [], []
        if not os.path.exists(TRAIN_DIR): raise FileNotFoundError(f"Klasör yok: {TRAIN_DIR}")
        classes = sorted(os.listdir(TRAIN_DIR))
        
        for cls in classes:
            path = os.path.join(TRAIN_DIR, cls)
            if not os.path.isdir(path): continue
            for img_name in os.listdir(path):
                try:
                    img = cv2.imread(os.path.join(path, img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(cls)
                except: pass
        
        if len(X) == 0:
            raise ValueError("Eğitim verisi bulunamadı!")

        X = np.array(X, dtype="float32") / 255.0
        y = np.array(y)
        
        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y)
        classes_detected = lb.classes_
        joblib.dump(classes_detected, CLASSES_PATH)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        training_state["status"] = "Model Building"
        training_state["message"] = f"Model derleniyor... ({len(X_train)} veri)"
        
        model = Sequential([
            Conv2D(32, (3, 3), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)), Dropout(0.25),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)), Dropout(0.25),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)), Dropout(0.25),
            Flatten(),
            Dense(512), BatchNormalization(), Activation("relu"), Dropout(0.5),
            Dense(len(classes_detected), activation="softmax")
        ])
        
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        
        aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        
        training_state["status"] = "Training Started"
        
        history = model.fit(
            aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_val, y_val),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stop, lr_scheduler, FlaskStatusCallback()]
        )
        
        plot_training_curves(history)
        
        del X_train, X_val, y_train, y_val, X, y
        gc.collect()
        tf.keras.backend.clear_session()
        
        global_model = load_model(MODEL_PATH)
        global_class_names = classes_detected
        
        training_state["status"] = "Completed"
        training_state["message"] = "Eğitim Tamamlandı."

    except Exception as e:
        print(f"Eğitim Hatası: {str(e)}")
        training_state["status"] = "Error"
        training_state["message"] = f"Hata: {str(e)}"
    finally:
        training_state["is_training"] = False

# ============================================================
# 5. ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/train", methods=["GET", "POST"])
def trigger_training():
    if training_state["is_training"]:
        return jsonify({"status": "error", "message": "Eğitim zaten devam ediyor."})
    threading.Thread(target=train_model_background, daemon=True).start()
    return jsonify({"status": "success", "message": "Eğitim başlatıldı."})

@app.route("/train-status")
def get_training_status():
    return jsonify(training_state)

@app.route("/predict", methods=["POST"])
def predict_single_image():
    if global_model is None: return jsonify({"error": "Model yüklü değil."}), 500
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
        
        class_names_list = list(global_class_names)
        pred_class = class_names_list[pred_idx]
        
        return jsonify({"class": pred_class, "confidence": f"%{confidence * 100:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate")
def evaluate():
    global cached_results
    if global_model is None: return jsonify({"error": "Model bulunamadı."}), 500
    gc.collect()

    X_test, y_indices = [], []
    if len(global_class_names) == 0: return jsonify({"error": "Sınıf listesi boş."}), 500
    
    class_names_list = list(global_class_names)
    class_to_idx = {cls: i for i, cls in enumerate(class_names_list)}

    print("Evaluate: Veri okunuyor...")
    if os.path.exists(TEST_DIR):
        for cls in sorted(os.listdir(TEST_DIR)):
            if cls not in class_to_idx: continue
            cls_folder = os.path.join(TEST_DIR, cls)
            if not os.path.isdir(cls_folder): continue
            idx = class_to_idx[cls]
            for img_name in os.listdir(cls_folder):
                try:
                    img = cv2.imread(os.path.join(cls_folder, img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype("float32") / 255.0
                    X_test.append(img)
                    y_indices.append(idx)
                except: pass

    X_test = np.array(X_test)
    y_indices = np.array(y_indices)

    if len(X_test) == 0: return jsonify({"error": "Test verisi yok."}), 400

    try:
        print("Evaluate: Tahmin yapılıyor...")
        y_pred_probs = global_model.predict(X_test, batch_size=16, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_indices, y_pred)
        report = classification_report(y_indices, y_pred, target_names=class_names_list, output_dict=True)
        
        # --- CACHE'E AT ---
        cached_results["y_true"] = y_indices
        cached_results["y_pred"] = y_pred
        cached_results["y_probs"] = y_pred_probs
        cached_results["class_names"] = class_names_list
        print("Evaluate: Sonuçlar önbelleklendi.")

        del X_test
        gc.collect()

        return jsonify({
            "accuracy": f"{acc * 100:.2f}%",
            "model_type": "CNN (GPU)",
            "class_report": report
        })
        
    except Exception as e:
        print(f"Evaluate Kritik Hata: {e}")
        gc.collect()
        return jsonify({"error": f"Değerlendirme hatası: {str(e)}"}), 500

# --- GRAFİK ÇİZİMİ (NESNE TABANLI & GÜVENLİ) ---
@app.route("/get-plot/<plot_type>")
def get_plot(plot_type):
    global cached_results
    
    if cached_results["y_true"] is None:
        return jsonify({"error": "Lütfen önce 'Test Et' butonuna basın."}), 400

    filename = f"{plot_type}.png"
    save_path = os.path.join(plots_dir, filename)
    
    if plot_type == "training_curve":
        if os.path.exists(save_path):
            return send_from_directory(plots_dir, filename)
        else:
            return jsonify({"error": "Eğitim grafiği bulunamadı"}), 404

    y_true = cached_results["y_true"]
    y_pred = cached_results["y_pred"]
    y_probs = cached_results["y_probs"]
    class_names = cached_results["class_names"]

    # Çizim hatası olursa yakala
    try:
        with plot_lock:
            # Önceki çizimleri temizle
            plt.close('all')
            
            # Nesne tabanlı çizim (Figure ve Axis oluştur)
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
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig) # Figürü bellekten sil
            gc.collect() # Çöp topla
        
        return send_from_directory(plots_dir, filename)

    except Exception as e:
        print(f"Grafik hatası ({plot_type}): {e}")
        return jsonify({"error": f"Grafik oluşturulamadı: {e}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)