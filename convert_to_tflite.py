import tensorflow as tf

# Mevcut modeli yükle
model = tf.keras.models.load_model("models/cnn_fruit_best_model.h5")

# TFLite dönüştürücü oluştur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizasyonları aç (Boyutu ve RAM'i daha da düşürür)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Dönüştür
tflite_model = converter.convert()

# Kaydet
with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Başarılı! 'models/model.tflite' dosyası oluşturuldu.")

