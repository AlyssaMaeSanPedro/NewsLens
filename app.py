from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os
import re
import keras
from keras.models import Model
from keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import feedparser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load(filename):
    with open(os.path.join(BASE_DIR, filename), "rb") as f:
        return pickle.load(f)

@keras.saving.register_keras_serializable()
class AttentionLayer(Layer):
    supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        score = tf.squeeze(tf.tanh(tf.matmul(x, self.W)), axis=-1)
        if mask is not None:
            mask = tf.cast(mask, dtype=score.dtype)
            score += (1.0 - mask) * -1e9
        weights = tf.nn.softmax(score, axis=-1)
        weights = tf.expand_dims(weights, axis=-1)
        return tf.reduce_sum(x * weights, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        return None

bilstm = keras.saving.load_model(
    os.path.join(BASE_DIR, "bilstm_model.keras"),
    custom_objects={"AttentionLayer": AttentionLayer}
)
feature_extractor = Model(inputs=bilstm.input,
                          outputs=bilstm.get_layer('dense').output)

tokenizer = load("tokenizer.pkl")
le        = load("label_encoder.pkl")
svm       = load("svm_model.pkl")

LSTM_WEIGHT = 0.6
SVM_WEIGHT  = 0.4

RSS_FEEDS = {
    "GMA News":  "https://data.gmanetwork.com/gno/rss/news/ulatfilipino/feed.xml",
    "Inquirer":  "https://newsinfo.inquirer.net/feed",
    "Rappler":   "https://www.rappler.com/feed",
    "Philstar":  "https://www.philstar.com/rss/headlines",
    "ABS-CBN":   "https://news.abs-cbn.com/rss/news",
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Zàáâãäåæçèéêëìíîïñòóôõöùúûüýÿ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify(headline):
    seq    = tokenizer.texts_to_sequences([headline])
    padded = pad_sequences(seq, maxlen=50, padding="post", truncating="post")
    padded_tensor = tf.constant(padded)

    lstm_probs = bilstm(padded_tensor, training=False).numpy()[0]
    features   = feature_extractor(padded_tensor, training=False).numpy()
    svm_probs  = svm.predict_proba(features)[0]

    ensemble  = LSTM_WEIGHT * lstm_probs + SVM_WEIGHT * svm_probs
    ensemble /= ensemble.sum()

    pred_idx   = int(np.argmax(ensemble))
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(ensemble[pred_idx])

    return {
        "label":      pred_label,
        "confidence": round(confidence, 4),
        "all_scores": {
            le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(ensemble)
        }
    }

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data     = request.get_json(force=True)
    headline = data.get("text", "").strip()
    if not headline:
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_text(headline)
    if not cleaned:
        return jsonify({"error": "Text is empty after cleaning"}), 400

    return jsonify(classify(cleaned))

@app.route("/feed", methods=["GET"])
def feed():
    results = []
    errors  = []

    for source, url in RSS_FEEDS.items():
        try:
            parsed  = feedparser.parse(url)
            entries = parsed.entries[:25]
            for entry in entries:
                headline = entry.get("title", "").strip()
                if not headline:
                    continue
                cleaned = clean_text(headline)
                if not cleaned:
                    continue
                prediction = classify(cleaned)
                results.append({
                    "source":    source,
                    "headline":  headline,
                    "link":      entry.get("link", ""),
                    "published": entry.get("published", ""),
                    **prediction
                })
        except Exception as e:
            errors.append({"source": source, "error": str(e)})

    return jsonify({
        "total":   len(results),
        "results": results,
        "errors":  errors
    })

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)