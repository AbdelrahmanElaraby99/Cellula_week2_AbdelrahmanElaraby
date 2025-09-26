# type: ignore[import]

# ---- IMPORTS ----
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pickle


# ---- LOAD TOKENIZER and LABEL ENCODER ----
with open("helping_files/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("helping_files/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ---- TEXT PREPROCESSING ----
my_stop_words = {'in','at','on','what','as','be','because','been','before','being','below','between','both','but','by','can','when','where','which','while','who','whom','why','of','off','once','only','or','other','our','ours','ourselves','out','over','own','than','that','the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','a'}

def preprocess_text(text, stop_words):
    # Keep original casing, only lower for stopword comparison
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    return " ".join(tokens)


# ---- CLASS NAMES ----
classes = ['Child Sexual Exploitation', 'Elections', 'Non-Violent Crimes', 'Safe', 'Sex-Related Crimes', 'Suicide & Self-Harm', 'Unknown S-Type', 'Violent Crimes', 'unsafe']


# ---- SETTINGS ----
MAX_WORDS = 4950    
MAX_LEN = 120      
EMBEDDING_DIM = 100 
NUM_CLASSES = 9     


# ---- REBUILD MODEL ARCHITECTURE ----
def build_model():
    model = Sequential()
    model.add(Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN,
        trainable=True
    ))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    return model

# ---- LOAD MODEL + WEIGHTS ----
model = build_model()
model.build(input_shape=(None, MAX_LEN))   # Tell Keras input shape
model.load_weights("helping_files/best_model.weights.h5")

# ---- PREDICTION FUNCTION ----
def classify_text(text):
    # 1. Apply same preprocessing as training
    text = preprocess_text(text, my_stop_words)

    # 2. Tokenizer applies filters (punctuation removal etc.)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    # 3. Predict
    pred = model.predict(padded, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred))

    class_name = label_encoder.inverse_transform([class_idx])[0]
    return class_name, confidence


# Export the model + classify_text function for reuse
__all__ = ["classify_text", "model", "tokenizer", "label_encoder"]