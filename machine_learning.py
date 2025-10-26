import pandas as pd
df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.info())
print(df.head())

print(df.isnull().sum())
df = df.drop_duplicates()

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()   

# === LANGKAH 4: EXPLORATORY DATA ANALYSIS (EDA) ===

# 1. Statistik Deskriptif
print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

# 2. Histogram Distribusi IPK
plt.figure(figsize=(6,4))
sns.histplot(df['IPK'], bins=5, kde=True)
plt.title("Distribusi IPK Mahasiswa")
plt.xlabel("IPK")
plt.ylabel("Frekuensi")
plt.show()

# 3. Scatterplot: IPK vs Waktu Belajar
plt.figure(figsize=(6,4))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("Hubungan IPK vs Waktu Belajar (Dilihat dari Status Kelulusan)")
plt.show()

# 4. Heatmap Korelasi Antar Fitur
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Variabel")
plt.show()

# === LANGKAH 5: FEATURE ENGINEERING ===

# 1. Membuat fitur rasio absensi terhadap total pertemuan (misal 14 kali pertemuan)
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14

# 2. Membuat fitur interaksi antara IPK dan waktu belajar
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# 3. Simpan dataset baru yang sudah diproses
df.to_csv("processed_kelulusan.csv", index=False)

print("\n=== DATASET BARU DENGAN FITUR TAMBAHAN ===")
print(df.head())
print("\nFile processed_kelulusan.csv berhasil disimpan.")

# === LANGKAH 6: DATA SPLITTING SESUAI INSTRUKSI DOSEN (70% TRAIN, 15% VAL, 15% TEST) ===
from sklearn.model_selection import train_test_split

# Cek terlebih dahulu jumlah data
print("Jumlah data sebelum digandakan:", len(df))

# Gandakan dataset agar stratified split tidak error (karena data awal hanya 10 baris)
df = pd.concat([df] * 3, ignore_index=True)  # Gandakan 3x → dari 10 jadi 30 data

print("Jumlah data setelah digandakan:", len(df))
print("Distribusi kelas setelah digandakan:")
print(df['Lulus'].value_counts())  # Pastikan distribusi seimbang

# Pisahkan fitur dan target
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Pertama: Split 70% Train dan 30% (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Kedua: Split 30% tadi menjadi 15% Val dan 15% Test (dibagi 50:50)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Tampilkan hasil
print("\n=== HASIL SPLIT DATA ===")
print("Train Set      :", X_train.shape)
print("Validation Set :", X_val.shape)
print("Test Set       :", X_test.shape)

# === SIMPAN DATASET SPLIT AGAR BISA DIPAKAI UNTUK MODELING (PILIHAN A) ===
X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Semua file CSV split telah disimpan di folder proyek!")

# === LANGKAH 2: BASELINE MODEL DENGAN PIPELINE ===
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Deteksi kolom numerik
num_cols = X_train.select_dtypes(include="number").columns

# Preprocessing Pipeline
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),  # Isi missing value jika ada
        ("sc", StandardScaler())                   # Standarisasi fitur
    ]), num_cols),
], remainder="drop")

# Model baseline Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

# Training model baseline
pipe_lr.fit(X_train, y_train)

# Prediksi pada validation set
y_val_pred = pipe_lr.predict(X_val)

# Evaluasi baseline
print("\n=== BASELINE MODEL (Logistic Regression) ===")
print("F1 Score Validation (Macro):", f1_score(y_val, y_val_pred, average="macro"))
print("\n=== Classification Report Validation ===")
print(classification_report(y_val, y_val_pred, digits=3))

# === LANGKAH 3: MODEL ALTERNATIF (Random Forest) ===
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)

pipe_rf = Pipeline([("pre", pre), ("clf", rf)])
pipe_rf.fit(X_train, y_train)

y_val_rf = pipe_rf.predict(X_val)
print("\n=== RANDOM FOREST BASELINE ===")
print("RandomForest F1 Score Validation (Macro):", f1_score(y_val, y_val_rf, average="macro"))

# === LANGKAH 4: GRID SEARCH + CV ===
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Konfigurasi K-Fold dengan stratifikasi (agar distribusi 0/1 tetap seimbang di setiap fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Parameter yang ingin diuji di Grid Search
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(
    pipe_rf,
    param_grid=param,
    cv=skf,
    scoring="f1_macro",
    n_jobs=-1,      # gunakan semua core agar cepat
    verbose=1       # tampilkan progress saat tuning berjalan
)

print("\n=== GridSearchCV DIMULAI... ===")
gs.fit(X_train, y_train)

print("\n=== HASIL GRID SEARCH ===")
print("Best Params:", gs.best_params_)
print("Best Cross-Validation F1 Score:", gs.best_score_)

# Simpan model terbaik
best_rf = gs.best_estimator_

# Evaluasi di Validation Set
y_val_best = best_rf.predict(X_val)
print("\nBest RF F1 Score di Validation Set:", f1_score(y_val, y_val_best, average="macro"))

# === LANGKAH 5: EVALUASI DI TEST SET ===
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Prediksi di test set
y_test_pred = best_rf.predict(X_test)

# Hitung F1 Score
f1_test = f1_score(y_test, y_test_pred, average="macro")
print("\n=== EVALUASI DI TEST SET ===")
print("F1 Score (Test Set):", f1_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Lulus", "Lulus"])
disp.plot()

# === LANGKAH 6: SIMPAN MODEL ===
import joblib

# Pastikan final_model didefinisikan
final_model = best_rf  # gunakan model Random Forest terbaik dari tuning

# === SIMPAN MODEL ===
import joblib
joblib.dump(final_model, "model_kelulusan.pkl")
print("Model tersimpan ke model_kelulusan.pkl")


# Membaca file hasil split dari Pertemuan 5
X_train = pd.read_csv("X_train.csv")
X_val   = pd.read_csv("X_val.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze("columns")
y_val   = pd.read_csv("y_val.csv").squeeze("columns")
y_test  = pd.read_csv("y_test.csv").squeeze("columns")

# Mengecek ukuran data
print("Ukuran data:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

# === LANGKAH 2 — PIPELINE & BASELINE RANDOM FOREST ===
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# Pilih kolom numerik
num_cols = X_train.select_dtypes(include="number").columns

# Preprocessing pipeline
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

# Model Random Forest dasar
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

# Gabungkan preprocessing dan model
pipe = Pipeline([
    ("pre", pre),
    ("clf", rf)
])

# Latih model baseline
pipe.fit(X_train, y_train)

# Prediksi pada data validasi
y_val_pred = pipe.predict(X_val)

# Evaluasi baseline
print("=== BASELINE RANDOM FOREST ===")
print("F1 Score (Validation):", f1_score(y_val, y_val_pred, average="macro"))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, digits=3))

# === LANGKAH 3 — VALIDASI SILANG ===
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Siapkan 5-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Jalankan cross validation menggunakan F1-macro
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)

# Tampilkan hasil validasi silang
print("=== VALIDASI SILANG (CROSS VALIDATION) ===")
print("F1-macro rata-rata (train):", scores.mean())
print("Standar deviasi:", scores.std())
print("Semua skor:", scores)

# Langkah 4 — Tuning Ringkas (GridSearch)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# Parameter grid yang akan diuji
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

# Inisialisasi GridSearchCV
gs = GridSearchCV(
    estimator=pipe,             # pipeline dengan RandomForest
    param_grid=param,           # kombinasi parameter
    cv=skf,                     # Stratified K-Fold cross validation
    scoring="f1_macro",         # metrik evaluasi utama
    n_jobs=-1,                  # gunakan semua core CPU
    verbose=1                   # tampilkan progres
)

# Latih model dengan pencarian grid
gs.fit(X_train, y_train)

# Tampilkan hasil terbaik
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

# Gunakan model terbaik hasil tuning
best_model = gs.best_estimator_

# Evaluasi pada data validasi
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))

# ======================================================
# LANGKAH 5 — EVALUASI AKHIR (TEST SET)
# ======================================================

from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

# Gunakan model terbaik dari hasil tuning
final_model = best_model  # jika baseline lebih baik, bisa gunakan pipe

# Prediksi pada data test
y_test_pred = final_model.predict(X_test)

# Evaluasi performa model
print("=== EVALUASI AKHIR (TEST SET) ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ======================================================
# ROC-AUC (bila model memiliki predict_proba)
# ======================================================
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    plt.tight_layout()
    plt.savefig("pr_test.png", dpi=120)

print("\n=== SELESAI — Hasil evaluasi dan grafik disimpan ===")

# ======================================================
# LANGKAH 6 — PENTINGNYA FITUR (FEATURE IMPORTANCE)
# ======================================================

try:
    import numpy as np

    # Ambil nilai importance dari model Random Forest
    importances = final_model.named_steps["clf"].feature_importances_

    # Ambil nama fitur dari hasil preprocessing (ColumnTransformer)
    fn = final_model.named_steps["pre"].get_feature_names_out()

    # Urutkan fitur berdasarkan tingkat kepentingan (descending)
    top_features = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)

    print("=== Top Feature Importance (Gini) ===")
    for name, val in top_features[:10]:  # tampilkan 10 fitur teratas
        print(f"{name}: {val:.4f}")

    # Visualisasi feature importance
    import matplotlib.pyplot as plt

    feat_names = [f[0] for f in top_features[:10]]
    feat_values = [f[1] for f in top_features[:10]]

    plt.figure(figsize=(8, 5))
    plt.barh(feat_names[::-1], feat_values[::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()

except Exception as e:
    print("Feature importance tidak tersedia:", e)


# ======================================================
# (6b) OPSIONAL — PERMUTATION IMPORTANCE
# ======================================================
# Dapat digunakan untuk mengecek kepentingan fitur dari sisi prediksi (lebih akurat tapi lambat)
# from sklearn.inspection import permutation_importance
# r = permutation_importance(final_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
# sorted_idx = r.importances_mean.argsort()[::-1]
# print("\n=== Top 10 Permutation Importance ===")
# for i in sorted_idx[:10]:
#     print(f"{X_val.columns[i]}: {r.importances_mean[i]:.4f}")

# ======================================================
# LANGKAH 7 — SIMPAN MODEL
# ======================================================

import joblib

# Simpan model terbaik ke dalam file .pkl
joblib.dump(final_model, "rf_model.pkl")

print(" Model disimpan sebagai 'rf_model.pkl'")

# ======================================================
# LANGKAH 8 — CEK INFERENCE LOKAL
# ======================================================

import pandas as pd
import joblib

# Muat kembali model yang sudah disimpan
mdl = joblib.load("rf_model.pkl")

# Contoh data baru (fiktif) — pastikan kolomnya sesuai dengan dataset asli
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4 * 7
}])

# Lakukan prediksi
pred = mdl.predict(sample)[0]
print("Prediksi:", int(pred))

# ==========================================================
# Langkah 1 — Siapkan Data
# ==========================================================

# Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Membaca dataset hasil preprocessing dari pertemuan sebelumnya
df = pd.read_csv("processed_kelulusan.csv")

#Memisahkan fitur (X) dan label/target (y)
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

#Normalisasi data agar nilai tiap fitur berada pada skala seragam
sc = StandardScaler()
Xs = sc.fit_transform(X)

#Membagi data menjadi Train (70%), Validation (15%), dan Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)
from sklearn.model_selection import train_test_split
import numpy as np

# Cek distribusi kelas terlebih dahulu
unique, counts = np.unique(y, return_counts=True)
print("Distribusi kelas:", dict(zip(unique, counts)))

# Jika semua kelas punya minimal 2 data, pakai stratify
if np.all(counts >= 2):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
else:
    print("Salah satu kelas terlalu sedikit, membagi data tanpa stratify.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


#Menampilkan ukuran data
print("Ukuran data:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

# ==========================================================
# Langkah 2 — Bangun Model ANN
# ==========================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Membangun arsitektur jaringan saraf tiruan (Artificial Neural Network)
model = keras.Sequential([
    # Layer input menyesuaikan jumlah fitur pada dataset
    layers.Input(shape=(X_train.shape[1],)),
    
    # Hidden layer pertama dengan 32 neuron dan aktivasi ReLU
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),  # Dropout 30% untuk mencegah overfitting
    
    # Hidden layer kedua dengan 16 neuron dan aktivasi ReLU
    layers.Dense(16, activation="relu"),
    
    # Output layer — neuron tunggal dengan aktivasi sigmoid untuk klasifikasi biner
    layers.Dense(1, activation="sigmoid")
])

#Menyusun (compile) model dengan optimizer, loss function, dan metrik evaluasi
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),   # Adam optimizer dengan learning rate 0.001
    loss="binary_crossentropy",              # Cocok untuk klasifikasi biner
    metrics=["accuracy", "AUC"]              # Mengukur akurasi & area under curve
)

#Menampilkan ringkasan arsitektur model
model.summary()

# ==========================================================
# Langkah 3 — Training dengan Early Stopping
# ==========================================================

from tensorflow import keras

#Membuat callback EarlyStopping
# EarlyStopping akan menghentikan pelatihan jika 'val_loss' tidak membaik
es = keras.callbacks.EarlyStopping(
    monitor="val_loss",          # Pantau loss pada data validasi
    patience=10,                 # Hentikan jika tidak membaik selama 10 epoch berturut-turut
    restore_best_weights=True    # Kembalikan bobot terbaik yang pernah dicapai
)

#Melatih model
history = model.fit(
    X_train, y_train,             # Data training
    validation_data=(X_val, y_val),  # Data validasi untuk memantau performa
    epochs=100,                   # Maksimum 100 epoch
    batch_size=32,                # Jumlah sampel per batch
    callbacks=[es],               # Terapkan early stopping
    verbose=1                     # Tampilkan progress training
)

# ==========================================================
# Langkah 4 — Evaluasi di Test Set
# ==========================================================

from sklearn.metrics import classification_report, confusion_matrix

#Evaluasi performa model di data uji
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print("Hasil Evaluasi Model di Test Set")
print("-----------------------------------")
print(f"Loss (Test): {loss:.4f}")
print(f"Akurasi (Test): {acc:.4f}")
print(f"AUC (Test): {auc:.4f}\n")

#Prediksi probabilitas dan konversi ke kelas 0/1
y_proba = model.predict(X_test).ravel()           # hasil probabilitas (0–1)
y_pred = (y_proba >= 0.5).astype(int)             # ubah jadi label 0 atau 1

#Tampilkan confusion matrix dan classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# ==========================================================
# Langkah 5 — Visualisasi Learning Curve
# ==========================================================

import matplotlib.pyplot as plt

#Plot kurva training vs validation loss
plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")

#Tambahkan label dan judul
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve — ANN Klasifikasi Kelulusan")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

#Tata letak dan simpan gambar
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

# === Eksperimen Model ANN ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import f1_score, roc_auc_score, classification_report

# --- VARIASI 1: Ubah jumlah neuron ---
model1 = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),       # dari 32 → 64 neuron
    layers.Dropout(0.4),                       # dropout lebih besar
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

history1 = model1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=1
)

# --- VARIASI 2: Ganti optimizer ke SGD + momentum ---
model2 = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)),  # tambahkan L2 regulasi
    layers.BatchNormalization(),                    # tambahkan Batch Normalization
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),                            # dropout lebih besar
    layers.Dense(1, activation="sigmoid")
])

model2.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=1
)

# --- Evaluasi di Test Set ---
for i, model in enumerate([model1, model2], start=1):
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n=== Model {i} ===")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC      : {auc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))
