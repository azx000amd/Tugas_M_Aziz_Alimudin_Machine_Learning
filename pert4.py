# DATA PREPARATION — KELULUSAN MAHASISWA
# ======================================

# Langkah 1 — Buat Dataset CSV
import pandas as pd
df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.info())
print(df.head())

df = pd.DataFrame(data)
df.to_csv("kelulusan_mahasiswa.csv", index=False)
print(" Dataset kelulusan_mahasiswa.csv berhasil dibuat.\n")

# Langkah 2 — Collection

df = pd.read_csv("kelulusan_mahasiswa.csv")
print("=== Info Dataset ===")
print(df.info())
print("\n=== 5 Data Teratas ===")
print(df.head())


# Langkah 3 — Cleaning

print("\n=== Cek Missing Value ===")
print(df.isnull().sum())

# Hapus duplikat (jika ada)
df = df.drop_duplicates()

# Deteksi Outlier dengan Boxplot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,3))
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()


# Langkah 4 — Exploratory Data Analysis (EDA)
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# Histogram distribusi IPK
plt.figure(figsize=(5,3))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK Mahasiswa")
plt.show()

# Scatterplot hubungan IPK vs Waktu Belajar
plt.figure(figsize=(5,3))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', palette='coolwarm')
plt.title("IPK vs Waktu Belajar berdasarkan Kelulusan")
plt.show()

# Heatmap korelasi antar variabel
plt.figure(figsize=(5,3))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi")
plt.show()


# Langkah 5 — Feature Engineering
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan dataset hasil feature engineering
df.to_csv("processed_kelulusan.csv", index=False, sep=';')
print("\n File processed_kelulusan.csv berhasil disimpan dengan fitur baru.")

# ------------------------------------------
# Langkah 6 — Splitting Dataset
# ------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Stratified split: Train 70%, Validation 15%, Test 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\n=== Ukuran Dataset ===")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

print("\n Data Preparation Selesai! Dataset siap digunakan untuk Modeling.")
