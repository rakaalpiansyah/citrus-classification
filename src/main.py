import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi Logging Modern
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CitrusClassifier:
    def __init__(self, data_path: str | Path):
        """
        Inisialisasi pipeline klasifikasi Citrus.
        """
        self.data_path = Path(data_path)
        self.dataset: pd.DataFrame | None = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Dictionary model untuk mempermudah iterasi dan komparasi
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42)
        }

    def load_data(self) -> None:
        """Tahap 1: Memuat dataset ke dalam memory."""
        if not self.data_path.exists():
            logger.error(f"Dataset tidak ditemukan pada rute: {self.data_path}")
            raise FileNotFoundError(f"Dataset tidak ditemukan di: {self.data_path}")
        
        self.dataset = pd.read_csv(self.data_path)
        logger.info(f"Dataset berhasil dimuat. Total baris: {len(self.dataset)}")

    def preprocess_data(self) -> None:
        """Tahap 2: Preprocessing (Encoding, Splitting, dan Scaling)."""
        if self.dataset is None:
            raise ValueError("Dataset belum dimuat. Panggil load_data() terlebih dahulu.")

        # Encoding: 'orange' -> 0, 'grapefruit' -> 1
        y = self.label_encoder.fit_transform(self.dataset['name'])
        X = self.dataset.drop(columns=['name'])

        # Data Splitting (80% Latih, 20% Uji)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Feature Scaling: Esensial untuk performa SVM dan Naive Bayes
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info("Preprocessing selesai: Label ter-encode, Splitting 80:20, Fitur di-scale.")

    def train_and_evaluate(self) -> pd.DataFrame:
        """Tahap 3 & 4: Pelatihan Model dan Evaluasi/Komparasi."""
        if any(v is None for v in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Data latih/uji belum siap. Jalankan preprocess_data().")

        logger.info("=" * 50)
        logger.info("MEMULAI PELATIHAN DAN EVALUASI 3 MODEL")
        logger.info("=" * 50)

        results_summary = []

        for model_name, model in self.models.items():
            # Proses Latih (Training)
            model.fit(self.X_train, self.y_train)
            
            # Proses Prediksi (Testing)
            y_pred = model.predict(self.X_test)
            
            # Evaluasi Kinerja
            accuracy = accuracy_score(self.y_test, y_pred) # pyright: ignore[reportArgumentType]
            
            print(f"\n--- {model_name.upper()} ---")
            print(classification_report(
                self.y_test, 
                y_pred, 
                target_names=self.label_encoder.classes_
            ))

            # Menyimpan hasil untuk komparasi akhir
            results_summary.append({
                "Model": model_name,
                "Accuracy (%)": round(accuracy * 100, 2)
            })

        # Mengembalikan tabel komparasi agar mudah dianalisis
        return pd.DataFrame(results_summary)
    
    def predict_new_data(self, input_data: dict, model_name: str = "Support Vector Machine (SVM)") -> str:
        """Melakukan prediksi untuk data buah baru dari GUI."""
        if any(v is None for v in [self.X_train]):
            raise ValueError("Model belum dilatih!")
            
        # Konversi input dictionary ke DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Scale data input menggunakan scaler yang sudah di-fit saat training
        scaled_input = self.scaler.transform(df_input)
        
        # Lakukan prediksi
        model = self.models[model_name]
        prediction_encoded = model.predict(scaled_input)[0]
        
        # Kembalikan ke label asli (orange / grapefruit)
        result = self.label_encoder.inverse_transform([prediction_encoded])[0]
        return result.capitalize()

    def run_pipeline(self) -> None:
        """Menjalankan seluruh tahapan Machine Learning Pipeline."""
        try:
            self.load_data()
            self.preprocess_data()
            comparison_df = self.train_and_evaluate()
            
            print("\n" + "=" * 50)
            print("KESIMPULAN KOMPARASI MODEL")
            print("=" * 50)
            print(comparison_df.sort_values(by="Accuracy (%)", ascending=False).to_string(index=False))
            print("=" * 50)
            
        except Exception as e:
            logger.error(f"Terjadi kesalahan pada pipeline: {e}")


if __name__ == "__main__":
    # Path dinamis menggunakan pathlib (Sangat aman dari error pathing)
    # Ini akan mencari folder 'data' yang sejajar dengan folder tempat main.py berada
    CURRENT_DIR = Path(__file__).parent
    DATASET_PATH = CURRENT_DIR.parent / "data" / "citrus.csv"
    
    # Inisialisasi dan jalankan pipeline klasifikasi
    classifier = CitrusClassifier(data_path=DATASET_PATH)
    classifier.run_pipeline()