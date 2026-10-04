import os
import json
import logging
import zipfile
import tempfile
import joblib
import h5py
import pandas as pd
import numpy as np

# Backward compatibility shim for unpickling scikit-learn 1.6 models on scikit-learn 1.7+
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        _ct._RemainderColsList = type("_RemainderColsList", (list,), {})
    from sklearn.impute import SimpleImputer
    if not hasattr(SimpleImputer, "_fill_dtype"):
        SimpleImputer._fill_dtype = property(lambda self: getattr(self, "_fit_dtype", np.float64))
except Exception:
    pass

logger = logging.getLogger("erflow.artifact_loader")


class LSTMWeights:
    """Container for the extracted weights of er_patient_arrival_lstm.keras."""

    def __init__(self, k1, rk1, b1, k2, rk2, b2, kd1, bd1, kd2, bd2):
        self.k_lstm1 = k1
        self.rk_lstm1 = rk1
        self.b_lstm1 = b1

        self.k_lstm2 = k2
        self.rk_lstm2 = rk2
        self.b_lstm2 = b2

        self.k_d1 = kd1
        self.b_d1 = bd1

        self.k_d2 = kd2
        self.b_d2 = bd2

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def _lstm_layer(self, X, kernel, recurrent_kernel, bias, return_sequences=False):
        batch_size, timesteps, _ = X.shape
        units = recurrent_kernel.shape[0]

        h = np.zeros((batch_size, units), dtype=np.float32)
        c = np.zeros((batch_size, units), dtype=np.float32)

        outputs = []
        for t in range(timesteps):
            xt = X[:, t, :]
            z = np.dot(xt, kernel) + np.dot(h, recurrent_kernel) + bias
            i = self._sigmoid(z[:, 0:units])
            f = self._sigmoid(z[:, units:2 * units])
            cand = np.tanh(z[:, 2 * units:3 * units])
            o = self._sigmoid(z[:, 3 * units:4 * units])

            c = f * c + i * cand
            h = o * np.tanh(c)
            if return_sequences:
                outputs.append(h)

        if return_sequences:
            return np.stack(outputs, axis=1)
        return h

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Run exact forward pass through 2 LSTM layers and 2 Dense layers."""
        out_lstm1 = self._lstm_layer(X, self.k_lstm1, self.rk_lstm1, self.b_lstm1, return_sequences=True)
        out_lstm2 = self._lstm_layer(out_lstm1, self.k_lstm2, self.rk_lstm2, self.b_lstm2, return_sequences=False)
        out_d1 = np.maximum(0, np.dot(out_lstm2, self.k_d1) + self.b_d1)  # Dense ReLU
        out_d2 = np.dot(out_d1, self.k_d2) + self.b_d2                   # Dense Linear
        return out_d2


class ArtifactLoader:
    """Singleton responsible for locating and loading ML model artifacts and datasets."""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.artifacts_dir = os.path.join(self.base_dir, "artifacts")

        # Supervised
        self.xgb_classifier = None
        self.xgb_regressor = None
        self.supervised_preprocessor = None
        self.label_encoder = None

        # Unsupervised
        self.kmeans_model = None
        self.unsupervised_scaler = None
        self.pca_model = None
        self.cluster_profiles = {}
        self.dbscan_params = {}

        # Deep Learning
        self.lstm_model = None
        self.lstm_feature_scaler = None
        self.lstm_target_scaler = None
        self.lstm_config = {}

        # Reference Dataset Buffer
        self.dataset_df = None
        self.is_loaded = False
        try:
            self.load_all()
        except Exception as e:
            logger.warning(f"[ArtifactLoader] Initial load note: {e}")

    def load_all(self):
        """Load all model artifacts into memory."""
        if self.is_loaded:
            return

        logger.info(f"Loading ML artifacts from: {self.artifacts_dir}")

        # 1. Supervised Learning Artifacts
        sup_dir = os.path.join(self.artifacts_dir, "supervised")
        try:
            self.xgb_classifier = joblib.load(os.path.join(sup_dir, "final_xgb_classifier.pkl"))
            self.xgb_regressor = joblib.load(os.path.join(sup_dir, "final_xgb_regressor.pkl"))
            self.supervised_preprocessor = joblib.load(os.path.join(sup_dir, "preprocessor_reg.pkl"))
            self.label_encoder = joblib.load(os.path.join(sup_dir, "label_encoder.pkl"))
            logger.info("Supervised learning artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load supervised artifacts: {e}", exc_info=True)
            raise

        # 2. Unsupervised Learning Artifacts
        unsup_dir = os.path.join(self.artifacts_dir, "unsupervised")
        try:
            self.kmeans_model = joblib.load(os.path.join(unsup_dir, "kmeans_model.joblib"))
            self.unsupervised_scaler = joblib.load(os.path.join(unsup_dir, "unsupervised_scaler.joblib"))
            self.pca_model = joblib.load(os.path.join(unsup_dir, "pca_model.joblib"))

            with open(os.path.join(unsup_dir, "cluster_profiles.json"), "r") as fp:
                self.cluster_profiles = json.load(fp)

            with open(os.path.join(unsup_dir, "dbscan_params.json"), "r") as fp:
                self.dbscan_params = json.load(fp)
            logger.info("Unsupervised learning artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load unsupervised artifacts: {e}", exc_info=True)
            raise

        # 3. Deep Learning Artifacts
        dl_dir = os.path.join(self.artifacts_dir, "deep_learning")
        try:
            keras_path = os.path.join(dl_dir, "er_patient_arrival_lstm.keras")

            # Load scalers and configuration
            self.lstm_feature_scaler = joblib.load(os.path.join(dl_dir, "er_feature_scaler.pkl"))
            self.lstm_target_scaler = joblib.load(os.path.join(dl_dir, "er_target_scaler.pkl"))

            with open(os.path.join(dl_dir, "er_lstm_config.json"), "r") as fp:
                self.lstm_config = json.load(fp)

            # Extract weights directly from .keras archive for robust, zero-DLL execution
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(keras_path, "r") as z:
                    z.extractall(tmp)
                h5_path = os.path.join(tmp, "model.weights.h5")
                with h5py.File(h5_path, "r") as h5:
                    self.lstm_model = LSTMWeights(
                        k1=h5["layers/lstm/cell/vars/0"][:],
                        rk1=h5["layers/lstm/cell/vars/1"][:],
                        b1=h5["layers/lstm/cell/vars/2"][:],
                        k2=h5["layers/lstm_1/cell/vars/0"][:],
                        rk2=h5["layers/lstm_1/cell/vars/1"][:],
                        b2=h5["layers/lstm_1/cell/vars/2"][:],
                        kd1=h5["layers/dense/vars/0"][:],
                        bd1=h5["layers/dense/vars/1"][:],
                        kd2=h5["layers/dense_1/vars/0"][:],
                        bd2=h5["layers/dense_1/vars/1"][:],
                    )

            logger.info("Deep learning LSTM model and scalers loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load deep learning artifacts: {e}", exc_info=True)
            raise

        # 4. Reference Historical Dataset
        data_path = os.path.join(self.artifacts_dir, "data", "ER_dataset.csv")
        if os.path.exists(data_path):
            try:
                self.dataset_df = pd.read_csv(data_path)
                logger.info(f"Loaded reference dataset with {len(self.dataset_df)} records.")
            except Exception as e:
                logger.warning(f"Could not load reference dataset: {e}")

        self.is_loaded = True


# Global Singleton Instance
artifact_loader = ArtifactLoader()
