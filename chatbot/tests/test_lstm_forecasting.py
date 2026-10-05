import pytest
import numpy as np
from pathlib import Path
from app.schemas.prediction_schema import PredictionInputData
from app.ml_service.model_registry import model_registry

# Skip the entire module when the LSTM artifact is not present (CI without model binaries)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LSTM_ARTIFACT = _REPO_ROOT / "backend" / "artifacts" / "deep_learning" / "er_patient_arrival_lstm.keras"

pytestmark = pytest.mark.skipif(
    not _LSTM_ARTIFACT.exists(),
    reason="LSTM .keras artifact not present (expected in CI environment without model binaries)",
)


def test_lstm_model_registered_and_loaded():
    model = model_registry.get_model("patient_volume_model")
    assert model is not None
    assert hasattr(model, "predict_patient_volume")


def test_lstm_sequence_map_inputs():
    model = model_registry.get_model("patient_volume_model")
    input_data = PredictionInputData(
        features={"arrival_rate": 32.0, "hour_of_day": 14, "day_of_week": 2, "month": 5},
        triage_level="Standard",
        time_window="24h",
    )
    sequence = model.map_inputs(input_data)

    assert sequence.shape == (1, 168, 17)
    assert not np.isnan(sequence).any()


def test_lstm_inference_outputs_deterministic_and_no_random():
    model = model_registry.get_model("patient_volume_model")
    input_data = PredictionInputData(
        features={"arrival_rate": 25.0, "hour_of_day": 10, "day_of_week": 1, "month": 3},
        triage_level="Standard",
        time_window="24h",
    )

    resp1 = model.predict_patient_volume(input_data)
    resp2 = model.predict_patient_volume(input_data)

    assert resp1.is_available is True
    assert resp2.is_available is True

    # Exact deterministic equivalence (No np.random!)
    assert resp1.prediction["horizons"] == resp2.prediction["horizons"]
    assert resp1.prediction["predicted_volume"] == resp2.prediction["predicted_volume"]
    assert len(resp1.prediction["series"]) == 24


def test_lstm_horizons_cumulative_monotonicity():
    model = model_registry.get_model("patient_volume_model")
    input_data = PredictionInputData(
        features={"arrival_rate": 35.0, "hour_of_day": 18, "day_of_week": 4, "month": 8},
        triage_level="Standard",
        time_window="24h",
    )
    resp = model.predict_patient_volume(input_data)
    h = resp.prediction["horizons"]

    assert h["1h"] >= 1
    assert h["3h"] >= h["1h"]
    assert h["6h"] >= h["3h"]
    assert h["24h"] >= h["6h"]
