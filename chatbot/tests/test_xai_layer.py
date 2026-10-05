import pytest
import numpy as np
from app.ml_service.model_adapters import WaitingTimeModelAdapter, CrowdingModelAdapter
from app.ml_service.xai_explainer import explain_prediction, get_feature_names_from_preprocessor
from app.schemas.prediction_schema import PredictionInputData
import joblib
from pathlib import Path

# Resolve the supervised artifact directory relative to the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
ml_dir = _REPO_ROOT / "backend" / "artifacts" / "supervised"

_ARTIFACTS_AVAILABLE = (
    (ml_dir / "final_xgb_regressor.pkl").exists()
    and (ml_dir / "final_xgb_classifier.pkl").exists()
    and (ml_dir / "preprocessor_reg.pkl").exists()
)

pytestmark = pytest.mark.skipif(
    not _ARTIFACTS_AVAILABLE,
    reason="Trained ML artifact files not present (expected in CI environment without model binaries)",
)


@pytest.fixture
def loaded_adapters():
    reg_model = joblib.load(ml_dir / "final_xgb_regressor.pkl")
    cls_model = joblib.load(ml_dir / "final_xgb_classifier.pkl")
    prep_scaler = joblib.load(ml_dir / "preprocessor_reg.pkl")

    wt_adapter = WaitingTimeModelAdapter(model_artifact=reg_model, preprocessor=prep_scaler)
    wt_adapter.preprocessor_pipeline = prep_scaler

    cr_adapter = CrowdingModelAdapter(model_artifact=cls_model, preprocessor=prep_scaler)
    cr_adapter.preprocessor_pipeline = prep_scaler

    return wt_adapter, cr_adapter


def test_xai_does_not_change_prediction(loaded_adapters):
    wt_adapter, cr_adapter = loaded_adapters
    inp = PredictionInputData(
        features={
            "arrival_rate": 32.0,
            "occupancy_percent": 82.0,
            "patients_waiting": 24,
            "available_beds": 8,
            "available_doctors": 5,
            "available_nurses": 9,
            "severity_level": 3.0,
            "hour_of_day": 18,
            "day_of_week": 4,
            "month": 7,
        }
    )

    # 1. Run prediction
    res1 = wt_adapter.predict_waiting_time(inp)
    res2 = wt_adapter.predict_waiting_time(inp)

    # Verify prediction numerical value is unchanged and deterministic
    assert res1.prediction["estimated_wait_minutes"] == res2.prediction["estimated_wait_minutes"]

    cr_res1 = cr_adapter.predict_crowding(inp)
    cr_res2 = cr_adapter.predict_crowding(inp)
    assert cr_res1.prediction["crowding_level"] == cr_res2.prediction["crowding_level"]
    assert cr_res1.prediction["crowding_score"] == cr_res2.prediction["crowding_score"]


def test_xai_explanation_fields_validity(loaded_adapters):
    wt_adapter, cr_adapter = loaded_adapters
    inp = PredictionInputData(
        features={
            "arrival_rate": 35.0,
            "occupancy_percent": 85.0,
            "patients_waiting": 30,
            "available_beds": 5,
            "available_doctors": 4,
            "available_nurses": 8,
            "severity_level": 3.5,
            "hour_of_day": 19,
        }
    )

    res = wt_adapter.predict_waiting_time(inp)
    assert "explanation" in res.prediction
    explanation = res.prediction["explanation"]
    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) > 0

    for factor in explanation["top_factors"]:
        assert isinstance(factor["feature"], str)
        assert factor["direction"] in ["increases", "decreases"]
        assert 0.0 <= factor["importance"] <= 1.0
        assert not np.isnan(factor["importance"])
        assert not np.isnan(factor["shap_value"])


def test_xai_unknown_features_handled_safely(loaded_adapters):
    wt_adapter, _ = loaded_adapters
    inp = PredictionInputData(
        features={
            "unknown_custom_feature_foo": 999.0,
            "another_random_feature_bar": "hello",
            "arrival_rate": 20.0,
        }
    )

    res = wt_adapter.predict_waiting_time(inp)
    assert res.is_available is True
    assert "explanation" in res.prediction
    assert isinstance(res.prediction["explanation"]["top_factors"], list)
