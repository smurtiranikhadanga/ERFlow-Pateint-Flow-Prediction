import pytest
from app.schemas.prediction_schema import PredictionInputData
from app.ml_service.prediction_service import prediction_service
from app.ml_service.model_adapters import (
    WaitingTimeModelAdapter,
    CrowdingModelAdapter,
)
import joblib
from pathlib import Path

# Resolve the supervised artifact directory relative to the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
ml_dir = _REPO_ROOT / "backend" / "artifacts" / "supervised"

_ARTIFACTS_AVAILABLE = (
    (ml_dir / "final_xgb_regressor.pkl").exists()
    and (ml_dir / "final_xgb_classifier.pkl").exists()
    and (ml_dir / "preprocessor_reg.pkl").exists()
    and (ml_dir / "label_encoder.pkl").exists()
)

pytestmark = pytest.mark.skipif(
    not _ARTIFACTS_AVAILABLE,
    reason="Trained ML artifact files not present (expected in CI environment without model binaries)",
)


@pytest.fixture
def scenario_models():
    reg_model = joblib.load(ml_dir / "final_xgb_regressor.pkl")
    cls_model = joblib.load(ml_dir / "final_xgb_classifier.pkl")
    prep_scaler = joblib.load(ml_dir / "preprocessor_reg.pkl")
    lbl_encoder = joblib.load(ml_dir / "label_encoder.pkl")

    wt_adapter = WaitingTimeModelAdapter(model_artifact=reg_model, preprocessor=prep_scaler)
    wt_adapter.preprocessor_pipeline = prep_scaler

    cr_adapter = CrowdingModelAdapter(model_artifact=cls_model, preprocessor=prep_scaler)
    cr_adapter.preprocessor_pipeline = prep_scaler
    cr_adapter.label_encoder = lbl_encoder

    return wt_adapter, cr_adapter


def test_scenario_quiet_shift_low_demand(scenario_models):
    """1. Low-demand scenario: Quiet shift inputs (10 pts/hr, 5 waiting, 25% occ)."""
    wt_adapter, cr_adapter = scenario_models
    quiet_inp = PredictionInputData(
        features={
            "arrival_rate": 10.0,
            "patients_waiting": 5,
            "occupancy_percent": 25.0,
            "available_beds": 15,
            "available_doctors": 8,
            "available_nurses": 12,
            "severity_level": 2.0,
            "hour_of_day": 3,
        }
    )

    wt_res = wt_adapter.predict_waiting_time(quiet_inp)
    cr_res = cr_adapter.predict_crowding(quiet_inp)

    assert wt_res.is_available is True
    assert cr_res.is_available is True
    assert isinstance(wt_res.prediction["estimated_wait_minutes"], float)
    assert isinstance(cr_res.prediction["crowding_level"], str)


def test_scenario_busy_shift_moderate_demand(scenario_models):
    """2. Moderate scenario: Busy evening inputs (28 pts/hr, 25 waiting, 78% occ)."""
    wt_adapter, cr_adapter = scenario_models
    busy_inp = PredictionInputData(
        features={
            "arrival_rate": 28.0,
            "patients_waiting": 25,
            "occupancy_percent": 78.0,
            "available_beds": 8,
            "available_doctors": 5,
            "available_nurses": 9,
            "severity_level": 3.0,
            "hour_of_day": 18,
        }
    )

    wt_res = wt_adapter.predict_waiting_time(busy_inp)
    cr_res = cr_adapter.predict_crowding(busy_inp)

    assert wt_res.is_available is True
    assert cr_res.is_available is True


def test_scenario_surge_high_demand(scenario_models):
    """3. High-demand / surge scenario (52 pts/hr, 58 waiting, 96% occ)."""
    wt_adapter, cr_adapter = scenario_models
    surge_inp = PredictionInputData(
        features={
            "arrival_rate": 52.0,
            "patients_waiting": 58,
            "occupancy_percent": 96.0,
            "available_beds": 2,
            "available_doctors": 3,
            "available_nurses": 5,
            "severity_level": 4.2,
            "hour_of_day": 20,
        }
    )

    wt_res = wt_adapter.predict_waiting_time(surge_inp)
    cr_res = cr_adapter.predict_crowding(surge_inp)

    assert wt_res.is_available is True
    assert cr_res.is_available is True


def test_scenario_extreme_but_valid(scenario_models):
    """4. Extreme valid scenario (80 pts/hr, 100 waiting, 100% occ)."""
    wt_adapter, cr_adapter = scenario_models
    extreme_inp = PredictionInputData(
        features={
            "arrival_rate": 80.0,
            "patients_waiting": 100,
            "occupancy_percent": 100.0,
            "available_beds": 0,
            "available_doctors": 2,
            "available_nurses": 4,
            "severity_level": 5.0,
            "hour_of_day": 21,
        }
    )

    wt_res = wt_adapter.predict_waiting_time(extreme_inp)
    cr_res = cr_adapter.predict_crowding(extreme_inp)

    assert wt_res.is_available is True
    assert cr_res.is_available is True


def test_scenario_invalid_inputs_handled_safely(scenario_models):
    """5. Invalid inputs (negative numbers or zero staffing) handled safely without crashing."""
    wt_adapter, cr_adapter = scenario_models
    invalid_inp = PredictionInputData(
        features={
            "arrival_rate": -10.0,
            "patients_waiting": -5,
            "occupancy_percent": -20.0,
            "available_beds": 0,
            "available_doctors": 0,
            "available_nurses": 0,
            "severity_level": -1.0,
            "hour_of_day": 99,
        }
    )

    wt_res = wt_adapter.predict_waiting_time(invalid_inp)
    cr_res = cr_adapter.predict_crowding(invalid_inp)

    assert wt_res.is_available is True
    assert cr_res.is_available is True
