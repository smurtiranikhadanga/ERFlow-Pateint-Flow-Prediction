import pytest
from app.chatbot.chatbot_service import chatbot_service
from app.chatbot.intent_detector import intent_detector
from app.schemas.chat_schema import ChatRequest
from app.schemas.prediction_schema import Intent
from app.ml_service.model_registry import model_registry
from app.ml_service.model_adapters import (
    WaitingTimeModelAdapter,
    CrowdingModelAdapter,
    PatientVolumeModelAdapter,
    FlowPatternModelAdapter,
    HighDemandModelAdapter,
)
import joblib
from pathlib import Path

# Resolve artifact directories relative to the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
ml_dir = _REPO_ROOT / "backend" / "artifacts" / "supervised"
ml_unsup_dir = _REPO_ROOT / "backend" / "artifacts" / "unsupervised"
ml_dl_dir = _REPO_ROOT / "backend" / "artifacts" / "deep_learning"

_SUPERVISED_AVAILABLE = (
    (ml_dir / "final_xgb_regressor.pkl").exists()
    and (ml_dir / "final_xgb_classifier.pkl").exists()
    and (ml_dir / "preprocessor_reg.pkl").exists()
    and (ml_dir / "label_encoder.pkl").exists()
)
_UNSUPERVISED_AVAILABLE = (
    (ml_unsup_dir / "kmeans_model.joblib").exists()
    and (ml_unsup_dir / "pca_model.joblib").exists()
    and (ml_unsup_dir / "unsupervised_scaler.joblib").exists()
)
_ALL_ARTIFACTS_AVAILABLE = _SUPERVISED_AVAILABLE and _UNSUPERVISED_AVAILABLE


@pytest.fixture(autouse=True)
def setup_models():
    if not _ALL_ARTIFACTS_AVAILABLE:
        pytest.skip("Trained ML artifact files not present (expected in CI environment without model binaries)")

    reg_model = joblib.load(ml_dir / "final_xgb_regressor.pkl")
    cls_model = joblib.load(ml_dir / "final_xgb_classifier.pkl")
    prep_scaler = joblib.load(ml_dir / "preprocessor_reg.pkl")
    lbl_encoder = joblib.load(ml_dir / "label_encoder.pkl")
    km_model = joblib.load(ml_unsup_dir / "kmeans_model.joblib")
    pca_model = joblib.load(ml_unsup_dir / "pca_model.joblib")
    scaler_unsup = joblib.load(ml_unsup_dir / "unsupervised_scaler.joblib")

    wt_adapter = WaitingTimeModelAdapter(model_artifact=reg_model, preprocessor=prep_scaler)
    wt_adapter.preprocessor_pipeline = prep_scaler

    cr_adapter = CrowdingModelAdapter(model_artifact=cls_model, preprocessor=prep_scaler)
    cr_adapter.preprocessor_pipeline = prep_scaler
    cr_adapter.label_encoder = lbl_encoder

    pv_adapter = PatientVolumeModelAdapter(model_artifact=None)
    fp_adapter = FlowPatternModelAdapter(model_artifact=km_model, pca_artifact=pca_model, scaler_artifact=scaler_unsup)
    hd_adapter = HighDemandModelAdapter(model_artifact=km_model, scaler_artifact=scaler_unsup)

    model_registry.register_model("waiting_time_model", wt_adapter)
    model_registry.register_model("crowding_model", cr_adapter)
    model_registry.register_model("patient_volume_model", pv_adapter)
    model_registry.register_model("flow_pattern_model", fp_adapter)
    model_registry.register_model("high_demand_model", hd_adapter)


@pytest.fixture
def chat_service():
    return chatbot_service


def test_supported_intents_and_paraphrases(chat_service):
    default_ctx = {
        "features": {
            "arrival_rate": 28.0,
            "patients_waiting": 25,
            "occupancy_percent": 78.0,
            "available_beds": 8,
            "available_doctors": 5,
            "available_nurses": 9,
            "severity_level": 3.0,
            "hour_of_day": 18,
        }
    }

    queries_and_intents = [
        ("What is the current waiting time?", Intent.WAITING_TIME.value),
        ("Why is the wait so high?", Intent.WAITING_TIME.value),
        ("how long are patients waiting", Intent.WAITING_TIME.value),
        ("waiting-time risk", Intent.WAITING_TIME.value),
        ("queue getting longer", Intent.WAITING_TIME.value),
        ("Will the ER be crowded?", Intent.CROWDING.value),
        ("Why is crowding high?", Intent.CROWDING.value),
        ("how busy is the emergency room", Intent.CROWDING.value),
        ("How many patients are expected?", Intent.PATIENT_VOLUME.value),
        ("Is demand increasing?", Intent.PATIENT_VOLUME.value),
        ("What flow pattern are we seeing?", Intent.FLOW_PATTERN.value),
        ("Is there a surge?", Intent.HIGH_DEMAND_PERIOD.value),
        ("How is the ER doing right now?", Intent.GENERAL_STATUS.value),
        ("Give me a summary of the current ER.", Intent.GENERAL_STATUS.value),
        ("What is causing the current pressure?", Intent.GENERAL_STATUS.value),
        ("How busy is the ER?", Intent.GENERAL_STATUS.value),
        ("help", Intent.HELP.value),
        ("what can you do", Intent.HELP.value),
    ]

    for query, expected_intent in queries_and_intents:
        detection = intent_detector.detect_intent(query)
        assert detection["intent"] == expected_intent, f"Query '{query}' expected {expected_intent}, got {detection['intent']}"

        req = ChatRequest(message=query, session_id="test_paraphrases_session", context=default_ctx)
        resp = chat_service.process_message(req)
        assert resp.intent == expected_intent
        assert resp.response is not None
        assert len(resp.response) > 0


def test_conversational_followup_memory(chat_service):
    session_id = "test_followup_convo_123"
    ctx = {
        "features": {
            "arrival_rate": 28.0,
            "patients_waiting": 25,
            "occupancy_percent": 78.0,
            "available_beds": 8,
            "available_doctors": 5,
            "available_nurses": 9,
            "severity_level": 3.0,
            "hour_of_day": 18,
        }
    }

    # Turn 1: Initial query
    t1_req = ChatRequest(message="What is the current waiting time?", session_id=session_id, context=ctx)
    t1_resp = chat_service.process_message(t1_req)
    assert t1_resp.intent == Intent.WAITING_TIME.value
    assert "minutes" in t1_resp.response.lower()

    # Turn 2: Follow-up "Why?"
    t2_req = ChatRequest(message="Why?", session_id=session_id, context=ctx)
    t2_resp = chat_service.process_message(t2_req)
    assert t2_resp.intent == Intent.WAITING_TIME.value
    assert "factors" in t2_resp.response.lower() or "contributing" in t2_resp.response.lower() or "driving" in t2_resp.response.lower()

    # Turn 3: Follow-up "What about later?"
    t3_req = ChatRequest(message="What about later?", session_id=session_id, context=ctx)
    t3_resp = chat_service.process_message(t3_req)
    assert t3_resp.intent == Intent.WAITING_TIME.value
    assert "forecasted" in t3_resp.response.lower() or "reach" in t3_resp.response.lower() or "peak" in t3_resp.response.lower() or "minutes" in t3_resp.response.lower()


def test_unknown_queries(chat_service):
    unknown_queries = [
        "What is the capital of France?",
        "Who won the Super Bowl last year?",
        "Tell me a joke about cats",
    ]
    for query in unknown_queries:
        req = ChatRequest(message=query, session_id="test_unknown_session")
        resp = chat_service.process_message(req)
        assert resp.intent == Intent.UNKNOWN.value or "sorry" in resp.response.lower()


def test_intent_matrix_output(chat_service):
    ctx = {
        "features": {
            "arrival_rate": 28.0,
            "patients_waiting": 25,
            "occupancy_percent": 78.0,
            "available_beds": 8,
            "available_doctors": 5,
            "available_nurses": 9,
            "severity_level": 3.0,
            "hour_of_day": 18,
        }
    }

    test_cases = [
        ("What is the current waiting time?", Intent.WAITING_TIME.value, "Supervised XGBoost Regressor"),
        ("Why is the wait so high?", Intent.WAITING_TIME.value, "Supervised XGBoost Regressor"),
        ("Will the ER be crowded?", Intent.CROWDING.value, "Supervised XGBoost Classifier"),
        ("How many patients are expected?", Intent.PATIENT_VOLUME.value, "Deep Learning LSTM"),
        ("What flow pattern are we seeing?", Intent.FLOW_PATTERN.value, "Unsupervised K-Means + PCA"),
        ("Is there a surge?", Intent.HIGH_DEMAND_PERIOD.value, "Operational Surge Anomaly Detector"),
        ("How is the ER doing right now?", Intent.GENERAL_STATUS.value, "Multi-Model Aggregator"),
    ]

    print("\n" + "=" * 115)
    print(f"{'QUESTION':<40} | {'EXPECTED':<20} | {'ACTUAL':<20} | {'MODEL/API CALLED':<30} | RESULT")
    print("-" * 115)

    for query, expected_intent, model_name in test_cases:
        req = ChatRequest(message=query, session_id="matrix_session", context=ctx)
        resp = chat_service.process_message(req)
        status = "PASS" if resp.intent == expected_intent else "FAIL"
        print(f"{query:<40} | {expected_intent:<20} | {resp.intent:<20} | {model_name:<30} | {status}")
        assert resp.intent == expected_intent
    print("=" * 115 + "\n")
