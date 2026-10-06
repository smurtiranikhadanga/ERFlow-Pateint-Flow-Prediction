import logging

from .artifact_loader import artifact_loader
from .monitoring_service import monitoring_service
from ..utils.feature_engineering import engineer_supervised_features
from ..schemas.hospital_state import HospitalState
from ..schemas.supervised import WaitingTimeResponse, CrowdingRiskResponse, SupervisedPredictionResponse
from .xai_explainer import explain_prediction, get_feature_names_from_preprocessor

logger = logging.getLogger("erflow.supervised_service")


class SupervisedService:
    """Handles inference for XGBoost Regressor (Waiting Time) and Classifier (Crowding Risk)."""

    def predict_waiting_time(self, state: HospitalState) -> WaitingTimeResponse:
        """Predict expected waiting time in minutes."""
        if isinstance(state, dict):
            state = HospitalState(**state)
        t0 = monitoring_service.record_inference_start("waiting_time_model")
        try:
            state_dict = state.model_dump()
            df_features = engineer_supervised_features(state_dict)

            preprocessor = artifact_loader.supervised_preprocessor
            regressor = artifact_loader.xgb_regressor
            TARGET_MEAN_OFFSET = 43.35

            X_trans = preprocessor.transform(df_features)
            pred_raw = float(regressor.predict(X_trans)[0])
            pred_wait = max(1.0, round(pred_raw + TARGET_MEAN_OFFSET, 1))

            # TreeSHAP feature explanation
            feature_names = get_feature_names_from_preprocessor(preprocessor)
            explanation = explain_prediction(regressor, X_trans, feature_names, top_k=4)

            # Project 1h ahead (slight arrival & queue progression)
            state_1h = state_dict.copy()
            state_1h["hour_of_day"] = (state_1h["hour_of_day"] + 1) % 24
            state_1h["patients_waiting"] = max(1.0, state_1h["patients_waiting"] * 1.1)
            df_1h = engineer_supervised_features(state_1h)
            pred_1h_raw = float(regressor.predict(preprocessor.transform(df_1h))[0])
            pred_1h = max(1.0, round(pred_1h_raw + TARGET_MEAN_OFFSET, 1))

            # Projected peak (evening hour simulation)
            state_peak = state_dict.copy()
            state_peak["hour_of_day"] = 19
            state_peak["patients_waiting"] = max(state_peak["patients_waiting"], 35.0)
            state_peak["occupancy_percent"] = max(state_peak["occupancy_percent"], 85.0)
            df_peak = engineer_supervised_features(state_peak)
            pred_peak_raw = float(regressor.predict(preprocessor.transform(df_peak))[0])
            pred_peak = max(pred_wait, round(pred_peak_raw + TARGET_MEAN_OFFSET, 1))

            trend = "Increasing" if pred_1h > pred_wait else ("Decreasing" if pred_1h < pred_wait else "Stable")

            resp = WaitingTimeResponse(
                waiting_time_minutes=pred_wait,
                predicted_1h=pred_1h,
                predicted_peak=pred_peak,
                trend=trend,
                model_name="XGBoost Regressor",
                explanation=explanation,
            )
            monitoring_service.record_inference_success("waiting_time_model", t0, {"waiting_time_minutes": pred_wait}, state_dict)
            return resp
        except Exception as e:
            monitoring_service.record_inference_error("waiting_time_model", str(e))
            raise

    def predict_crowding_risk(self, state: HospitalState) -> CrowdingRiskResponse:
        """Predict multi-class crowding level and probability distribution."""
        if isinstance(state, dict):
            state = HospitalState(**state)
        t0 = monitoring_service.record_inference_start("crowding_model")
        try:
            state_dict = state.model_dump()
            df_features = engineer_supervised_features(state_dict)

            preprocessor = artifact_loader.supervised_preprocessor
            classifier = artifact_loader.xgb_classifier
            label_encoder = artifact_loader.label_encoder

            X_trans = preprocessor.transform(df_features)
            pred_enc = int(classifier.predict(X_trans)[0])
            pred_label = str(label_encoder.inverse_transform([pred_enc])[0])

            probs_arr = classifier.predict_proba(X_trans)[0]
            classes = [str(c) for c in label_encoder.classes_]

            prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs_arr)}

            # TreeSHAP feature explanation for assigned class index
            feature_names = get_feature_names_from_preprocessor(preprocessor)
            explanation = explain_prediction(classifier, X_trans, feature_names, top_k=4, class_idx=pred_enc)

            # Derive 0-100 severity index score
            weight_map = {"Low": 20, "Moderate": 50, "High": 80, "Critical": 100}
            score = sum(prob_dict.get(k, 0.0) * weight_map.get(k, 50) for k in weight_map)
            score = int(min(100, max(0, round(score))))

            display_level = pred_label.upper()

            resp = CrowdingRiskResponse(
                crowding_level=display_level,
                crowding_score=score,
                probabilities=prob_dict,
                model_name="XGBoost Classifier",
                expected_window="6:00 PM - 9:00 PM" if state.hour_of_day >= 15 else "Next 3 Hours",
                explanation=explanation,
            )
            monitoring_service.record_inference_success("crowding_model", t0, {"crowding_level": display_level, "crowding_score": score}, state_dict)
            return resp
        except Exception as e:
            monitoring_service.record_inference_error("crowding_model", str(e))
            raise

    def predict_all(self, state: HospitalState) -> SupervisedPredictionResponse:
        """Run both supervised models concurrently."""
        if isinstance(state, dict):
            state = HospitalState(**state)
        wt = self.predict_waiting_time(state)
        cr = self.predict_crowding_risk(state)
        return SupervisedPredictionResponse(
            waiting_time=wt,
            crowding_risk=cr
        )


supervised_service = SupervisedService()
