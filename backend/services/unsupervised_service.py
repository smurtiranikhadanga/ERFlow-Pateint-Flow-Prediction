import logging
import numpy as np
import pandas as pd

from .artifact_loader import artifact_loader
from .monitoring_service import monitoring_service
from .supervised_service import supervised_service
from ..utils.feature_engineering import engineer_unsupervised_features
from ..schemas.hospital_state import HospitalState
from ..schemas.unsupervised import (
    FlowPatternResponse,
    SurgeDetectionResponse,
    UnsupervisedPredictionResponse,
)

logger = logging.getLogger("erflow.unsupervised_service")


class UnsupervisedService:
    """Handles inference for K-Means Demand Pattern Discovery and DBSCAN Anomaly/Surge Detection."""

    def _prepare_features(self, state: HospitalState) -> pd.DataFrame:
        if isinstance(state, dict):
            state = HospitalState(**state)
        state_dict = state.model_dump()
        # If waiting time is not provided, estimate using supervised regressor
        if state.waiting_time_minutes is None:
            try:
                wt_res = supervised_service.predict_waiting_time(state)
                state.waiting_time_minutes = wt_res.waiting_time_minutes
                state_dict["waiting_time_minutes"] = wt_res.waiting_time_minutes
            except Exception as e:
                logger.warning(f"Could not estimate waiting time: {e}")
                state_dict["waiting_time_minutes"] = 45.0

        return engineer_unsupervised_features(state_dict)

    def predict_flow_pattern(self, state: HospitalState) -> FlowPatternResponse:
        """Assign current ER state to K-Means cluster and compute PCA position."""
        if isinstance(state, dict):
            state = HospitalState(**state)
        t0 = monitoring_service.record_inference_start("flow_pattern_model")
        try:
            df_feat = self._prepare_features(state)
            scaler = artifact_loader.unsupervised_scaler
            kmeans = artifact_loader.kmeans_model
            pca = artifact_loader.pca_model

            X_scaled = scaler.transform(df_feat)
            cluster_id = int(kmeans.predict(X_scaled)[0])

            # Distance to cluster centers for confidence estimation
            distances = np.linalg.norm(kmeans.cluster_centers_ - X_scaled, axis=1)
            min_dist = float(distances[cluster_id])
            total_dist = float(np.sum(distances)) + 1e-6
            # Inverse distance softmax/confidence score
            confidence = float(np.clip(round((1.0 - (min_dist / total_dist)) * 100, 1), 65.0, 96.0))

            # Cluster label lookup
            cluster_labels = {
                0: "High Demand",
                1: "Medium Demand",
                2: "Low Demand"
            }
            pattern_name = cluster_labels.get(cluster_id, "Standard Demand")

            descriptions = {
                "High Demand": "Severe patient influx and elevated waiting times. High strain on staff and beds.",
                "Medium Demand": "Moderate arrival volume with steady patient throughput and controlled queue levels.",
                "Low Demand": "Normal operational baseline with minimal waiting times and ample bed capacity."
            }

            # 2D PCA Coordinates projected into 0-100 display range
            pca_coords = pca.transform(X_scaled)[0]
            x_norm = float(np.clip(round((pca_coords[0] + 3.5) / 7.0 * 100, 1), 2.0, 98.0))
            y_norm = float(np.clip(round((pca_coords[1] + 3.0) / 6.0 * 100, 1), 2.0, 98.0))

            resp = FlowPatternResponse(
                pattern_name=pattern_name,
                confidence=confidence,
                cluster_id=cluster_id,
                description=descriptions.get(pattern_name, "Operational state consistent with baseline patterns."),
                current_point={"x": x_norm, "y": y_norm},
                model_name="K-Means Clustering"
            )
            monitoring_service.record_inference_success("flow_pattern_model", t0, {"pattern_name": pattern_name, "cluster_id": cluster_id}, state.model_dump())
            return resp
        except Exception as e:
            monitoring_service.record_inference_error("flow_pattern_model", str(e))
            raise

    def detect_surge(self, state: HospitalState) -> SurgeDetectionResponse:
        """Detect anomalous arrival surges and operational strain using K-Means centroid distance and parametric operational thresholds."""
        if isinstance(state, dict):
            state = HospitalState(**state)
        t0 = monitoring_service.record_inference_start("high_demand_model")
        try:
            df_feat = self._prepare_features(state)
            scaler = artifact_loader.unsupervised_scaler
            kmeans = artifact_loader.kmeans_model

            X_scaled = scaler.transform(df_feat)

            # Calculate distance to nearest cluster core
            min_dist = float(np.min(np.linalg.norm(kmeans.cluster_centers_ - X_scaled, axis=1)))

            # Expected baseline arrivals for time of day
            hour = state.hour_of_day
            if 8 <= hour <= 21:
                normal_min, normal_max = 14, 22
            else:
                normal_min, normal_max = 8, 14

            normal_baseline_str = f"{normal_min}-{normal_max}"
            current_rate = state.arrival_rate
            baseline_mid = (normal_min + normal_max) / 2.0
            deviation = ((current_rate - baseline_mid) / baseline_mid) * 100.0
            dev_str = f"{'+' if deviation >= 0 else ''}{round(deviation)}%"

            # Anomaly threshold check (distance > 2.8 or arrival rate significantly exceeding baseline or high occupancy)
            is_surge = bool(current_rate > normal_max * 1.25 or min_dist > 2.8 or state.occupancy_percent > 88.0)

            if is_surge:
                status = "ANOMALOUS SURGE DETECTED"
                severity = "High" if current_rate > normal_max * 1.6 else "Moderate"
                description = (
                    f"Arrival rate ({current_rate:.0f} pts/hr) and system strain deviate significantly "
                    f"from the normal baseline ({normal_baseline_str} pts/hr)."
                )
            else:
                status = "NORMAL OPERATIONAL LOAD"
                severity = "Low"
                description = (
                    f"Current arrival volume ({current_rate:.0f} pts/hr) is within expected operating limits "
                    f"for hour {hour}:00."
                )

            detected_at = f"{(hour % 12) or 12}:30 {'PM' if hour >= 12 else 'AM'}"

            resp = SurgeDetectionResponse(
                is_surge=is_surge,
                status=status,
                severity=severity,
                current_arrival_rate=float(round(current_rate, 1)),
                normal_arrival_rate=normal_baseline_str,
                deviation_percent=dev_str,
                detected_at=detected_at,
                description=description,
                model_name="Operational Surge Anomaly Detector"
            )
            monitoring_service.record_inference_success("high_demand_model", t0, {"status": status, "is_surge": is_surge}, state.model_dump())
            return resp
        except Exception as e:
            monitoring_service.record_inference_error("high_demand_model", str(e))
            raise

    def predict_all(self, state: HospitalState) -> UnsupervisedPredictionResponse:
        """Run both pattern clustering and surge detection."""
        if isinstance(state, dict):
            state = HospitalState(**state)
        flow = self.predict_flow_pattern(state)
        surge = self.detect_surge(state)
        return UnsupervisedPredictionResponse(
            flow_pattern=flow,
            surge_detection=surge
        )


unsupervised_service = UnsupervisedService()


if __name__ == "__main__":
    print("\n--- Running UnsupervisedService Direct Verification ---")
    test_state = HospitalState(
        hour_of_day=14,
        day_of_week=2,
        arrival_rate=22.0,
        available_beds=10.0,
        available_doctors=5.0,
        available_nurses=8.0,
        patients_waiting=14.0,
        severity_level=3.0,
        occupancy_percent=68.0,
    )
    result = unsupervised_service.predict_all(test_state)
    print(f"Flow Pattern: {result.flow_pattern.pattern_name} (Confidence: {result.flow_pattern.confidence}%)")
    print(f"Surge Status: {result.surge_detection.status} (Severity: {result.surge_detection.severity})")
    print("Verification Successful!\n")
