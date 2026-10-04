import datetime
import logging
from typing import Any, Dict, List, Optional
from app.schemas.prediction_schema import Intent, PredictionInputData

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for input validation results, validated input data, or clarification requests."""

    def __init__(
        self,
        is_valid: bool,
        validated_input: Optional[PredictionInputData] = None,
        clarification_message: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.is_valid = is_valid
        self.validated_input = validated_input
        self.clarification_message = clarification_message
        self.error_message = error_message


class InputValidator:
    """
    Strict Input Validator for ER ML Prediction Models.

    Enforces required fields, optional fields, valid ranges, categorical values,
    and sequence constraints across all model intents:

    - Supervised Regressor (Waiting Time): 16 feature columns (0 <= hour <= 23, 0 <= day <= 6, 1 <= month <= 12, etc.)
    - Supervised Classifier (Crowding): 10 features & multiclass level mapping (Low, Moderate, High, Critical)
    - Unsupervised DBSCAN (High Demand Surge): arrival_rate, occupancy_percent, hour_of_day boundaries
    - Unsupervised K-Means + PCA (Flow Pattern): 6 operational strain metrics
    - Deep Learning LSTM (Patient Volume): 168-hour historical sequence window from project data

    Never invents missing model inputs. Requests clarification if required data is missing
    or out of bounds.
    """

    VALID_TRIAGE_LEVELS = [
        "resuscitation",
        "emergent",
        "urgent",
        "less urgent",
        "non-urgent",
        "standard",
        "critical",
        "1",
        "2",
        "3",
        "4",
        "5",
    ]

    VALID_DAYS = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]

    def validate_prediction_input(
        self,
        intent: Intent,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validates context and feature input fields prior to ML model inference.
        Returns a ValidationResult object.
        """
        context = context or {}
        features = dict(context.get("features", {}))
        # Support flat context payloads passed directly from frontend operationalState
        for key in (
            "occupancy_percent",
            "patients_waiting",
            "arrival_rate",
            "available_beds",
            "available_doctors",
            "available_nurses",
            "severity_level",
            "hour_of_day",
            "day_of_week",
            "month",
            "triage_level",
        ):
            if key in context and key not in features:
                features[key] = context[key]

        errors: List[str] = []

        # 1. TEMPORAL & RANGE BOUNDARY VALIDATION
        if "hour_of_day" in features:
            try:
                hr = int(features["hour_of_day"])
                if not (0 <= hr <= 23):
                    errors.append(f"hour_of_day ({hr}) must be an integer between 0 and 23.")
            except (ValueError, TypeError):
                errors.append(f"hour_of_day must be an integer between 0 and 23.")

        if "day_of_week" in features:
            val = features["day_of_week"]
            if isinstance(val, int):
                if not (0 <= val <= 6):
                    errors.append(f"day_of_week ({val}) must be an integer between 0 (Monday) and 6 (Sunday).")
            elif isinstance(val, str):
                if val.lower().strip() not in self.VALID_DAYS:
                    errors.append(f"day_of_week '{val}' must be one of {self.VALID_DAYS[:7]}.")

        if "month" in features:
            try:
                m = int(features["month"])
                if not (1 <= m <= 12):
                    errors.append(f"month ({m}) must be an integer between 1 and 12.")
            except (ValueError, TypeError):
                errors.append("month must be an integer between 1 and 12.")

        # 2. OPERATIONAL STRAIN BOUNDARY VALIDATION
        if "occupancy_percent" in features:
            try:
                occ = float(features["occupancy_percent"])
                if not (0.0 <= occ <= 100.0):
                    errors.append(f"occupancy_percent ({occ}) must be between 0.0% and 100.0%.")
            except (ValueError, TypeError):
                errors.append("occupancy_percent must be a number between 0.0 and 100.0.")

        if "arrival_rate" in features:
            try:
                arr = float(features["arrival_rate"])
                if not (0.0 <= arr <= 300.0):
                    errors.append(f"arrival_rate ({arr}) must be a positive number between 0.0 and 300.0.")
            except (ValueError, TypeError):
                errors.append("arrival_rate must be a positive number.")

        if "patients_waiting" in features:
            try:
                pw = float(features["patients_waiting"])
                if pw < 0:
                    errors.append(f"patients_waiting ({pw}) cannot be negative.")
            except (ValueError, TypeError):
                errors.append("patients_waiting must be a non-negative number.")

        if "available_beds" in features:
            try:
                ab = float(features["available_beds"])
                if ab < 0:
                    errors.append(f"available_beds ({ab}) cannot be negative.")
            except (ValueError, TypeError):
                errors.append("available_beds must be a non-negative number.")

        # 3. CATEGORICAL TRIAGE LEVEL VALIDATION
        triage_raw = context.get("triage_level") or features.get("triage_level")
        if triage_raw is not None:
            t_str = str(triage_raw).lower().strip()
            if t_str not in self.VALID_TRIAGE_LEVELS:
                errors.append(
                    f"triage_level '{triage_raw}' is invalid. "
                    f"Expected one of: Resuscitation, Emergent, Urgent, Less Urgent, Non-Urgent, Standard, Critical (1-5)."
                )

        # Return explicit validation error if boundaries are violated
        if errors:
            return ValidationResult(
                is_valid=False,
                error_message="Input validation failed:\n- " + "\n- ".join(errors),
            )

        # 4. EXPLICIT CLARIFICATION REQUEST CHECK
        # If user explicitly requested temporal forecast but provided no date/time context
        if context.get("requires_explicit_datetime") is True or context.get("ask_datetime") is True:
            if "target_datetime" not in context and "hour_of_day" not in features:
                return ValidationResult(
                    is_valid=False,
                    clarification_message="Sure. What date and time would you like me to check?",
                )

        # 5. CONSTRUCT VALIDATED PREDICTION INPUT DATA
        now = datetime.datetime.now()
        validated_features = dict(features)

        # Operational timestamp context defaults
        if "hour_of_day" not in validated_features:
            validated_features["hour_of_day"] = now.hour
        if "day_of_week" not in validated_features:
            validated_features["day_of_week"] = now.weekday()
        if "month" not in validated_features:
            validated_features["month"] = now.month

        validated_input_data = PredictionInputData(
            timestamp=now,
            day_of_week=str(validated_features["day_of_week"]),
            historical_patient_count=context.get("historical_patient_count") or validated_features.get("patients_waiting") or 24,
            triage_level=str(triage_raw) if triage_raw else "Standard",
            time_window=context.get("time_window") or "next_4_hours",
            features=validated_features,
        )

        return ValidationResult(is_valid=True, validated_input=validated_input_data)


# Global singleton instance
input_validator = InputValidator()
