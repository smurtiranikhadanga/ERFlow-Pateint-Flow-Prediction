import xgboost as xgb
from typing import Dict, List, Any, Optional

RAW_FEATURE_DISPLAY_NAMES = {
    "patients_waiting": "Patients Waiting",
    "arrival_rate": "Arrival Rate",
    "occupancy_percent": "Occupancy Percent",
    "available_beds": "Available Beds",
    "available_doctors": "Available Doctors",
    "available_nurses": "Available Nurses",
    "patients_per_staff": "Patients Per Staff",
    "patients_per_bed": "Patients Per Bed",
    "severity_level": "Severity Level",
    "hour_of_day": "Hour Of Day",
    "day_of_week": "Day Of Week",
    "month": "Month",
    "staff_total": "Staff Total",
    "is_weekend": "Is Weekend",
    "season": "Season",
    "time_period": "Time Period",
}


def get_feature_names_from_preprocessor(preprocessor: Any) -> List[str]:
    """Extracts ordered feature names from ColumnTransformer preprocessor."""
    if hasattr(preprocessor, "transformers_"):
        output_features = []
        for name, pipe, features in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(pipe, "get_feature_names_out"):
                try:
                    names = pipe.get_feature_names_out(features)
                except Exception:
                    names = features
            else:
                names = features
            for f in names:
                clean_f = f.split("__")[-1] if "__" in str(f) else str(f)
                output_features.append(clean_f)
        return output_features

    return list(RAW_FEATURE_DISPLAY_NAMES.keys())


def explain_prediction(
    model: Any,
    X_scaled: Any,
    feature_names: List[str],
    top_k: int = 4,
    class_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes exact TreeSHAP feature contributions for XGBoost models.
    """
    if model is None or X_scaled is None:
        return {"top_factors": []}

    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dmat = xgb.DMatrix(X_scaled)
        contribs = booster.predict(dmat, pred_contribs=True)

        if contribs.ndim == 3:  # Multi-class classifier (1, num_classes, num_features + 1)
            idx = class_idx if class_idx is not None else 0
            raw_vals = contribs[0, idx, :-1]
        else:
            raw_vals = contribs[0, :-1]

        feat_map: Dict[str, float] = {}
        for i, fname in enumerate(feature_names):
            if i >= len(raw_vals):
                break
            val = float(raw_vals[i])
            base_fname = fname
            for raw_k in RAW_FEATURE_DISPLAY_NAMES:
                if fname.startswith(raw_k):
                    base_fname = raw_k
                    break
            display_name = RAW_FEATURE_DISPLAY_NAMES.get(base_fname, base_fname.replace("_", " ").title())
            feat_map[display_name] = feat_map.get(display_name, 0.0) + val

        sorted_features = sorted(feat_map.items(), key=lambda item: abs(item[1]), reverse=True)
        total_abs = sum(abs(v) for _, v in sorted_features) + 1e-6

        factors = []
        for fname, val in sorted_features[:top_k]:
            rel_imp = round(float(abs(val) / total_abs), 2)
            direction = "increases" if val >= 0 else "decreases"
            factors.append({
                "feature": fname,
                "direction": direction,
                "importance": rel_imp,
                "shap_value": float(round(val, 4)),
            })

        return {"top_factors": factors}
    except Exception as e:
        return {"top_factors": [], "error": str(e)}
