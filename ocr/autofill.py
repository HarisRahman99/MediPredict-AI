"""
Map parsed OCR values to existing MediPredict form fields.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


# ==========================================================
# CKD FORM FIELD MAP
# ==========================================================

CKD_FIELD_MAP = {

    # Vital Signs
    "bp_systolic": "bp_systolic",
    "bp_diastolic": "bp_diastolic",

    # Kidney Function
    "serum_creatinine": "serum_creatinine",
    "blood_urea_nitrogen": "blood_urea_nitrogen",
    "albumin_serum": "albumin_serum",
    "albumin": "albumin_serum",

    "phosphorus": "phosphorus",
    "calcium": "calcium",
    "uric_acid": "uric_acid",

    "urine_albumin": "urine_albumin",
    "urine_creatinine": "urine_creatinine",
    "albumin_creatinine_ratio": "albumin_creatinine_ratio",

    "bicarbonate": "bicarbonate",

    # Patient
    "age": "age",
    "height_cm": "height_cm",
    "weight_kg": "weight_kg",
    "bmi": "bmi",
    "poverty_income_ratio": "poverty_income_ratio",

    # Lifestyle
    "diabetes_diagnosed": "diabetes_diagnosed",
    "insulin_use": "insulin_use",
    "diabetes_pills": "diabetes_pills",
    "ever_smoked": "ever_smoked",
    "current_smoker": "current_smoker",

    # Text
    "ethnicity": "ethnicity",
    "education_level": "education_level",
    "gender": "gender",
}


# ==========================================================
# HEART FORM FIELD MAP
# ==========================================================

HEART_FIELD_MAP = {

    # Demographics
    "age": "age",
    "gender": "male",

    # Lifestyle
    "current_smoker": "currentSmoker",
    "diabetes_diagnosed": "diabetes",

    # Vitals
    "bp_systolic": "sysBP",
    "bp_diastolic": "diaBP",
    "heart_rate": "heartRate",
    "bmi": "BMI",

    # Blood Tests
    "cholesterol": "totChol",
    "glucose": "glucose",
}


# ==========================================================
# VALUE CONVERTERS
# ==========================================================

def _convert_gender(value):

    if not isinstance(value, str):
        return value

    gender = value.strip().lower()

    if gender == "male":
        return 1

    if gender == "female":
        return 0

    return None


def _convert_boolean(value):

    if isinstance(value, (int, float)):
        return int(value)

    if not isinstance(value, str):
        return value

    value = value.strip().lower()

    if value in ("yes", "true", "positive"):
        return 1

    if value in ("no", "false", "negative"):
        return 0

    return value


# ==========================================================
# MAIN MAPPER
# ==========================================================

def values_for_target(
    parsed_values: Mapping[str, Any],
    target: str,
) -> Dict[str, Any]:

    if target == "ckd":
        field_map = CKD_FIELD_MAP

    elif target == "heart":
        field_map = HEART_FIELD_MAP

    else:
        return {}

    mapped: Dict[str, Any] = {}

    for parsed_key, form_field in field_map.items():

        if parsed_key not in parsed_values:
            continue

        value = parsed_values[parsed_key]

        # -------------------------
        # Gender Conversion
        # -------------------------

        if parsed_key == "gender":

            converted = _convert_gender(value)

            if converted is None:

                # CKD form expects text
                if target == "ckd":
                    mapped[form_field] = value.lower()

                continue

            value = converted

        # -------------------------
        # Boolean Conversion
        # -------------------------

        if parsed_key in (
            "diabetes_diagnosed",
            "insulin_use",
            "diabetes_pills",
            "ever_smoked",
            "current_smoker",
        ):

            value = _convert_boolean(value)

        mapped[form_field] = value

    return mapped