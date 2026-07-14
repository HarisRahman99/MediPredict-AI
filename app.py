"""
MediPredict AI — Flask Application
====================================
Disease Risk Prediction: Chronic Kidney Disease & Heart Disease

Run locally:
    python app.py

Environment variables (all optional):
    FLASK_DEBUG   — 'true' to enable debug mode   (default: false)
    SECRET_KEY    — Flask secret key               (default: dev key)
    HOST          — bind host                      (default: 0.0.0.0)
    PORT          — bind port                      (default: 5000)
"""

import os
import logging
import hashlib
from datetime import datetime

import numpy as np
import pickle
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from ocr.autofill import values_for_target
from ocr.extract import extract_text
from ocr.parser import parse_medical_values
from ocr.preprocess import OCRDependencyError, OCRProcessingError, is_allowed_file, preprocess_upload


# ══════════════════════════════════════════════════════════════
#  LOGGING  (replaces bare print() calls)
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medipredict")


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

class Config:
    DEBUG      = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    SECRET_KEY = os.environ.get("SECRET_KEY", "medipredict-dev-secret-change-in-prod")
    HOST       = os.environ.get("HOST", "0.0.0.0")
    PORT       = int(os.environ.get("PORT", 5000))
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024

    # Model file paths (override via env vars if needed)
    CKD_MODEL_PATH    = os.environ.get("CKD_MODEL_PATH",    "models/ckd_model.pkl")
    CKD_FEATURES_PATH = os.environ.get("CKD_FEATURES_PATH", "models/features.pkl")
    
    HEART_MODEL_PATH = os.environ.get("HEART_MODEL_PATH","models/heart/heart_model.pkl" )
    HEART_SCALER_PATH = os.environ.get("HEART_SCALER_PATH","models/heart/heart_scaler.pkl")
    HEART_FEATURES_PATH = os.environ.get("HEART_FEATURES_PATH","models/heart/heart_features.pkl")


    
# ══════════════════════════════════════════════════════════════
#  CONSTANTS  (no more scattered magic numbers)
# ══════════════════════════════════════════════════════════════

# Risk classification thresholds (%)
RISK_HIGH_THRESHOLD     = 80
RISK_MODERATE_THRESHOLD = 50

HEART_THRESHOLD = 0.50

# CKD stage upper bounds (%)  — Stage 5 is the remainder
STAGE_MAP = [
    (20, "Stage 1"),
    (40, "Stage 2"),
    (60, "Stage 3"),
    (80, "Stage 4"),
]

# Water intake table: stage -> (label, bar_fill_percent)
WATER_MAP = {
    "Stage 1": ("3.0 Litres / Day",                              100),
    "Stage 2": ("2.8 Litres / Day",                               90),
    "Stage 3": ("2.5 Litres / Day",                               75),
    "Stage 4": ("2.0 Litres / Day",                               60),
    "Stage 5": ("1.0 Litre / Day  (Under Doctor Supervision)",    40),
}

# Clinical recommendations
RECS_CKD = [
    "Reduce sodium intake to under 2,300 mg/day",
    "Avoid processed and packaged foods",
    "Monitor blood pressure daily",
    "Stay hydrated as advised by your doctor",
    "Consult a nephrologist promptly",
    "Limit dietary protein if recommended by your doctor",
    "Schedule regular eGFR and creatinine monitoring",
]

RECS_HEALTHY = [
    "Maintain a balanced, kidney-friendly diet",
    "Exercise at least 30 minutes, 5 days a week",
    "Drink 2–3 litres of water daily",
    "Have annual kidney function tests",
    "Avoid overuse of NSAIDs and nephrotoxic medications",
]

RECS_HEART_LOW = [
    "Maintain a balanced diet",
    "Exercise regularly",
    "Monitor blood pressure annually",
    "Continue healthy lifestyle habits"
]

RECS_HEART_MODERATE = [
    "Reduce sodium intake",
    "Monitor cholesterol levels",
    "Increase physical activity",
    "Maintain healthy body weight"
]

RECS_HEART_HIGH = [
    "Consult a cardiologist",
    "Quit smoking immediately",
    "Control blood pressure",
    "Reduce cholesterol intake",
    "Schedule regular heart checkups"
]

# ══════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config.from_object(Config)


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING  (graceful — won't crash the server on startup)
# ══════════════════════════════════════════════════════════════

def _load_pickle(path: str):
    """Load a pickle file safely; return None on failure."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log.error("File not found: %s", path)
    except Exception as exc:
        log.error("Failed to load %s — %s: %s", path, type(exc).__name__, exc)
    return None


ckd_model    = _load_pickle(Config.CKD_MODEL_PATH)
ckd_features = _load_pickle(Config.CKD_FEATURES_PATH)

heart_model = _load_pickle(Config.HEART_MODEL_PATH)
heart_scaler = _load_pickle(Config.HEART_SCALER_PATH)
heart_features = _load_pickle(Config.HEART_FEATURES_PATH)

if ckd_model and ckd_features:
    log.info("CKD model loaded — %d features", len(ckd_features))
else:
    log.warning("CKD model is NOT loaded. Prediction routes will return 503.")

if heart_model and heart_scaler and heart_features:
    log.info("Heart model loaded — %d features",len(heart_features))
else:
    log.warning("Heart model is NOT loaded.")

# ══════════════════════════════════════════════════════════════
#  BUSINESS LOGIC HELPERS  (single source of truth — no duplication)
# ══════════════════════════════════════════════════════════════

def classify_risk(risk: float) -> str:
    """Return a human-readable risk tier from a percentage score."""
    if risk >= RISK_HIGH_THRESHOLD:
        return "High Risk"
    if risk >= RISK_MODERATE_THRESHOLD:
        return "Moderate Risk"
    return "Low Risk"

def classify_heart_risk(risk: float) -> str:
    """
    Heart Disease risk categories.
    risk is percentage (0-100)
    """
    if risk >= 60:
        return "High Risk"
    elif risk >= 30:
        return "Moderate Risk"
    return "Low Risk"

def get_ckd_stage(risk: float) -> str:
    """Map a risk percentage to a CKD stage label."""
    for upper_bound, label in STAGE_MAP:
        if risk < upper_bound:
            return label
    return "Stage 5"


def run_ckd_inference(input_values: list) -> dict:
    """
    Run the CKD model on a flat list of feature values.

    Returns a result dictionary ready to pass to templates or the API.
    Raises RuntimeError if the model is not loaded.
    Raises ValueError if input_values contains non-numeric data.
    """
    if ckd_model is None or ckd_features is None:
        raise RuntimeError("CKD model is not loaded.")

    arr          = np.array([input_values], dtype=float)
    prediction   = int(ckd_model.predict(arr)[0])
    probabilities = ckd_model.predict_proba(arr)[0]

    # P(CKD=1) drives the risk score
    risk = round(float(probabilities[1]) * 100, 2)

    # Model confidence = how certain it is (regardless of class direction)
    # e.g. 92% sure "No CKD" is still confidence=92
    confidence = round(float(max(probabilities)) * 100, 1)

    ckd_stage               = get_ckd_stage(risk)
    water_intake, water_width = WATER_MAP.get(ckd_stage, ("2.0 Litres / Day", 60))

    return {
        "prediction":       prediction,
        "result":           "CKD Detected" if prediction == 1 else "No CKD Detected",
        "risk":             risk,
        "risk_level":       classify_risk(risk),
        "confidence":       confidence,
        "ckd_stage":        ckd_stage,
        "water_intake":     water_intake,
        "water_width":      water_width,
        "recommendations":  RECS_CKD if prediction == 1 else RECS_HEALTHY,
    }

def run_heart_inference(input_values: list) -> dict:
    """
    Run Heart Disease model inference.
    """

    if (
        heart_model is None
        or heart_scaler is None
        or heart_features is None
    ):
        raise RuntimeError(
            "Heart model is not loaded."
        )

    arr = np.array(
        [input_values],
        dtype=float
    )

    arr_scaled = heart_scaler.transform(
        arr
    )

    probability = float(
        heart_model.predict_proba(
            arr_scaled
        )[0][1]
    )

    prediction = (
        1
        if probability >= HEART_THRESHOLD
        else 0
    )

    risk = round(
        probability * 100,
        2
    )

    risk_level = classify_heart_risk(
        risk
    )

    if risk_level == "High Risk":
        recommendations = RECS_HEART_HIGH

    elif risk_level == "Moderate Risk":
        recommendations = RECS_HEART_MODERATE

    else:
        recommendations = RECS_HEART_LOW

    return {
        "prediction": prediction,
        "risk": risk,
        "risk_level": risk_level,
        "recommendations": recommendations
    }
# ══════════════════════════════════════════════════════════════
#  INPUT VALIDATION HELPERS
# ══════════════════════════════════════════════════════════════

def parse_form_inputs(form) -> tuple:
    """
    Extract and validate feature values from a Flask form submission.

    Returns:
        (values: list[float], errors: list[str])
    """
    values, errors = [], []
    for feature in ckd_features:
        raw = form.get(feature, "").strip()
        if not raw:
            errors.append(f"Missing value: '{feature}'")
            continue
        try:
            values.append(float(raw))
        except ValueError:
            errors.append(f"'{feature}' must be a number (got '{raw}')")
    return values, errors


def parse_json_inputs(data: dict) -> tuple:
    """
    Extract and validate feature values from a JSON payload.

    Returns:
        (values: list[float], errors: list[str])
    """
    values, errors = [], []
    for feature in ckd_features:
        if feature not in data:
            errors.append(f"Missing field: '{feature}'")
            continue
        try:
            values.append(float(data[feature]))
        except (ValueError, TypeError):
            errors.append(f"'{feature}' must be numeric (got '{data[feature]}')")
    return values, errors

def parse_heart_form(form):

    try:

        values = [

            float(form.get("male", 0)),
            float(form.get("age", 0)),
            2,  # education fixed

            float(form.get("currentSmoker", 0)),
            float(form.get("cigsPerDay", 0)),

            float(form.get("BPMeds", 0)),
            float(form.get("prevalentStroke", 0)),
            float(form.get("prevalentHyp", 0)),
            float(form.get("diabetes", 0)),

            float(form.get("totChol", 0)),
            float(form.get("sysBP", 0)),
            float(form.get("diaBP", 0)),

            float(form.get("BMI", 0)),
            float(form.get("heartRate", 0)),
            float(form.get("glucose", 0))
        ]

        return values, []

    except ValueError:

        return [], [
            "Invalid numeric value supplied."
        ]


def _ocr_session_key(target: str, suffix: str) -> str:
    return f"ocr_{suffix}_{target}"


def _form_endpoint_for_target(target: str) -> str:
    return "ckd_form" if target == "ckd" else "heart_form"


def _cache_ocr_values(target: str, file_hash: str, field_values: dict) -> None:
    session[_ocr_session_key(target, "hash")] = file_hash
    session[_ocr_session_key(target, "autofill")] = field_values


def _set_ocr_message(target: str, message: str = "", error: str = "") -> None:
    if message:
        session[_ocr_session_key(target, "status")] = message
    if error:
        session[_ocr_session_key(target, "error")] = error


def _ocr_template_context(target: str) -> dict:

    return {
        "ocr_autofill": session.pop(
            _ocr_session_key(target, "autofill"),
            {}
        ),

        "ocr_status": session.pop(
            _ocr_session_key(target, "status"),
            None
        ),

        "ocr_error": session.pop(
            _ocr_session_key(target, "error"),
            None
        ),
    }


def _wants_json_response() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _ocr_upload_response(target: str, endpoint: str, message: str = "", error: str = "", values: dict | None = None):
    if _wants_json_response():
        return jsonify({
            "ok": not bool(error),
            "message": message,
            "error": error,
            "values": values or {},
        }), (400 if error else 200)

    _set_ocr_message(target, message=message, error=error)
    return redirect(url_for(endpoint))
# ══════════════════════════════════════════════════════════════
#  ROUTES — PAGE VIEWS
# ══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/ckd")
def ckd():
    return render_template("ckd_home.html")


@app.route("/ckd-form")
def ckd_form():
    if ckd_model is None:
        log.warning("CKD form requested but model is not loaded.")
        return render_template(
            "error.html",
            code=503,
            message="The CKD prediction model is currently unavailable. Please try again later.",
        ), 503
    return render_template("ckd.html", **_ocr_template_context("ckd"))


@app.route("/heart")
def heart():
    return render_template("heart_home.html")

@app.route("/heart-form")
def heart_form():

    if heart_model is None:
        return render_template(
            "error.html",
            code=503,
            message="Heart model unavailable."
        ), 503

    return render_template("heart_form.html", **_ocr_template_context("heart"))

@app.route("/haris-test")
def haris_test():
    ...

@app.route("/upload")
def upload():
    """Report upload page — placeholder for the OCR feature."""
    return render_template("dashboard.html")


# ══════════════════════════════════════════════════════════════
#  ROUTES — CKD PREDICTION  (web form POST)
# ══════════════════════════════════════════════════════════════

@app.route("/ocr-upload/<target>", methods=["POST"])
def ocr_upload(target):
    """Extract report values and return the user to the existing form."""

    if target not in {"ckd", "heart"}:
        return render_template(
            "error.html",
            code=404,
            message="Unknown OCR target.",
        ), 404

    endpoint = _form_endpoint_for_target(target)
    uploaded_file = request.files.get("report")
    if not uploaded_file or not uploaded_file.filename:
        return _ocr_upload_response(target, endpoint, error="Please choose a report file before uploading.")

    if not is_allowed_file(uploaded_file.filename):
        return _ocr_upload_response(
            target,
            endpoint,
            error="Unsupported file type. Upload a PDF, JPG, JPEG, or PNG report.",
        )

    content = uploaded_file.read()
    if not content:
        return _ocr_upload_response(target, endpoint, error="The uploaded file is empty.")

    file_hash = hashlib.sha256(content).hexdigest()
    cached_hash = session.get(_ocr_session_key(target, "hash"))
    cached_values = session.get(_ocr_session_key(target, "autofill"))
    if cached_hash == file_hash and cached_values:
        return _ocr_upload_response(
            target,
            endpoint,
            message="Reused extracted values from your previous upload.",
            values=cached_values,
        )

    try:
        images = preprocess_upload(uploaded_file.filename, content)
        print("=" * 80)
        print("STEP 1: Images created")
        print(images)

        extracted_text = extract_text(images)
        print("=" * 80)
        print("STEP 2: OCR TEXT")
        print(extracted_text)

        parsed_values = parse_medical_values(extracted_text)
        print("=" * 80)
        print("STEP 3: PARSED VALUES")
        print(parsed_values)

        field_values = values_for_target(parsed_values, target)
        print("=" * 80)
        print("STEP 4: AUTOFILL VALUES")
        print(field_values)
        print("=" * 80)
    except OCRDependencyError as exc:
        log.warning("OCR dependency missing: %s", exc)
        return _ocr_upload_response(target, endpoint, error=str(exc))
    except OCRProcessingError as exc:
        log.warning("OCR processing failed: %s", exc)
        return _ocr_upload_response(target, endpoint, error=str(exc))
    except Exception as exc:
        log.exception("Unexpected OCR failure: %s", exc)
        return _ocr_upload_response(
            target,
            endpoint,
            error="OCR could not process this report. Try a clearer image or enter values manually.",
        )

    if not field_values:
        return _ocr_upload_response(
            target,
            endpoint,
            error="OCR completed, but no supported laboratory values were found. You can still enter values manually.",
        )

    _cache_ocr_values(target, file_hash, field_values)
    return _ocr_upload_response(
        target,
        endpoint,
        message=f"Extracted {len(field_values)} value{'s' if len(field_values) != 1 else ''}. Please review before prediction.",
        values=field_values,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Guard: model must be loaded
    if ckd_model is None or ckd_features is None:
        log.error("Prediction attempted but CKD model is not loaded.")
        return render_template(
            "error.html",
            code=503,
            message="The prediction model is unavailable. Please try again later.",
        ), 503

    # Validate form inputs
    values, errors = parse_form_inputs(request.form)
    if errors:
        log.warning("Form validation failed: %s", errors)
        return render_template(
            "ckd.html",
            error="Please fix the following issues: " + "; ".join(errors),
        ), 400

    # Run inference
    try:
        r = run_ckd_inference(values)
    except Exception as exc:
        log.exception("Inference error on /predict: %s", exc)
        return render_template(
            "error.html",
            code=500,
            message="An error occurred during prediction. Please try again.",
        ), 500

    log.info(
        "CKD prediction — %s | risk=%.2f%% | stage=%s | confidence=%.1f%%",
        r["result"], r["risk"], r["ckd_stage"], r["confidence"],
    )

    return render_template(
        "result.html",
        result           = r["result"],
        risk             = r["risk"],
        risk_level       = r["risk_level"],
        confidence       = r["confidence"],
        ckd_stage        = r["ckd_stage"],
        recommendations  = r["recommendations"],
        water_intake     = r["water_intake"],
        water_width      = r["water_width"],
        # Patient summary fields — use .get() to avoid KeyError
        age              = request.form.get("age",              "N/A"),
        bmi              = request.form.get("bmi",              "N/A"),
        bp_systolic      = request.form.get("bp_systolic",      "N/A"),
        bp_diastolic     = request.form.get("bp_diastolic",     "N/A"),
        serum_creatinine = request.form.get("serum_creatinine", "N/A"),
    )


@app.route("/predict-heart", methods=["POST"])
def predict_heart():

    if (
        heart_model is None
        or heart_scaler is None
        or heart_features is None
    ):
        return render_template(
            "error.html",
            code=503,
            message="Heart model unavailable."
        ), 503

    values, errors = parse_heart_form(request.form)

    if errors:
        return render_template(
            "heart_form.html",
            error="; ".join(errors)
        ), 400

    try:

        result = run_heart_inference(values)

        risk = float(result["risk"])

        # ----------------------------
        # Prediction Status
        # ----------------------------
        if risk >= 20:
            prediction_status = "Heart Disease Risk Detected"
        else:
            prediction_status = "No Heart Disease Risk Detected"

        # ----------------------------
        # Risk Category
        # ----------------------------
        if risk >= 30:
            heart_risk_category = "Very High"
        elif risk >= 20:
            heart_risk_category = "High"
        elif risk >= 10:
            heart_risk_category = "Moderate"
        else:
            heart_risk_category = "Low"

        # ----------------------------
        # Approximate confidence
        # (Later we will use predict_proba)
        # ----------------------------
        confidence = round(
            max(risk / 100, 1 - risk / 100) * 100,
            1
        )

        # ----------------------------
        # Exercise Recommendation
        # ----------------------------
        if risk < 10:
            exercise_goal = "150 min/week Moderate Exercise"
            exercise_width = 90

        elif risk < 20:
            exercise_goal = "120 min/week Moderate Exercise"
            exercise_width = 75

        elif risk < 30:
            exercise_goal = "Consult Doctor Before Exercise"
            exercise_width = 55

        else:
            exercise_goal = "Medical Supervision Required"
            exercise_width = 35

        return render_template(

            "heart_result.html",

            # Prediction
            result=prediction_status,
            risk=round(risk, 2),
            risk_level=result["risk_level"],
            recommendations=result["recommendations"],

            # Category
            heart_risk_category=heart_risk_category,

            # Confidence
            confidence=confidence,

            # Patient Information
            age=request.form.get("age"),
            gender="Male" if request.form.get("male") == "1" else "Female",

            smoker="Yes"
            if request.form.get("currentSmoker") == "1"
            else "No",

            currentSmoker=request.form.get("currentSmoker"),

            # Clinical Values
            sysBP=request.form.get("sysBP"),
            diaBP=request.form.get("diaBP"),
            totChol=request.form.get("totChol"),
            bmi=request.form.get("BMI"),
            heartRate=request.form.get("heartRate"),
            glucose=request.form.get("glucose"),

            # Exercise
            exercise_goal=exercise_goal,
            exercise_width=exercise_width

        )

    except Exception as exc:

        log.exception(
            "Heart prediction failed: %s",
            exc
        )

        return render_template(
            "error.html",
            code=500,
            message="Heart prediction failed."
        ), 500

# ══════════════════════════════════════════════════════════════
#  ROUTES — REST API  (/api/v1/...)
# ══════════════════════════════════════════════════════════════

@app.route("/api/v1/predict/ckd", methods=["POST"])
@app.route("/api/ckd_predict",    methods=["POST"])   # legacy alias
def api_ckd_predict():
    """
    CKD prediction API endpoint.

    Accepts:  JSON body with all CKD feature fields
    Returns:  JSON with prediction, risk score, stage, and recommendations

    Example request body:
        { "age": 45, "bmi": 26.5, "serum_creatinine": 1.1, ... }
    """
    if ckd_model is None or ckd_features is None:
        return jsonify({"error": "Model unavailable", "status": 503}), 503

    # Require JSON content
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error":  "Request body must be a valid JSON object",
            "status": 400,
        }), 400

    # Validate fields
    values, errors = parse_json_inputs(data)
    if errors:
        return jsonify({
            "error":   "Validation failed",
            "details": errors,
            "status":  400,
        }), 400

    # Run inference
    try:
        r = run_ckd_inference(values)
    except Exception as exc:
        log.exception("API inference error: %s", exc)
        return jsonify({"error": "Inference failed", "status": 500}), 500

    log.info("API CKD — %s | risk=%.2f%%", r["result"], r["risk"])

    return jsonify({
        "status":          200,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "prediction":      r["result"],
        "risk":            r["risk"],
        "risk_level":      r["risk_level"],
        "confidence":      r["confidence"],
        "ckd_stage":       r["ckd_stage"],
        "water_intake":    r["water_intake"],
        "recommendations": r["recommendations"],
    })


@app.route("/api/v1/health")
@app.route("/health")
def health():
    """
    Health check endpoint.
    Returns 200 if the model is loaded, 503 if degraded.
    """
    model_ready = ckd_model is not None and ckd_features is not None

    return jsonify({
        "status":    "ok" if model_ready else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": {
            "ckd": {
                "loaded":        model_ready,
                "feature_count": len(ckd_features) if ckd_features else 0,
            },
            "heart": {
                "loaded": False,   # reserved for future module
            },
        },
    }), (200 if model_ready else 503)


# ══════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════

def _is_api_request() -> bool:
    """Return True if the request expects a JSON response."""
    return request.is_json or request.path.startswith("/api/")


@app.errorhandler(400)
def bad_request(exc):
    if _is_api_request():
        return jsonify({"error": "Bad request", "status": 400}), 400
    return render_template("error.html", code=400,
                           message="Bad request — please check your inputs."), 400


@app.errorhandler(404)
def not_found(exc):
    if _is_api_request():
        return jsonify({"error": "Not found", "status": 404}), 404
    return render_template("error.html", code=404,
                           message="Page not found."), 404


@app.errorhandler(405)
def method_not_allowed(exc):
    if _is_api_request():
        return jsonify({"error": "Method not allowed", "status": 405}), 405
    return render_template("error.html", code=405,
                           message="Method not allowed."), 405


@app.errorhandler(500)
def internal_error(exc):
    log.exception("Unhandled 500: %s", exc)
    if _is_api_request():
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return render_template("error.html", code=500,
                           message="Something went wrong on our end. Please try again."), 500


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
print("\nROUTES:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == "__main__":
    app.run(
        host  = Config.HOST,
        port  = Config.PORT,
        debug = Config.DEBUG,
    )
