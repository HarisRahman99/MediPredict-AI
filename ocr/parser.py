from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

print("=" * 60)
print("USING NEW PARSER V2")
print("=" * 60)

ParsedValues = Dict[str, object]
NUMBER_RE = re.compile(
    r"(?<![\w.])([<>]?\s*\d+(?:\.\d+)?)"
)

YES_RE = re.compile(r"\byes\b", re.IGNORECASE)

NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

OCR_REPLACEMENTS = {

    "mgldl": "mg/dl",
    "mgidl": "mg/dl",
    "mgdl": "mg/dl",

    "mglg": "mg/g",

    "kg/m?": "kg/m²",

    "agelgender": "age\ngender",

    "bp systolic": "bp systolic",
    "bp diastolic": "bp diastolic",

    "albumin creatinine ratio": "albumin creatinine ratio",

}

NUMERIC_PATTERNS = {

    "age": (
        r"\bage\b",
        r"\bage\s*/\s*gender\b",
        r"\bagegender\b",
        r"\bagelgender\b",
    ),

    "height_cm": (
        r"\bheight\b",
    ),

    "weight_kg": (
        r"\bweight\b",
    ),

    "bmi": (
        r"\bbmi\b",
    ),

    "bp_systolic": (
        r"\bbp\s*systolic\b",
        r"\bsystolic\b",
    ),

    "bp_diastolic": (
        r"\bbp\s*diastolic\b",
        r"\bdiastolic\b",
    ),

    "serum_creatinine": (
        r"\bserum\s+creatinine\b",
        
    ),

    "blood_urea_nitrogen": (
        r"\bblood\s+urea\s+nitrogen\b",
        r"\bblood\s+urea\b",
        r"\bbun\b",
    ),

    "albumin_serum": (
        r"\balbumin\s*\(serum\)\b",
        r"\bserum\s+albumin\b",
        r"\balbumin\s+serum\b",
    ),

    "urine_albumin": (
        r"\burine\s+albumin\b",
    ),

    "albumin_creatinine_ratio": (
        r"\balbumin\s+creatinine\s+ratio\b",
        r"\bacr\b",
    ),

    "urine_creatinine": (
        r"\burine\s+creatinine\b",
    ),

    "calcium": (
        r"\bcalcium\b",
    ),

    "phosphorus": (
        r"\bphosphorus\b",
    ),

    "bicarbonate": (
        r"\bbicarbonate\b",
    ),

    "uric_acid": (
        r"\buric\s+acid\b",
    ),

    "glucose": (
        r"\bglucose\b",
    ),

    "heart_rate": (
        r"\bheart\s+rate\b",
        r"\bpulse\b",
    ),

    "cholesterol": (
        r"\bcholesterol\b",
    ),

    "hdl": (
        r"\bhdl\b",
    ),

    "ldl": (
        r"\bldl\b",
    ),

    "triglycerides": (
        r"\btriglycerides\b",
    ),

    "hemoglobin": (
        r"\bhemoglobin\b",
        r"\bhb\b",
    ),

    "egfr": (
        r"\begfr\b",
    ),

    "poverty_income_ratio": (
        r"\bpoverty\s+income\s+ratio\b",
    ),

}

BOOLEAN_PATTERNS = {

    "diabetes_diagnosed": (
        r"\bdiabetes\s+diagnosed\b",
    ),

    "insulin_use": (
        r"\binsulin\s+use\b",
    ),

    "diabetes_pills": (
        r"\bdiabetes\s+medication\b",
        r"\bdiabetes\s+pills\b",
    ),

    "ever_smoked": (
        r"\bever\s+smoked\b",
    ),

    "current_smoker": (
        r"\bcurrent\s+smoker\b",
    ),

}

TEXT_PATTERNS = {

    "gender": (
        r"\bgender\b",
    ),

    "ethnicity": (
        r"\bethnicity\b",
    ),

    "education_level": (
        r"\beducation\s+level\b",
    ),

}

#helper1
def normalize_text(text: str) -> str:

    text = text.lower()

    text = text.replace("\r", "\n")

    for old, new in OCR_REPLACEMENTS.items():
        text = text.replace(old, new)

    text = re.sub(r"[ ]{2,}", " ", text)

    return text

#helper2
def to_float(value: str):

    value = value.replace("<", "")
    value = value.replace(">", "")
    value = value.strip()

    try:
        return float(value)
    except:

        return None
    
#part2:helper 
def _find_number(text: str):

    match = NUMBER_RE.search(text)

    if not match:
        return None

    return to_float(match.group(1))

#numeric parser
def _find_numeric(lines: List[str], patterns: Tuple[str, ...]):

    label_re = re.compile(
        "|".join(f"(?:{p})" for p in patterns),
        re.IGNORECASE,
    )

    print("\nSearching patterns:", patterns)

    for i, line in enumerate(lines):

        if label_re.search(line):
            print(f"Matched line {i}: {line}")

            value = _find_number(line)

            if value is not None:
                print("Found on same line:", value)
                return value

            for offset in (1, 2, 3):

                if i + offset >= len(lines):
                    break

                print(f"Checking line {i+offset}: {lines[i+offset]}")

                value = _find_number(lines[i + offset])

                if value is not None:
                    print("Found on next line:", value)
                    return value

    print("No match found")
    return None

#boolean parser
def _find_boolean(lines: List[str], patterns: Tuple[str, ...]):

    label_re = re.compile(
        "|".join(f"(?:{p})" for p in patterns),
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):

        if not label_re.search(line):
            continue

        candidates = []

        candidates.append(line)

        if i + 1 < len(lines):
            candidates.append(lines[i + 1])

        if i + 2 < len(lines):
            candidates.append(lines[i + 2])

        joined = " ".join(candidates)

        if YES_RE.search(joined):
            return 1

        if NO_RE.search(joined):
            return 0

    return None

#text parser
def _find_text(lines: List[str], patterns: Tuple[str, ...]):

    label_re = re.compile(
        "|".join(f"(?:{p})" for p in patterns),
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):

        if not label_re.search(line):
            continue

        if i + 1 >= len(lines):
            return None

        value = lines[i + 1].strip()

        if not value:
            return None

        if NUMBER_RE.search(value):
            return None

        return value

    return None

#normalize ocr lines
def _prepare_lines(text: str):

    text = normalize_text(text)

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    return lines

NUMERIC_KEYS = list(NUMERIC_PATTERNS.keys())

BOOLEAN_KEYS = list(BOOLEAN_PATTERNS.keys())

TEXT_KEYS = list(TEXT_PATTERNS.keys())

#part 3 parse_medical_values()
def parse_medical_values(text: str) -> ParsedValues:

    lines = _prepare_lines(text)

    print("\nAGE TEST")
    age = _find_numeric(lines, NUMERIC_PATTERNS["age"])
    print("AGE =", age)

    values: ParsedValues = {}

    # Numeric fields
    for key in NUMERIC_KEYS:
        value = _find_numeric(lines, NUMERIC_PATTERNS[key])
        if value is not None:
            values[key] = value

    # Boolean fields
    for key in BOOLEAN_KEYS:
        value = _find_boolean(lines, BOOLEAN_PATTERNS[key])
        if value is not None:
            values[key] = value

    # Text fields
    for key in TEXT_KEYS:
        value = _find_text(lines, TEXT_PATTERNS[key])
        if value is not None:
            values[key] = value

    return values