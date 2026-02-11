"""Utility helper functions."""

import re


def validate_text(text: str, min_length: int = 10, max_length: int = 5000) -> tuple:
    """Validate input text. Returns (is_valid, error_message)."""
    if not text or not isinstance(text, str):
        return False, "Text input is required."

    text = text.strip()

    if len(text) < min_length:
        return False, f"Text must be at least {min_length} characters long."

    if len(text) > max_length:
        return False, f"Text must not exceed {max_length} characters."

    # Check if text has actual words
    words = re.findall(r"[a-zA-Z]+", text)
    if len(words) < 3:
        return False, "Text must contain at least 3 words."

    return True, None


def format_risk_response(prediction: dict, config_risk_levels: dict) -> dict:
    """Format the prediction response with risk level metadata."""
    risk_code = prediction.get("risk_code", 0)
    risk_info = config_risk_levels.get(risk_code, {})

    return {
        "risk_level": prediction.get("risk_level", "Unknown"),
        "risk_code": risk_code,
        "color": risk_info.get("color", "#gray"),
        "description": risk_info.get("description", ""),
        "probabilities": prediction.get("probabilities", {}),
        "model_used": prediction.get("model_used", "unknown"),
    }
