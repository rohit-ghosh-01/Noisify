"""
verification_module.py

Compares an original image to its cloaked version using face_recognition
(dlib-based) to check whether the cloak actually fools recognition. Uses
no TensorFlow, so it runs fine in the same environment as Fawkes on
Python 3.7 with no dependency conflicts.

Public function returns a status dict shaped like:

    {
        "status": "ok" | "error",
        "protected": True | False,   # True = model could NOT match faces (cloak worked)
        "distance": float,           # raw face-encoding distance
        "threshold": float,          # same-person cutoff
        "confidence": float,         # 0-100, how far the verdict is from the threshold
        "message": <str, present on error>
    }
"""

import face_recognition

THRESHOLD = 0.6  # standard face_recognition same-person cutoff


def verify_cloak(original_path: str, cloaked_path: str) -> dict:
    """Compare original vs cloaked image and return protection verdict + confidence."""
    try:
        original_image = face_recognition.load_image_file(original_path)
        cloaked_image = face_recognition.load_image_file(cloaked_path)
    except Exception as e:
        return {"status": "error", "message": f"Could not load images: {e}"}

    original_encodings = face_recognition.face_encodings(original_image)
    if not original_encodings:
        return {"status": "error", "message": "No face detected in original image."}

    cloaked_encodings = face_recognition.face_encodings(cloaked_image)
    if not cloaked_encodings:
        return {"status": "error", "message": "No face detected in cloaked image."}

    distance = face_recognition.face_distance([original_encodings[0]], cloaked_encodings[0])[0]
    same_person = distance <= THRESHOLD

    # Confidence: how far the distance sits from the decision threshold,
    # as a percentage of the threshold itself.
    confidence = min(100.0, abs(distance - THRESHOLD) / THRESHOLD * 100)
    level="high" if confidence > 50 else "medium" if confidence > 20 else "low"
    return {
        "status": "ok",
        "protected": not same_person,
        "distance": round(float(distance), 4),
        "threshold": THRESHOLD,
        "confidence": round(confidence, 1),
        "level":level
    }