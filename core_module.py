"""
core_module.py

Local web orchestrator for the Fawkes cloaking/verification pipeline.

The existing pipeline is intentionally preserved:
    fawkes_module.cloak_folder()
    -> organize_cloaked.sh
    -> verification_module.verify_cloak()

This version adds:
- multi-face extraction per image (instead of silently taking face #1)
- cross-image identity clustering using DBSCAN when available
- original <-> cloaked face matching by nearest embedding
- JSON cluster data for a modern browser UI
- safer subprocess/error handling
- PCA fallback for visualisation data
- backward-compatible "cluster_image" PNG output
"""

import base64
import io
import subprocess
from pathlib import Path

import face_recognition
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request
from sklearn.decomposition import PCA

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

import fawkes_module
import verification_module

app = Flask(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CLUSTER_EPS = 0.48
DEFAULT_MIN_SAMPLES = 1


def classify_level(distance, threshold):
    """Turn a raw distance/threshold pair into a graded protection level."""
    if not threshold:
        return "Weak"
    ratio = distance / threshold
    if ratio >= 1.3:
        return "Strong"
    if ratio >= 1.0:
        return "Moderate"
    return "Weak"


def load_faces(image_path):
    """
    Return all face encodings and their locations.

    face_recognition's default HOG detector is retained for compatibility
    with older installations.  A failed image simply returns an empty list.
    """
    try:
        image = face_recognition.load_image_file(str(image_path))
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)
        return [
            {"encoding": encoding, "location": list(location)}
            for encoding, location in zip(encodings, locations)
        ]
    except Exception:
        return []


def match_faces(original_faces, cloaked_faces, tolerance=0.60):
    """
    Match faces between one original and one cloaked image.

    Greedy nearest-neighbour matching is used deliberately: it requires no
    newer face-recognition dependency and prevents one cloaked face from
    being assigned to multiple original faces.
    """
    if not original_faces or not cloaked_faces:
        return []

    distances = []
    for oi, original in enumerate(original_faces):
        for ci, cloaked in enumerate(cloaked_faces):
            try:
                d = float(face_recognition.face_distance(
                    [original["encoding"]], cloaked["encoding"]
                )[0])
            except Exception:
                continue
            distances.append((d, oi, ci))

    matches = []
    used_original = set()
    used_cloaked = set()

    for distance, oi, ci in sorted(distances):
        if oi in used_original or ci in used_cloaked:
            continue
        # Do not force an obviously unrelated face into a pair.
        if distance > tolerance:
            continue
        used_original.add(oi)
        used_cloaked.add(ci)
        matches.append({
            "original_index": oi,
            "cloaked_index": ci,
            "distance": distance,
        })

    return matches


def _project_vectors(vectors):
    """Project vectors into 2D without assuming a large dataset."""
    if not vectors:
        return []

    if len(vectors) == 1:
        return [[0.0, 0.0]]

    n_components = min(2, len(vectors), len(vectors[0]))
    pca = PCA(n_components=n_components)
    points = pca.fit_transform(vectors)

    if n_components == 1:
        return [[float(p[0]), 0.0] for p in points]
    return [[float(p[0]), float(p[1])] for p in points]


def build_cluster_data(face_records, eps=DEFAULT_CLUSTER_EPS):
    """
    Identify people from ORIGINAL embeddings, then project cloaked faces onto
    those identity clusters by nearest-neighbour matching.

    This is preferable to clustering original + cloaked embeddings together:
    cloaking is specifically expected to move an embedding, so mixing both
    variants can accidentally turn one person into two "people".
    """
    if not face_records:
        return {"points": [], "clusters": [], "algorithm": "none"}

    originals = [r for r in face_records if r["variant"] == "original"]
    cloaked = [r for r in face_records if r["variant"] == "cloaked"]

    if not originals:
        points_2d = _project_vectors([r["encoding"] for r in face_records])
        return {
            "points": [
                dict(
                    id=r["id"], image=r["image"], face_index=r["face_index"],
                    variant=r["variant"], cluster=-1,
                    x=round(xy[0], 5), y=round(xy[1], 5),
                )
                for r, xy in zip(face_records, points_2d)
            ],
            "clusters": [{"id": -1, "name": "Unclustered", "size": len(face_records)}],
            "algorithm": "no-originals",
            "eps": eps,
        }

    original_vectors = [r["encoding"] for r in originals]

    if DBSCAN is not None:
        try:
            # min_samples=1 means a face can still represent a person when
            # only one photo of that person exists.
            original_labels = DBSCAN(
                eps=eps, min_samples=1, metric="euclidean"
            ).fit(original_vectors).labels_.tolist()
            algorithm = "DBSCAN(originals)+nearest(cloaked)"
        except Exception:
            original_labels = list(range(len(originals)))
            algorithm = "individual(originals)+nearest(cloaked)"
    else:
        original_labels = list(range(len(originals)))
        algorithm = "individual(originals)+nearest(cloaked)"

    # A cluster is represented by the closest original embedding. This avoids
    # requiring a newer classifier package and makes the threshold explicit.
    cluster_representatives = {}
    for record, label in zip(originals, original_labels):
        cluster_representatives.setdefault(label, []).append(record["encoding"])

    def nearest_original_cluster(encoding):
        best_label, best_distance = None, float("inf")
        for label, members in cluster_representatives.items():
            for member in members:
                try:
                    distance = float(face_recognition.face_distance(
                        [member], encoding
                    )[0])
                except Exception:
                    continue
                if distance < best_distance:
                    best_label, best_distance = label, distance
        if best_distance <= eps:
            return best_label
        return -1

    all_records = []
    for record, label in zip(originals, original_labels):
        all_records.append((record, label))

    for record in cloaked:
        all_records.append((record, nearest_original_cluster(record["encoding"])))

    vectors = [record["encoding"] for record, _ in all_records]
    points_2d = _project_vectors(vectors)
    points = []

    for (record, label), xy in zip(all_records, points_2d):
        points.append({
            "id": record["id"],
            "image": record["image"],
            "face_index": record["face_index"],
            "variant": record["variant"],
            "cluster": int(label),
            "x": round(xy[0], 5),
            "y": round(xy[1], 5),
        })

    cluster_ids = sorted(set(label for _, label in all_records))
    clusters = []
    for cluster_id in cluster_ids:
        members = [p for p in points if p["cluster"] == cluster_id]
        if cluster_id == -1:
            name = "Unmatched / unknown"
        else:
            name = "Person %d" % (cluster_id + 1)
        clusters.append({
            "id": int(cluster_id),
            "name": name,
            "size": len(members),
        })

    return {
        "points": points,
        "clusters": clusters,
        "algorithm": algorithm,
        "eps": eps,
    }


def build_cluster_plot(face_records, cluster_data=None):
    """Backward-compatible PNG plot generated from the cluster data."""
    if not face_records:
        return ""

    # Do not read ``record["cluster"]`` here: face_records intentionally keep
    # raw embeddings and are not mutated by build_cluster_data().
    if cluster_data is None:
        cluster_data = build_cluster_data(face_records)

    plot_points = cluster_data.get("points", [])
    if not plot_points:
        return ""

    point_by_id = {p["id"]: p for p in plot_points}
    fig, ax = plt.subplots(figsize=(7, 5))

    for record in face_records:
        point = point_by_id.get(record["id"])
        if point is None:
            continue
        cluster = point.get("cluster", -1)
        marker = "o" if record["variant"] == "original" else "X"
        ax.scatter(
            point["x"], point["y"],
            s=75,
            alpha=0.85,
            marker=marker,
            label="Person %d" % (cluster + 1) if cluster >= 0 else "Unmatched",
        )

    # Avoid a huge repeated legend when several faces share a cluster.
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    ax.set_title("Face Embedding Clusters")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.18)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def verify_pair_fallback(original, cloaked):
    """
    Keep the existing verification module authoritative for normal
    single-face images. This fallback only provides multi-face information
    when that module cannot represent multiple faces.
    """
    original_faces = load_faces(original)
    cloaked_faces = load_faces(cloaked)
    matches = match_faces(original_faces, cloaked_faces)

    if not matches:
        return []

    return matches


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/browse", methods=["GET"])
def browse():
    """Open a native OS folder-picker dialog on the local machine."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder_selected = filedialog.askdirectory()
        root.destroy()
        return jsonify({"folder_path": folder_selected})
    except Exception as exc:
        return jsonify({"folder_path": "", "error": str(exc)}), 500


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    """Return JSON for API failures instead of Flask's HTML traceback page."""
    app.logger.exception("Unhandled request error")
    return jsonify({
        "status": "error",
        "message": "Internal server error: %s" % exc,
    }), 500


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(force=True) or {}
    folder_path = (data.get("folder_path") or "").strip()
    mode = data.get("mode", "mid")
    try:
        cluster_eps = float(data.get("cluster_eps", DEFAULT_CLUSTER_EPS))
    except (TypeError, ValueError):
        cluster_eps = DEFAULT_CLUSTER_EPS

    cluster_eps = max(0.20, min(1.00, cluster_eps))

    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        return jsonify({
            "status": "error",
            "message": "Not a valid directory: %s" % folder,
        }), 400

    original_images = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not original_images:
        return jsonify({"status": "error", "message": "No images found in folder."}), 400

    # Step 1: existing Fawkes module.
    try:
        status = fawkes_module.cloak_folder(str(folder), mode=mode)
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": "Fawkes cloaking raised an exception: %s" % exc,
        }), 500

    if status != "done":
        return jsonify({
            "status": "error",
            "message": "Fawkes cloaking failed.",
        }), 500

    # Step 2: existing organiser script.
    script_path = Path(__file__).parent / "organize_cloaked.sh"
    if not script_path.is_file():
        return jsonify({
            "status": "error",
            "message": "organize_cloaked.sh not found.",
        }), 500

    try:
        org_result = subprocess.run(
            ["bash", str(script_path), str(folder)],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": "Could not run organizer: %s" % exc,
        }), 500

    if org_result.returncode != 0 or org_result.stdout.strip() != "done":
        detail = (org_result.stderr or org_result.stdout).strip()
        return jsonify({
            "status": "error",
            "message": "Organizing cloaked files failed%s." %
                       (": " + detail if detail else ""),
        }), 500

    cloaked_dir = folder / "cloaked_version"
    cloaked_files = sorted(
        f for f in cloaked_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ) if cloaked_dir.is_dir() else []

    if not cloaked_files:
        return jsonify({
            "status": "error",
            "message": "No cloaked files produced.",
        }), 500

    results = []
    face_records = []
    pair_links = []

    # Cache face extraction so each image is decoded once.
    original_face_cache = {}
    cloaked_face_cache = {}

    for original in original_images:
        # Preserve the original filename-prefix matching behaviour, but make
        # it deterministic and extension-aware.
        candidates = [
            c for c in cloaked_files
            if c.stem == original.stem
            or c.stem.startswith(original.stem + "_")
            or c.stem.startswith(original.stem + "-")
        ]
        if not candidates:
            candidates = [c for c in cloaked_files if c.stem.startswith(original.stem)]
        match = candidates[0] if candidates else None

        if not match:
            results.append({
                "name": original.name,
                "status": "skipped",
                "message": "No matching cloaked file.",
            })
            continue

        # Keep the old verifier as the authoritative per-image verdict.
        try:
            verdict = verification_module.verify_cloak(
                str(original), str(match)
            )
        except Exception as exc:
            verdict = {
                "status": "error",
                "message": "Verification exception: %s" % exc,
            }

        if verdict.get("status") == "ok":
            distance = float(verdict.get("distance", 0.0))
            threshold = float(verdict.get("threshold", 1.0))
            confidence = float(verdict.get("confidence", 0.0))
            results.append({
                "name": original.name,
                "cloaked_name": match.name,
                "status": "ok",
                "protected": verdict.get("protected"),
                "distance": distance,
                "threshold": threshold,
                "confidence": confidence,
                "level": classify_level(distance, threshold),
            })
        else:
            results.append({
                "name": original.name,
                "cloaked_name": match.name,
                "status": "error",
                "message": verdict.get("message", "Verification failed."),
            })

        # Multi-face extraction and clustering.
        original_faces = load_faces(original)
        cloaked_faces = load_faces(match)
        original_face_cache[original] = original_faces
        cloaked_face_cache[match] = cloaked_faces

        for idx, face in enumerate(original_faces):
            face_records.append({
                "id": "%s:original:%d" % (original.stem, idx),
                "image": original.name,
                "face_index": idx,
                "variant": "original",
                "encoding": face["encoding"],
            })

        for idx, face in enumerate(cloaked_faces):
            face_records.append({
                "id": "%s:cloaked:%d" % (original.stem, idx),
                "image": match.name,
                "face_index": idx,
                "variant": "cloaked",
                "encoding": face["encoding"],
            })

        # Explicit original <-> cloaked matching for every detected face.
        for item in match_faces(original_faces, cloaked_faces):
            pair_links.append({
                "original_id": "%s:original:%d" % (
                    original.stem, item["original_index"]
                ),
                "cloaked_id": "%s:cloaked:%d" % (
                    original.stem, item["cloaked_index"]
                ),
                "distance": round(item["distance"], 5),
            })

    cluster_data = build_cluster_data(face_records, eps=cluster_eps)
    cluster_data["pair_links"] = pair_links
    cluster_data["face_count"] = len(face_records)
    cluster_data["people_count"] = len([
        c for c in cluster_data["clusters"] if c["id"] != -1
    ])

    return jsonify({
        "status": "ok",
        "results": results,
        "cluster": cluster_data,
        # Retained so older frontends or scripts can still consume a PNG.
        "cluster_image": build_cluster_plot(face_records, cluster_data),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
