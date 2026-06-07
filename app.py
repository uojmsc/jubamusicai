import os
import pickle
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "best_model.h5"))
GENRES_PATH = Path(os.getenv("GENRES_PATH", APP_DIR / "genres.pkl"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", APP_DIR / "uploads"))


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["ALLOWED_EXTENSIONS"] = {"wav", "mp3", "ogg", "m4a"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(APP_DIR / "static").mkdir(parents=True, exist_ok=True)

GENRE_DATA = {
    "blues": {
        "description": "A genre and musical form which originated in the Deep South of the United States around the 1860s.",
        "origin": "Deep South, USA",
        "culture": "African-American communities, evolving from spirituals, work songs, and field hollers."
    },
    "classical": {
        "description": "Serious or conventional music following long-established principles rather than a folk, jazz, or popular tradition.",
        "origin": "Europe",
        "culture": "Western liturgical and secular music, spanning from the 11th century to the present day."
    },
    "hiphop": {
        "description": "A culture and art movement that emerged from the Bronx in New York City during the early 1970s.",
        "origin": "Bronx, New York",
        "culture": "Developed by African Americans, Latino Americans, and Caribbean Americans; characterized by four pillars: MCing, DJing, breakdancing, and graffiti."
    },
    # ... you can expand this dictionary with more genres
}


_MODEL = None
_GENRES = None
_AUDIO_CACHE: dict[str, dict[str, object]] = {}
_AUDIO_TTL_SECONDS = int(os.getenv("AUDIO_TTL_SECONDS", "900"))


def _read_h5_keras_version(model_path: Path) -> str | None:
    if model_path.suffix.lower() not in {".h5", ".hdf5"}:
        return None
    try:
        import h5py  # type: ignore
    except Exception:
        return None

    try:
        with h5py.File(model_path, "r") as f:
            v = f.attrs.get("keras_version")
            if v is None:
                return None
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            return str(v)
    except Exception:
        return None


class CompatInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, batch_shape=None, optional=None, **kwargs):
        if batch_shape is not None:
            batch_shape = tuple(batch_shape)

            if "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = batch_shape

            kwargs.pop("batch_shape", None)

        kwargs.pop("optional", None)

        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # Fallback for older InputLayer signatures
            batch_input_shape = kwargs.pop("batch_input_shape", None)
            if batch_input_shape is not None:
                batch_size = batch_input_shape[0]
                input_shape = batch_input_shape[1:]
                super().__init__(input_shape=input_shape, batch_size=batch_size, **kwargs)
            else:
                super().__init__(*args, **kwargs)


class CompatDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        # Some saved models (newer Keras) include `quantization_config` in Dense configs.
        # Older TF/Keras releases don't recognize it, so drop it for compatibility.
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def _prune_audio_cache() -> None:
    now = time.time()
    expired_tokens = []
    for token, entry in list(_AUDIO_CACHE.items()):
        expires_at = float(entry.get("expires_at", 0))
        path = entry.get("path")
        if expires_at <= now or not isinstance(path, Path) or not path.exists():
            expired_tokens.append(token)

    for token in expired_tokens:
        entry = _AUDIO_CACHE.pop(token, None)
        if not entry:
            continue
        path = entry.get("path")
        if isinstance(path, Path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


@app.before_request
def _before_request():
    _prune_audio_cache()


@app.context_processor
def inject_copyright_year():
    return {"copyright_year": datetime.now().year}


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/audio/<token>")
def audio(token: str):
    entry = _AUDIO_CACHE.get(token)
    if not entry:
        abort(404)

    path = entry.get("path")
    if not isinstance(path, Path):
        abort(404)

    resolved = path.resolve()
    uploads_resolved = UPLOAD_DIR.resolve()
    if uploads_resolved not in resolved.parents and resolved != uploads_resolved:
        abort(404)

    if not resolved.exists():
        abort(404)

    resp = send_file(str(resolved), as_attachment=False, conditional=True)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def load_artifacts():
    global _MODEL, _GENRES

    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Set MODEL_PATH or place best_model.h5 next to app.py."
            )
        saved_keras_version = _read_h5_keras_version(MODEL_PATH)
        if saved_keras_version is not None:
            try:
                import keras as _keras  # type: ignore

                runtime_keras_version = str(_keras.__version__)
            except Exception:
                runtime_keras_version = None

            if runtime_keras_version is not None:
                saved_major = saved_keras_version.split(".", 1)[0]
                runtime_major = runtime_keras_version.split(".", 1)[0]
                if saved_major != runtime_major:
                    raise RuntimeError(
                        "Incompatible Keras versions: "
                        f"{MODEL_PATH.name} was saved with Keras {saved_keras_version}, "
                        f"but this environment is running Keras {runtime_keras_version}. "
                        "Upgrade TensorFlow/Keras (recommended for this repo) or re-save the model using your current "
                        "Keras version."
                    )
        _MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "InputLayer": CompatInputLayer,
                "Dense": CompatDense,
                "DTypePolicy": tf.keras.mixed_precision.Policy,
            },
        )

    if _GENRES is None:
        if not GENRES_PATH.exists():
            raise FileNotFoundError(
                f"Genres file not found at {GENRES_PATH}. Set GENRES_PATH or place genres.pkl next to app.py."
            )
        with GENRES_PATH.open("rb") as f:
            _GENRES = pickle.load(f)

    return _MODEL, _GENRES


def extract_features(file_path: str, duration: int = 30, sr: int = 22050) -> np.ndarray | None:
    try:
        audio, sr = librosa.load(file_path, duration=duration, sr=sr)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        tonnetz_std = np.std(tonnetz.T, axis=0)

        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms[0])
        rms_std = np.std(rms[0])

        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr[0])
        zcr_std = np.std(zcr[0])

        features = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                chroma_mean,
                chroma_std,
                contrast_mean,
                contrast_std,
                tonnetz_mean,
                tonnetz_std,
                [rms_mean, rms_std, zcr_mean, zcr_std],
            ]
        ).astype(np.float32)

        return features
    except Exception:
        return None


def predict_genre(file_path: str):
    model, genres = load_artifacts()

    features = extract_features(file_path)
    if features is None:
        return None, None, None

    features = features.reshape(1, -1)
    predictions = np.asarray(model.predict(features, verbose=0))

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(f"Unexpected model output shape: {predictions.shape}")

    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_genre = genres[predicted_idx]

    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3 = [(genres[int(i)], float(predictions[0][int(i)])) for i in top_3_idx]

    return predicted_genre, confidence, top_3


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/contact")
def contact():
    return render_template("contact.html")


@app.get("/search")
def search():
    query = request.args.get("q", "").strip().lower()
    results = {}
    if query:
        for genre, info in GENRE_DATA.items():
            if (query in genre.lower() or 
                query in info["description"].lower() or 
                query in info["culture"].lower()):
                results[genre] = info
    return render_template("search_results.html", results=results, query=query)


@app.get("/classify")
def classify():
    try:
        _, genres = load_artifacts()
        return render_template("index.html", genres=genres)
    except Exception as exc:
        flash(str(exc), "error")
        return render_template("index.html", genres=[])


@app.post("/predict")
def predict():
    if "audio_file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("classify"))

    file = request.files["audio_file"]
    if file.filename is None or file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("classify"))

    if not allowed_file(file.filename):
        flash("File type not allowed. Use WAV, MP3, OGG, or M4A", "error")
        return redirect(url_for("classify"))

    original_filename = secure_filename(file.filename)
    ext = Path(original_filename).suffix.lower()
    token = uuid.uuid4().hex
    saved_name = f"{token}{ext}"
    filepath = UPLOAD_DIR / saved_name

    try:
        file.save(str(filepath))

        predicted_genre, confidence, top_3 = predict_genre(str(filepath))
        if predicted_genre is None:
            flash("Error processing audio", "error")
            return redirect(url_for("classify"))

        _AUDIO_CACHE[token] = {
            "path": filepath,
            "expires_at": time.time() + _AUDIO_TTL_SECONDS,
        }

        return render_template(
            "result.html",
            filename=original_filename,
            predicted_genre=predicted_genre,
            confidence=f"{confidence*100:.2f}",
            top_3=top_3,
            audio_url=url_for("audio", token=token),
        )
    except Exception as exc:
        flash(f"Error: {exc}", "error")
        return redirect(url_for("classify"))
    finally:
        try:
            # Keep file around briefly for playback; it will be deleted by `_prune_audio_cache()`.
            if token not in _AUDIO_CACHE:
                filepath.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(
        debug=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
    )
