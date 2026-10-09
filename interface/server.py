"""Later-added reading interface for the unchanged historical classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent
LABELS = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
CHECKPOINT_SHA256 = "6f1de7d7c1251143cbfdc57a1b5058d7ed54beb8c4c734c2a7bcb7f9281fbeb9"
MAX_CHARACTERS = 20_000
MAX_SENTENCES = 80


def validate_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paste an abstract before analysing it.")
    if len(value) > MAX_CHARACTERS:
        raise ValueError("Use an abstract of 20,000 characters or fewer.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("The abstract contains an invalid Unicode character. Paste plain text and try again.") from None
    return value.strip()


def position_features(count):
    # Match the original zero-based positions. Out-of-range one-hot rows are zero.
    lines = [[int(j == i) for j in range(15)] for i in range(count)]
    totals = [[int(j == count - 1) for j in range(20)] for _ in range(count)]
    return lines, totals


class Classifier:
    def __init__(self, model_path):
        self.path = Path(model_path).expanduser() if model_path else None
        self.lock = threading.Lock()
        self.state = "loading" if self.path else "unconfigured"
        self.model = self.tf = self.nlp = self.inference = None

    def status(self):
        return {"state": self.state, "checkpoint": "Historical tribrid · 7 December 2024",
                "interface_added": "9 October 2026", "stores_submissions": False}

    def load(self):
        try:
            if not self.path or not self.path.is_file():
                raise ValueError("Set PAPER_SKIMMING_MODEL to the historical .keras file.")
            with self.path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != CHECKPOINT_SHA256:
                raise ValueError("Checkpoint does not match the verified historical artifact.")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            import tensorflow_hub as hub
            from spacy.lang.en import English

            tf.config.optimizer.set_jit(False)

            class UniversalSentenceEncoderLayer(tf.keras.layers.Layer):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.use_layer = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

                def call(self, inputs):
                    return self.use_layer(inputs)

                def compute_output_shape(self, input_shape):
                    return (input_shape[0], 512)

                def get_config(self):
                    return super().get_config()

            self.model = tf.keras.models.load_model(
                str(self.path), custom_objects={"UniversalSentenceEncoderLayer": UniversalSentenceEncoderLayer},
                compile=False, safe_mode=True)
            shapes = [tuple(t.shape) for t in self.model.inputs]
            if shapes != [(None, 15), (None, 20), (None,), (None, 1)] or self.model.output_shape != (None, 5):
                raise ValueError("Unexpected checkpoint input/output contract.")
            self.tf = tf
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
            # Trace once before serving visitors. Streamlit clears Keras' Python
            # state between runs; a concrete graph avoids that shared state at inference.
            signature = [[tf.TensorSpec((None, 15), tf.int32), tf.TensorSpec((None, 20), tf.int32),
                          tf.TensorSpec((None,), tf.string), tf.TensorSpec((None, 1), tf.string)]]

            @tf.function(input_signature=signature, autograph=False, jit_compile=False)
            def infer(inputs):
                return self.model(inputs, training=False)

            self.inference = infer.get_concrete_function()
            self._predict(["A research abstract."])
            self.state = "ready"
            print("Historical classifier ready.", flush=True)
        except Exception as error:
            self.state = "error"
            # Diagnostics remain on the host; the API never exposes local file paths.
            print(f"Classifier setup failed: {type(error).__name__}: {error}", flush=True)
            import traceback
            traceback.print_exc()

    def _predict(self, sentences):
        tf = self.tf
        lines, totals = position_features(len(sentences))
        inputs = [tf.constant(lines, dtype=tf.int32), tf.constant(totals, dtype=tf.int32),
                  tf.constant(sentences), tf.constant([[" ".join(list(s))] for s in sentences])]
        return self.inference(inputs).numpy()

    def predict(self, text):
        text = validate_text(text)
        if self.state != "ready":
            raise RuntimeError("The historical model is not ready. Try the recorded example while it loads.")
        # Keep the notebook's spaCy sentencizer and original-case inference inputs.
        sentences = [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]
        if len(sentences) > MAX_SENTENCES:
            raise ValueError("Use an abstract of 80 sentences or fewer.")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("The model is processing another abstract. Please try again shortly.")
        try:
            probabilities = self._predict(sentences)
        finally:
            self.lock.release()
        if probabilities.shape != (len(sentences), 5):
            raise RuntimeError("The model returned an invalid prediction.")
        rows = []
        for i, (sentence, scores) in enumerate(zip(sentences, probabilities)):
            scores = [float(x) for x in scores]
            if len(scores) != 5 or not all(0 <= x <= 1 for x in scores) or abs(sum(scores) - 1) > 1e-4:
                raise RuntimeError("The model returned an invalid prediction.")
            rows.append({"index": i, "text": sentence, "label": LABELS[max(range(5), key=scores.__getitem__)],
                         "scores": dict(zip(LABELS, scores))})
        return {"mode": "live", "model": self.status()["checkpoint"], "sentences": rows,
                "position_note": len(sentences) > 15, "text": text}


class Handler(BaseHTTPRequestHandler):
    classifier: Classifier
    static_files = {"/": ROOT / "static/index.html", "/app.css": ROOT / "static/app.css",
                    "/app.js": ROOT / "static/app.js", "/sample.json": ROOT / "static/sample.json",
                    "/report.pdf": ARCHIVE / "report/final_year_report.pdf"}

    def log_message(self, *args):
        pass  # Abstracts and requests are not logged or written to disk.

    def send_json(self, status, data):
        self.send_bytes(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def send_bytes(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return self.send_json(200, self.classifier.status())
        file = self.static_files.get(path)
        if file and file.is_file():
            content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            if file.suffix == ".js":
                content_type = "text/javascript"
            return self.send_bytes(200, file.read_bytes(), content_type)
        self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if self.path != "/api/predict":
            return self.send_json(404, {"error": "Page not found."})
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.send_json(403, {"error": "Use this app's own page to submit an abstract."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 120_000:
                return self.send_json(413, {"error": "The abstract is too large."})
            self.connection.settimeout(20)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Submit an abstract as a JSON object.")
            result = self.classifier.predict(data.get("text"))
            self.send_json(200, result)
        except (ValueError, UnicodeDecodeError) as error:
            self.send_json(400, {"error": str(error) if not isinstance(error, json.JSONDecodeError) else "Invalid JSON request."})
        except RuntimeError as error:
            self.send_json(503, {"error": str(error)})
        except Exception:
            self.send_json(500, {"error": "Classification failed. Please try again."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("PAPER_SKIMMING_MODEL"))
    parser.add_argument("--host", default=os.environ.get("PAPER_SKIMMING_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3000")))
    args = parser.parse_args()
    Handler.classifier = Classifier(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    if args.model:
        threading.Thread(target=Handler.classifier.load, daemon=True).start()
    print(f"Paper Skimming: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
