# Paper Skimming interface

[**Open the free public app**](https://paper-skimming-fyp.streamlit.app/). Live historical-model inference was verified on Streamlit Community Cloud on 9 October 2026. The free host may sleep when inactive, and the first model load can take a few minutes.

Added **9 October 2026**, after the original undergraduate project was completed. This is a presentation and inference layer around a historical model, not a new training experiment or the separate rebuilt version. The archived notebooks and their results are unchanged.

The interface supports pasted abstracts, five role filters, sentence-order/grouped reading, inspection of all five model scores, JSON/Markdown exports, and a thesis PDF. It includes a clearly labelled recorded example, extracted from the unchanged 20k notebook, for use while the model loads. Live submissions use the real checkpoint; no fallback heuristic generates predictions.

## Run locally

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r interface\requirements.txt
$env:PAPER_SKIMMING_MODEL = 'D:\path\outside-the-repository\historical-model.keras'
$env:TFHUB_CACHE_DIR = 'D:\path\outside-the-repository\tfhub-cache'
.\.venv\Scripts\python.exe interface\server.py
```

Open **http://127.0.0.1:3000**. Keep the terminal running. The first model load downloads Universal Sentence Encoder v4 from TensorFlow Hub and may take several minutes; subsequent starts use the cache. No training dataset is required. Dependencies are pinned for this later inference layer, not claimed as the exact original Colab environment.

Without `PAPER_SKIMMING_MODEL`, the app still opens and offers the **recorded example only**. Arbitrary pasted text requires the checkpoint. Weights are deliberately excluded from this repository; readers need a separately supplied authorized checkpoint or a hosted instance. The PDF route expects `report/final_year_report.pdf`.

## Historical checkpoint and inference contract

The authorized `.keras` file records Keras 3.5.0 and a save date of **7 December 2024**. Its SHA-256 is `6f1de7d7c1251143cbfdc57a1b5058d7ed54beb8c4c734c2a7bcb7f9281fbeb9`. The loader checks this hash before deserializing it with Keras safe mode and without the saved optimizer. It has the original tribrid input/output shapes and character vocabulary.

Its exact relationship to the thesis evaluation runs is unverified. A live run of the saved eight-sentence example differs from the notebook's recorded probabilities and one predicted role. The interface therefore separates **live historical-checkpoint predictions** from **recorded notebook predictions** and does not attach the thesis accuracy/F1 values to live results.

The inference path retains the notebook's blank English spaCy sentencizer, original-case sentences, space-separated characters, zero-based sentence positions, and `total_lines = sentence_count - 1`. One-hot depths remain 15 and 20. Positions beyond those ranges produce zero vectors, as in the original code; the interface shows a note for long abstracts. There is no new number replacement, lowercasing, label smoothing at inference, training, or calibration. Softmax indices use the notebook's alphabetical `LabelEncoder` order: BACKGROUND, CONCLUSIONS, METHODS, OBJECTIVE, RESULTS.

The loader traces inference once into a TensorFlow concrete function before serving requests. This prevents Streamlit's Keras cleanup between page runs from interrupting another visitor's prediction. The weights, architecture and preprocessing are unchanged; graph and original eager predictions agree within an absolute probability tolerance of `1e-5` on the historical example, with identical predicted labels.

The training domain is biomedical randomized controlled trial abstracts. Reviews and other paper types may receive less reliable labels. Paste the abstract body without keywords or publisher notices; the app preserves submitted wording and does not silently remove these lines. Successful execution on arbitrary text does not establish classification accuracy on that text.

## Deployment

### Free Streamlit Community Cloud

Deploy the public repository's `main` branch with **`interface/streamlit_app.py`** as the entry point and **Python 3.12**. Community Cloud reads `interface/requirements.txt`. This adapter serves the same reading workspace and calls the same historical classifier; it does not change the model or its preprocessing.

The checkpoint is supplied separately. An authorized owner can store `historical-model.keras` in a **private Hugging Face model repository** and configure these values in Streamlit's secrets settings:

```toml
PAPER_SKIMMING_MODEL_REPOSITORY = "your-account/your-private-model"
PAPER_SKIMMING_MODEL_TOKEN = "your-read-only-token"
```

Keep actual tokens in host secrets, outside Git. Give the token read access only to that private model. The verified checkpoint hash is still enforced. Without model settings, only the clearly labelled recorded example is available. The thesis link opens the public, cleaned PDF on GitHub.

Hosted JSON and Markdown exports use Streamlit's in-memory HTTP download service, with links maintained for the active session. They include every sentence and its scores even when a role filter is selected. Input and export contents are held in session memory and are not logged or saved to host disk. The adapter uses the pinned Streamlit 1.50 file manager; an upgrade needs the download lifecycle checks repeated.

The service is free for personal/educational apps and may sleep when inactive. Its shared CPU and memory limits must accommodate TensorFlow and USE; a successful local run does not guarantee a cloud deployment will fit. No paid upgrade is needed to use this entry point. The model storage provider currently includes 100 GB of private storage for free accounts; this checkpoint is approximately 1.7 MB. The encoder downloads separately into the host's cache.

For local verification of the same hosting entry point after setting `PAPER_SKIMMING_MODEL`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run interface\streamlit_app.py
```

### Standalone Python or Docker

The frontend and Python API run together on one host. The Docker build context is the repository root:

```sh
docker build -f interface/Dockerfile -t paper-skimming .
docker run --rm -p 3000:3000 \
  -e PAPER_SKIMMING_MODEL=/model/historical-model.keras \
  -v /absolute/external/model-folder:/model:ro \
  -v /absolute/external/tfhub-cache:/cache/tfhub \
  paper-skimming
```

The image does not contain the checkpoint. Mount it at runtime from storage outside the repository. Public hosting needs a persistent Python/TensorFlow service and separately provisioned model storage. The first encoder download is large; allow several GB of disk and sufficient RAM for TensorFlow and USE. A static-page host alone cannot run this model. The Docker recipe is provided for deployment; it has not been exercised in the Windows verification environment.

The local server binds loopback by default; `--host`/`--port` or `PAPER_SKIMMING_HOST`/`PORT` configure a host deployment. Use HTTPS through the hosting platform's proxy. Requests are bounded to 20,000 characters and 80 sentences, inference is serialized, and cross-origin browser submissions are rejected. Input text is held in memory for processing and is not logged, uploaded to a third-party inference API, or persisted by this app. TensorFlow Hub is contacted for the encoder download during setup. Host-level proxy/logging policies are separate.

## Validation

Verified locally with the pinned Python runtime: the checkpoint loads, produces five probabilities per sentence, and processes the eight-sentence example through the API. Input rejection, same-origin handling, static-file boundaries, label mapping and the original out-of-range position behaviour were checked. The hosted app loaded the private checkpoint and classified the eight-sentence example with the same labels and displayed scores as the local runtime. Public access was confirmed in the host's sharing settings. The original notebooks were not executed. Interface additions do not establish new benchmark scores.

The regression suite covers JSON/Markdown preservation and provenance, simultaneous Keras cleanup and inference, repeated caller threads, recovery after invalid input or model failure, busy requests, invalid output shape, Unicode and pasted headings, 1–80 sentences, the character limit, and graph/eager agreement. Run it after setting the external model and encoder cache as above:

```powershell
.\.venv\Scripts\python.exe interface\test_inference.py -v
```

The suite skips when no external checkpoint is configured. Weights and user-supplied regression abstracts are not included in Git.
