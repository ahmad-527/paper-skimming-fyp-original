"""Free Community Cloud entry point for the later reading interface."""
from pathlib import Path
import os
import traceback

import streamlit as st
import streamlit.components.v1 as components
from server import Classifier, validate_text

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Paper Skimming", page_icon="📄", layout="wide")
st.markdown("""<style>
.stApp {background:#f5f3ed;}
.block-container {padding:0;max-width:1440px;}
header[data-testid="stHeader"] {background:transparent;}
</style>""", unsafe_allow_html=True)
reading_workspace = components.declare_component("paper_skimming", path=str(ROOT / "static"))


def setting(name, default=""):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


@st.cache_resource
def historical_classifier():
    local_path = setting("PAPER_SKIMMING_MODEL")
    model_repository = setting("PAPER_SKIMMING_MODEL_REPOSITORY")
    token = setting("PAPER_SKIMMING_MODEL_TOKEN")
    classifier = Classifier(local_path)
    if not local_path and not model_repository:
        return classifier
    classifier.state = "loading"
    os.environ.setdefault("TFHUB_CACHE_DIR", str(Path.home() / ".cache" / "paper-skimming" / "tfhub"))

    def prepare():
        try:
            if not local_path:
                from huggingface_hub import hf_hub_download
                classifier.path = Path(hf_hub_download(
                    repo_id=model_repository, filename="historical-model.keras",
                    token=token or False))
            classifier.load()
        except Exception as error:
            classifier.state = "error"
            print(f"Model provisioning failed: {type(error).__name__}", flush=True)

    prepare()
    return classifier


classifier = historical_classifier()


@st.fragment(run_every=2)
def workspace():
    request = reading_workspace(
        status=classifier.status(), response=st.session_state.get("readout_response"),
        thesis_url="https://github.com/ahmad-527/paper-skimming-fyp-original/blob/main/report/final_year_report.pdf",
        key="reading_workspace", default=None)
    if not isinstance(request, dict):
        return
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or len(request_id) > 100:
        return
    previous = st.session_state.get("readout_response", {})
    if previous.get("request_id") == request_id:
        return
    response = {"request_id": request_id}
    try:
        response["text"] = validate_text(request.get("text"))
        response["data"] = classifier.predict(response["text"])
    except (ValueError, RuntimeError) as error:
        response["error"] = str(error)
    except Exception as error:
        # Record code locations only: exception messages may contain submitted text.
        locations = ' > '.join(f'{Path(frame.filename).name}:{frame.lineno}:{frame.name}'
                               for frame in traceback.extract_tb(error.__traceback__))
        print(f'Inference failed: {type(error).__name__}; {locations}', flush=True)
        response["error"] = "The model could not complete this read. Please try again."
    st.session_state["readout_response"] = response
    # The next heartbeat sends the response into the existing component iframe.
    # A full rerun would remount it and discard its pending browser request.


workspace()
