"""Serialize complete readouts for session-scoped downloads."""
import json

INTERFACE_ADDED = "9 October 2026"
INTENDED_DOMAIN = "Biomedical randomized controlled trial abstracts"


def readout_exports(readout):
    """Keep every sentence and score, independently of the displayed role filter."""
    data = {**readout, "interface_added": INTERFACE_ADDED,
            "scores_are_calibrated": False, "intended_domain": INTENDED_DOMAIN}
    encoded_json = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    source = "Preserved notebook example" if readout["mode"] == "recorded" else readout["model"]
    markdown = (f"# Paper Skimming readout\n\nSource: {source}\n"
                f"Interface added: {INTERFACE_ADDED}, after original project completion.\n"
                f"Intended domain: {INTENDED_DOMAIN}.\nScores are not calibrated certainty.\n\n")
    markdown += "\n".join(f"## {row['index'] + 1}. {row['label']}\n\n{row['text']}\n\n"
                          f"Model score: {row['scores'][row['label']] * 100:.1f}%\n"
                          for row in readout["sentences"])
    return {"json": encoded_json, "md": markdown.encode("utf-8")}
