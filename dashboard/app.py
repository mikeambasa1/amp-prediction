from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
GROUPS = {"hydrophobic": "AVILMFWC", "polar": "STNQ", "positive": "KR",
          "negative": "DE", "small": "GAS"}


def hydrophobic_moment(seq, angle=100):
    theta = np.radians(angle) * np.arange(len(seq))
    h = np.array([KD[a] for a in seq])
    return np.hypot((h * np.cos(theta)).sum(), (h * np.sin(theta)).sum()) / len(seq)


def compact_features(seq):
    n = len(seq)
    pa = ProteinAnalysis(seq)
    f = {f"frac_{a}": seq.count(a) / n for a in AA}
    f["charge_pH7"] = pa.charge_at_pH(7.4)
    f["pI"] = pa.isoelectric_point()
    f["aromaticity"] = pa.aromaticity()
    f["instability"] = pa.instability_index()
    f["mean_hydropathy"] = pa.gravy()
    f["hydrophobic_moment"] = hydrophobic_moment(seq)
    for g, letters in GROUPS.items():
        f[f"grp_{g}"] = sum(seq.count(a) for a in letters) / n
    return f


@st.cache_resource
def load_model():
    path = Path(__file__).resolve().parent.parent / "models" / "amp_rf_compact.joblib"
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


st.title("Antimicrobial peptide predictor")
st.write("Paste a peptide sequence (one-letter amino-acid codes) to estimate "
         "the probability that it is antimicrobial.")

raw = st.text_input("Peptide sequence", placeholder="e.g. GLSRLFTALK")

if raw:
    seq = "".join(raw.split()).upper()
    if seq.startswith("M"):
        seq = seq[1:]
        st.caption("Leading M removed, because the model was trained without it.")

    bad = sorted(set(seq) - set(AA))
    if bad:
        st.error(f"Unsupported characters: {', '.join(bad)}. Use only the 20 standard amino acids.")
    elif len(seq) < 5:
        st.error("Sequence is too short (minimum 5 amino acids).")
    else:
        if not 10 <= len(seq) <= 50:
            st.warning("The model was trained on peptides of 10 to 50 amino acids, "
                       "so this prediction is less reliable.")
        model, features = load_model()
        feats = compact_features(seq)
        X = pd.DataFrame([feats])[features]
        prob = model.predict_proba(X)[0, 1]

        st.metric("Probability of being an AMP", f"{prob:.1%}")
        st.progress(float(prob))
        st.write("**Prediction:** " + ("AMP" if prob >= 0.5 else "not an AMP"))

        st.subheader("Key properties")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Length", len(seq))
        c2.metric("Net charge (pH 7.4)", f"{feats['charge_pH7']:.2f}")
        c3.metric("pI", f"{feats['pI']:.2f}")
        c4.metric("Hydrophobic moment", f"{feats['hydrophobic_moment']:.2f}")

st.caption("Random forest trained on DRAMP AMPs and non-AMP peptides, "
           "test ROC AUC 0.93 on unseen sequence clusters. A research tool, not a diagnostic.")