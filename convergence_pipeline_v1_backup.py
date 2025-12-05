#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive–Emotional Convergence Pipeline (Bilingual)
----------------------------------------------------
Reads a conversation .txt (AI left, Human right, alternating lines),
detects language (English/Spanish), and uses appropriate
embedding and sentiment models.

Outputs:
  - CSV with U_t, I_t, A_t, D_t, Δ_t per step
  - Optional plots for D_t and |Δ_t|

Usage:
  python convergence_pipeline.py --input conversation.txt --output results.csv --plot
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Models for English and Spanish
MODELS = {
    "en": {
        "embed": "all-MiniLM-L6-v2",
        "sent": "cardiffnlp/twitter-roberta-base-sentiment"
    },
    "es": {
        "embed": "paraphrase-multilingual-MiniLM-L12-v2",
        "sent": "pysentimiento/robertuito-sentiment-analysis"
    }
}

def read_conversation(path, starts_with="AI"):
    """Reads .txt conversation alternating AI/Human lines"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    cleaned = [re.sub(r"^(AI|Human|User|Yo|Asistente|ChatGPT|Modelo)\s*:\s*", "", l, flags=re.I) for l in lines]
    pairs = []
    if starts_with.upper() == "AI":
        for i in range(0, len(cleaned)-1, 2):
            pairs.append((cleaned[i+1], cleaned[i]))
    else:
        for i in range(0, len(cleaned)-1, 2):
            pairs.append((cleaned[i], cleaned[i+1]))
    return pairs

def polarity(sentiment_pipe, text):
    """Maps model sentiment outputs to [0,1]"""
    res = sentiment_pipe(text[:512])[0]
    label = res.get("label", "").upper()
    if "NEG" in label: return 0.0
    if "NEU" in label: return 0.5
    if "POS" in label: return 1.0
    return float(res.get("score", 0.5))

def style_similarity(h, i):
    """Style proxy: similarity in length"""
    lh, li = max(len(h.split()),1), max(len(i.split()),1)
    return 1 - abs(lh - li) / lh

def compute(embedder, sent_pipe, human, ai, prev_emo_h, prev_emo_i):
    """Computes attributes for a step"""
    emb_h = embedder.encode(human, convert_to_tensor=True)
    emb_i = embedder.encode(ai, convert_to_tensor=True)
    logic = float(util.cos_sim(emb_h, emb_i))
    emo_h, emo_i = polarity(sent_pipe, human), polarity(sent_pipe, ai)
    style = style_similarity(human, ai)
    res = 1 - abs((emo_h - prev_emo_h) - (emo_i - prev_emo_i))
    res = max(0, min(1, res))
    U = np.array([logic, emo_h, style, res])
    I = np.array([logic, emo_i, style, res])
    A = (U + I) / 2
    D = np.linalg.norm(U - I)
    return U, I, A, D, emo_h, emo_i

def run(input_file, output_csv, starts_with, plot):
    pairs = read_conversation(input_file, starts_with)
    if not pairs: 
        raise ValueError("No dialogue pairs found. Check file format and --starts_with option.")

    lang_detected = detect(" ".join(pairs[0]))
    lang = "es" if "es" in lang_detected else "en"
    print(f"[INFO] Detected language: {lang.upper()}")

    embedder = SentenceTransformer(MODELS[lang]["embed"])
    sentiment_pipe = pipeline("sentiment-analysis", model=MODELS[lang]["sent"])

    prev_emo_h, prev_emo_i = 0.5, 0.5
    data, A_list = [], []

    for t, (human, ai) in enumerate(pairs):
        U, I, A, D, emo_h, emo_i = compute(embedder, sentiment_pipe, human, ai, prev_emo_h, prev_emo_i)
        delta = np.zeros_like(A) if t == 0 else A - A_list[-1]
        A_list.append(A)
        data.append({
            "t": t,
            "human": human,
            "ai": ai,
            "U_logic": U[0], "U_emotion": U[1], "U_style": U[2], "U_resonance": U[3],
            "I_logic": I[0], "I_emotion": I[1], "I_style": I[2], "I_resonance": I[3],
            "A_logic": A[0], "A_emotion": A[1], "A_style": A[2], "A_resonance": A[3],
            "D_t": D,
            "Δ_logic": delta[0], "Δ_emotion": delta[1], "Δ_style": delta[2], "Δ_resonance": delta[3],
            "‖Δ_t‖": np.linalg.norm(delta)
        })
        prev_emo_h, prev_emo_i = emo_h, emo_i

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Saved results to {output_csv}")

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(df["t"], df["D_t"], marker="o")
        plt.title("Convergence Distance D_t (lower = higher convergence)")
        plt.xlabel("Step"); plt.ylabel("D_t"); plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(df["t"], df["‖Δ_t‖"], marker="o", color="orange")
        plt.title("Variation Magnitude ‖Δ_t‖")
        plt.xlabel("Step"); plt.ylabel("‖Δ_t‖"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive–Emotional Convergence (Bilingual pipeline)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--starts_with", default="AI", choices=["AI","HUMAN"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, args.starts_with, args.plot)
