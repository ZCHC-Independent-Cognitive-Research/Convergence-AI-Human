#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive–Emotional Convergence Pipeline (Bilingual)
----------------------------------------------------
Analyzes the convergence between Human and AI across multiple dimensions:
- Logic: Semantic cosine similarity
- Emotion: Neural sentiment analysis + keyword markers
- Style: Lexical diversity + Clause density (Spacy)
- Resonance: Sentimental alignment

Refactored to OOP. Outputs detailed CSV and plots.
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import spacy
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
MODELS = {
    "en": {
        "embed": "all-MiniLM-L6-v2",
        "sent": "cardiffnlp/twitter-roberta-base-sentiment",
        "spacy": "en_core_web_sm"
    },
    "es": {
        "embed": "paraphrase-multilingual-MiniLM-L12-v2",
        "sent": "pysentimiento/robertuito-sentiment-analysis",
        "spacy": "es_core_news_sm"
    }
}

EMOTION_KEYWORDS_ES = {
    "miedo", "rabia", "odio", "dolor", "triste", "tristeza", "morir", "morirme", 
    "soledad", "vacío", "fracaso", "culpa", "derrumbe", "pérdida", "llorar", 
    "ansiedad", "asfixia", "paz", "feliz", "felicidad", "esperanza", "amor", 
    "calma", "alivio", "consuelo", "ternura"
}
# A simple English set for fallback/expansion
EMOTION_KEYWORDS_EN = {
    "fear", "rage", "hate", "pain", "sad", "sadness", "die", "death", "lonely", 
    "emptiness", "failure", "guilt", "collapse", "loss", "cry", "anxiety", 
    "peace", "happy", "happiness", "hope", "love", "calm", "relief", "comfort"
}

CLAUSE_MARKERS_ES = {",", ";", "y", "e", "o", "u", "pero", "aunque", "porque", "que", "pues", "si"}
CLAUSE_MARKERS_EN = {",", ";", "and", "or", "but", "although", "because", "that", "since", "if"}


class TextAnalyzer:
    """Helper class to extract linguistic metrics using Spacy."""
    def __init__(self, spacy_model_name, emotion_keywords, clause_markers):
        try:
            self.nlp = spacy.load(spacy_model_name)
        except OSError:
            print(f"[WARNING] Spacy model '{spacy_model_name}' not found. Downloading...")
            spacy.cli.download(spacy_model_name)
            self.nlp = spacy.load(spacy_model_name)
            
        self.emotion_keywords = emotion_keywords
        self.clause_markers = clause_markers

    def analyze(self, text):
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc if token.is_alpha]
        total_tokens = len(words) if words else 1
        
        # Lexical Diversity (Type-Token Ratio)
        unique_tokens = set(words)
        lexical_diversity = len(unique_tokens) / total_tokens

        # Clause Density Estimation
        clause_count = sum(1 for w in words if w in self.clause_markers) + 1
        # Normalize roughly by sentence count or length. 
        # Here we return density per 10 tokens to keep it in a manageable range [0, 1] usually
        clause_density = min(1.0, (clause_count / (len(list(doc.sents)) or 1)) / 5.0) 

        # Keyword Emotion Density
        emo_count = sum(1 for w in words if w in self.emotion_keywords)
        emo_density = min(1.0, emo_count / 5.0) # Cap at 1.0 for high density

        return lexical_diversity, clause_density, emo_density


class ConvergenceMonitor:
    def __init__(self, lang="es"):
        self.lang = lang
        print(f"[INIT] Loading models for language: {lang.upper()}...")
        
        # Load Transformer Models
        self.embedder = SentenceTransformer(MODELS[lang]["embed"])
        self.sentiment_pipe = pipeline("sentiment-analysis", model=MODELS[lang]["sent"])
        
        # Load Linguistic Analyzer
        if lang == "es":
            self.analyzer = TextAnalyzer(
                MODELS["es"]["spacy"], EMOTION_KEYWORDS_ES, CLAUSE_MARKERS_ES
            )
        else:
            self.analyzer = TextAnalyzer(
                MODELS["en"]["spacy"], EMOTION_KEYWORDS_EN, CLAUSE_MARKERS_EN
            )
            
        self.prev_emo_h = 0.5
        self.prev_emo_i = 0.5

    def _get_neural_polarity(self, text):
        """Maps model sentiment outputs to [0,1]"""
        try:
            res = self.sentiment_pipe(text[:512], truncation=True)[0]
            label = res.get("label", "").upper()
            score = float(res.get("score", 0.5))
            
            # Normalize to 0-1 range
            if "NEG" in label: return 0.0 + (1-score)*0.1 # Close to 0
            if "NEU" in label: return 0.5
            if "POS" in label: return 1.0 - (1-score)*0.1 # Close to 1
            
            # Fallback for models outputting stars/other labels
            return score 
        except Exception:
            return 0.5

    def compute_step(self, human, ai):
        # 1. Logic (Semantic Similarity)
        emb_h = self.embedder.encode(human, convert_to_tensor=True)
        emb_i = self.embedder.encode(ai, convert_to_tensor=True)
        logic_sim = float(util.cos_sim(emb_h, emb_i))

        # 2. Emotion (Neural + Keyword)
        neu_h = self._get_neural_polarity(human)
        neu_i = self._get_neural_polarity(ai)
        
        lex_h, clause_h, kw_emo_h = self.analyzer.analyze(human)
        lex_i, clause_i, kw_emo_i = self.analyzer.analyze(ai)

        # Composite Emotion Score (Weighted average of Neural and Keyword density influence)
        # We treat keyword density as an intensifier or separate channel? 
        # For simplicity, let's keep the neural polarity as the main "Emotion" dimension for D_t,
        # but record the others as "Style" components.
        emo_h = neu_h
        emo_i = neu_i

        # 3. Style (Composite Vector)
        # We compare style by similarity. 
        # Ideally, we want the RAW values in U and I to compute distance.
        # Let's define Style as a subspace of 2 dimensions: Lexical Diversity and Clause Density.
        
        # 4. Resonance (Derivative of Emotion)
        # 1 - |ΔEmo_h - ΔEmo_i|
        res_val = 1 - abs((emo_h - self.prev_emo_h) - (emo_i - self.prev_emo_i))
        res_val = max(0, min(1, res_val))
        
        self.prev_emo_h = emo_h
        self.prev_emo_i = emo_i

        # Assemble Vectors
        # Dimensions: [Logic, Emotion, LexicalDiv, ClauseDens, Resonance]
        # Re-scaling to ensure all are approx [0,1]
        U = np.array([logic_sim, emo_h, lex_h, clause_h, res_val])
        I = np.array([logic_sim, emo_i, lex_i, clause_i, res_val])
        # Note: Logic is shared (similarity between them), so dist is 0 in that dimension?
        # Actually in the original code: logic was util.cos_sim(h, i).
        # And U_logic = logic, I_logic = logic. So D_logic = 0. 
        # This effectively removes logic from the Distance metric D_t! 
        # It only contributes to the Average A_t.
        # That is a theoretical choice: "Logic convergence is implicit in the similarity score".
        
        # Calculate Convergence Distance
        D = np.linalg.norm(U - I)
        A = (U + I) / 2
        
        return {
            "U": U, "I": I, "A": A, "D": D,
            "metrics": {
                "lex_h": lex_h, "lex_i": lex_i,
                "clause_h": clause_h, "clause_i": clause_i,
                "emo_h": emo_h, "emo_i": emo_i
            }
        }

def read_conversation(path, starts_with="AI"):
    """Reads .txt conversation alternating AI/Human lines"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    cleaned = [re.sub(r"^(AI|Human|User|Yo|Asistente|ChatGPT|Modelo)\s*:\s*", "", l, flags=re.I) for l in lines]
    pairs = []
    start_idx = 0
    
    # Adjust start index if the first line doesn't match expectations
    # But strictly following the toggle is safer.
    
    # If starts with AI: AI (0), Human (1) -> Pair
    if starts_with.upper() == "AI":
        for i in range(0, len(cleaned)-1, 2):
            pairs.append((cleaned[i+1], cleaned[i])) # Human, AI
    else:
        for i in range(0, len(cleaned)-1, 2):
            pairs.append((cleaned[i], cleaned[i+1])) # Human, AI
            
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Cognitive–Emotional Convergence (OOP Refactor)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results_v2.csv")
    parser.add_argument("--starts_with", default="AI", choices=["AI","HUMAN"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Detect Language once
    pairs = read_conversation(args.input, args.starts_with)
    if not pairs:
        print("No pairs found.")
        return

    sample_text = " ".join(pairs[0])
    lang_code = detect(sample_text)
    lang = "es" if "es" in lang_code else "en"
    
    monitor = ConvergenceMonitor(lang=lang)
    
    data = []
    A_list = []
    
    print(f"[RUN] Processing {len(pairs)} interaction steps...")
    
    for t, (human, ai) in enumerate(pairs):
        res = monitor.compute_step(human, ai)
        U, I, A, D = res["U"], res["I"], res["A"], res["D"]
        metrics = res["metrics"]
        
        delta = np.zeros_like(A) if t == 0 else A - A_list[-1]
        A_list.append(A)
        
        row = {
            "t": t,
            "human": human, "ai": ai,
            "D_t": D,
            "‖Δ_t‖": np.linalg.norm(delta),
            # Detailed Breakdown
            "U_logic": U[0], "U_emo": U[1], "U_lex": U[2], "U_clause": U[3], "U_res": U[4],
            "I_logic": I[0], "I_emo": I[1], "I_lex": I[2], "I_clause": I[3], "I_res": I[4],
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"[DONE] Results saved to {args.output}")

    if args.plot:
        plt.figure(figsize=(10,6))
        plt.plot(df["t"], df["D_t"], marker="o", label="Convergence Distance $D_t$")
        plt.plot(df["t"], df["‖Δ_t‖"], marker="x", linestyle="--", label="Variation Magnitude $‖Δ_t‖$")
        plt.title(f"Cognitive-Emotional Convergence ({lang.upper()})")
        plt.xlabel("Interaction Step")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
