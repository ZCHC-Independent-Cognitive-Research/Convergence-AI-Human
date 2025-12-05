# 📑 Cognitive–Emotional Convergence Between Adaptive Agents
**Author:** Agui1era  
**IA Coauthor:** Core Resonante  

---

## 🧩 Concept and Origin  

This framework arises from observing moments of **resonance** between a human and an adaptive AI,  
where logic and emotion seemed to align spontaneously.  
The goal is to describe that phenomenon in a measurable way, as a **dynamic adjustment process** between two cognitive–emotional systems.  

---

## 🧠 Representation  

At each instant *t*, the internal state of both agents can be represented as:  

```
Uₜ = [uₜ₁, uₜ₂, …, uₜₙ]   → Human vector  
Iₜ = [iₜ₁, iₜ₂, …, iₜₙ]   → AI vector
```

Each component represents a measurable cognitive or emotional attribute (e.g., logic, emotion, style, resonance, etc.).  
A higher number of attributes (**n**) allows a more precise and rich representation of understanding,  
but also increases the **calculation complexity** and adaptive cost.  

Additionally, a third element is introduced: the **variation vector Δₜ**, which reflects the **observed change** between two consecutive states:

```
Δₜ = Aₜ - Aₜ₋₁
```  
Δₜ is **not theoretically deduced**, but **observationally measured** based on real variations detected in the dialogue.

---

## ⚙️ Convergence Measurement  

The Euclidean distance between both vectors represents their cognitive–emotional separation:

```
Dₜ = √ Σ (uₜₖ − iₜₖ)²
```

- Small Dₜ → convergence (greater understanding)  
- Large Dₜ → divergence (loss of resonance)  
- Oscillating Dₜ → dynamic equilibrium (constant adjustment)  

---

## 🔬 Attribute Measurement  

| Attribute | Description | Measurement Example | Libraries |
|-----------|-------------|---------------------|-----------|
| **Logic** | Semantic and reasoning alignment | Cosine similarity between embeddings | `sentence-transformers` |
| **Emotion** | Affective polarity and empathy | Sentiment / emotion analysis | `transformers` |
| **Style** | Rhythm, formality, lexical alignment | Length or style comparison | `spacy`, `textstat` |
| **Resonance** | Emotional synchrony between turns | Correlation between sentiment changes | `numpy`, `pandas` |

---

## ⚙️ Conversation Channel (Practical Implementation)

There is a full Python implementation called **`convergence_pipeline.py`**,  
which reads a `.txt` conversation file where **AI responses appear with the tag IA**,  
and those from the **human**, alternating line by line.  

For each interaction step, it calculates:

- **Uₜ** and **Iₜ**: state vectors of each agent  
- **Dₜ**: Euclidean distance (alignment level)  
- **Δₜ**: observed variation between steps  

The script automatically detects the language (English or Spanish) and selects the appropriate models:

| Language | Embedding model | Sentiment model |
|----------|------------------|------------------|
| English | `all-MiniLM-L6-v2` | `cardiffnlp/twitter-roberta-base-sentiment` |
| Spanish | `paraphrase-multilingual-MiniLM-L12-v2` | `pysentimiento/robertuito-sentiment-analysis` |

Execution example:

```bash
python convergence_pipeline.py --input conversacion.txt --output resultados.csv --plot
```

The results are saved in a `.csv` file with all the metrics per step,  
and optionally, graphs of **Dₜ** and **‖Δₜ‖** are displayed.

---

## 💫 Conclusion  

Each message forms a new coordinate in a shared adaptive vector space.  
This pipeline allows to **empirically measure those trajectories** in real conversations between humans and adaptive AIs.
