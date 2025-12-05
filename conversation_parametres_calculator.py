import argparse
import os
import spacy

emotion_words = {
    "miedo", "rabia", "odio", "dolor", "triste", "tristeza", "morir", "morirme", "desaparecer",
    "soledad", "vacío", "fracaso", "culpa", "vergüenza", "colapso", "sufrimiento", "abandono", "quebrarse",
    "derrumbe", "pérdida", "llorar", "infierno", "nada", "fracturarme", "resistir", "agonía", "hundirme",
    "saturado", "agotado", "saturación", "indiferencia", "desconexión", "quebrado", "ansiedad", "asfixia",
    "paz", "feliz", "felicidad", "esperanza", "renacer", "claridad", "dignidad", "valor", "amor",
    "resiliencia", "luz", "honor", "sentido", "tregua", "calma", "alivio", "consuelo", "presencia",
    "quietud", "descanso", "permanecer", "acompañado", "abrazo", "sostén", "ternura", "vivir",
}

clause_markers = {
    ",", ";", "y", "e", "o", "u", "pero", "aunque", "porque", "que", "pues", "como", "cuando", "donde", "mientras",
    "si", "ya que", "sin embargo", "entonces", "además", "incluso",
}


def analyze(text):
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(text)

    total_tokens = 0
    total_sentences = 0
    emotion_count = 0
    clause_estimate = 0
    unique_tokens = set()

    for sent in doc.sents:
        total_sentences += 1
        words = [token.text.lower() for token in sent if token.is_alpha]
        total_tokens += len(words)
        unique_tokens.update(words)
        emotion_count += sum(1 for word in words if word in emotion_words)

        sentence_tokens = [token.text.lower() for token in sent]
        clause_estimate += sum(1 for tok in sentence_tokens if tok in clause_markers) + 1

    tokens_per_sentence = total_tokens / total_sentences if total_sentences else 0
    lexical_diversity = len(unique_tokens) / total_tokens if total_tokens else 0
    emotion_per_sentence = emotion_count / total_sentences if total_sentences else 0
    clauses_per_sentence = clause_estimate / total_sentences if total_sentences else 0

    return {
        "Total Tokens": total_tokens,
        "Avg Tokens/Sentence": tokens_per_sentence,
        "Lexical Diversity": lexical_diversity,
        "Emotion Markers/Sentence": emotion_per_sentence,
        "Avg Clauses/Sentence": clauses_per_sentence,
    }


def main():
    parser = argparse.ArgumentParser(description="Spanish conversation metrics")
    parser.add_argument("--input", default="session.txt", help="Ruta del archivo de conversación (txt)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No se encontró el archivo: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    metrics = analyze(text)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
