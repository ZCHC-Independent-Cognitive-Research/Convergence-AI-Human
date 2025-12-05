#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Convergence Analyzer with Caching
----------------------------------------
Processes multiple conversation files in batch with performance optimizations:
- Embedding caching to avoid recomputing identical messages
- Batch processing of embeddings
- Parallel file processing (optional)
- Progress tracking

Usage:
  python batch_analyzer.py --input-dir ./conversations/ --output-dir ./results/
  python batch_analyzer.py --files conv1.txt conv2.txt conv3.txt --output-dir ./results/
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib
import pickle

# Import from existing modules
from convergence_pipeline import read_conversation, MODELS
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


class EmbeddingCache:
    """
    Persistent cache for sentence embeddings to avoid recomputation.
    Uses file-based storage with pickle.
    """
    def __init__(self, cache_dir=".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _hash_text(self, model_name, text):
        """Generate unique hash for model + text combination"""
        key = f"{model_name}:{text}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def get(self, model_name, text):
        """Retrieve embedding from cache"""
        text_hash = self._hash_text(model_name, text)

        # Check memory cache first
        if text_hash in self.cache:
            self.hits += 1
            return self.cache[text_hash]

        # Check disk cache
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.cache[text_hash] = embedding
                self.hits += 1
                return embedding
            except Exception:
                pass

        self.misses += 1
        return None

    def set(self, model_name, text, embedding):
        """Store embedding in cache"""
        text_hash = self._hash_text(model_name, text)
        self.cache[text_hash] = embedding

        # Persist to disk
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logging.warning(f"Failed to persist cache: {e}")

    def stats(self):
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


class BatchConvergenceAnalyzer:
    """
    Optimized batch analyzer with caching and batch processing
    """
    def __init__(self, lang="en", use_cache=True, batch_size=32):
        self.lang = lang
        self.batch_size = batch_size
        self.embedder = SentenceTransformer(MODELS[lang]["embed"])
        self.sentiment_pipe = pipeline("sentiment-analysis", model=MODELS[lang]["sent"])

        if use_cache:
            self.cache = EmbeddingCache()
        else:
            self.cache = None

    def encode_with_cache(self, texts, show_progress=False):
        """
        Encode texts with caching support

        Args:
            texts: List of strings to encode
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(MODELS[self.lang]["embed"], text)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)

        # Batch encode uncached texts
        if uncached_texts:
            if show_progress:
                logging.info(f"Encoding {len(uncached_texts)} uncached texts in batches of {self.batch_size}")

            new_embeddings = self.embedder.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            # Store in cache and results
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                if self.cache:
                    self.cache.set(MODELS[self.lang]["embed"], text, emb)
                embeddings.append((idx, emb))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def polarity(self, text):
        """Get sentiment polarity [0,1]"""
        res = self.sentiment_pipe(text[:512], truncation=True)[0]
        label = res.get("label", "").upper()
        if "NEG" in label:
            return 0.0
        if "NEU" in label:
            return 0.5
        if "POS" in label:
            return 1.0
        return float(res.get("score", 0.5))

    def style_similarity(self, h, i):
        """Style proxy: similarity in length (clamped to [0,1])"""
        lh, li = max(len(h.split()), 1), max(len(i.split()), 1)
        max_len = max(lh, li)
        return 1 - abs(lh - li) / max_len

    def analyze_conversation(self, pairs):
        """
        Analyze a single conversation with optimized batch processing

        Args:
            pairs: List of (human, ai) message tuples

        Returns:
            pd.DataFrame with analysis results
        """
        # Batch encode all messages at once
        all_texts = [msg for pair in pairs for msg in pair]
        all_embeddings = self.encode_with_cache(all_texts, show_progress=False)

        # Reshape into human/ai embeddings
        human_embeddings = all_embeddings[0::2]
        ai_embeddings = all_embeddings[1::2]

        # Process each turn
        prev_emo_h, prev_emo_i = 0.5, 0.5
        data = []
        A_list = []

        for t, ((human, ai), emb_h, emb_i) in enumerate(zip(pairs, human_embeddings, ai_embeddings)):
            # Logic: cosine similarity
            logic = float(np.dot(emb_h, emb_i) / (np.linalg.norm(emb_h) * np.linalg.norm(emb_i)))

            # Emotion: sentiment
            emo_h = self.polarity(human)
            emo_i = self.polarity(ai)

            # Style: length similarity
            style = self.style_similarity(human, ai)

            # Resonance: emotional synchrony
            res = 1 - abs((emo_h - prev_emo_h) - (emo_i - prev_emo_i))
            res = max(0, min(1, res))

            # Vectors
            U = np.array([logic, emo_h, style, res])
            I = np.array([logic, emo_i, style, res])
            A = (U + I) / 2
            D = np.linalg.norm(U - I)

            # Delta
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

        return pd.DataFrame(data)


def process_single_file(input_file, output_dir, starts_with="AI", lang_override=None, use_cache=True):
    """
    Process a single conversation file

    Returns:
        dict with results and metadata
    """
    try:
        # Read conversation
        pairs = read_conversation(input_file, starts_with)
        if not pairs:
            return {"status": "error", "file": input_file, "error": "No dialogue pairs found"}

        # Detect language
        if lang_override:
            lang = lang_override.lower()
        else:
            sample_text = " ".join([" ".join(p) for p in pairs[:5]])
            lang_detected = detect(sample_text)
            lang = "es" if "es" in lang_detected else "en"

        # Analyze
        analyzer = BatchConvergenceAnalyzer(lang=lang, use_cache=use_cache)
        df = analyzer.analyze_conversation(pairs)

        # Save results
        output_file = Path(output_dir) / f"{Path(input_file).stem}_analysis.csv"
        df.to_csv(output_file, index=False)

        # Cache stats
        cache_stats = analyzer.cache.stats() if analyzer.cache else {"hits": 0, "misses": 0, "hit_rate": "N/A"}

        return {
            "status": "success",
            "file": input_file,
            "output": str(output_file),
            "turns": len(df),
            "language": lang,
            "final_D_t": float(df["D_t"].iloc[-1]),
            "cache_stats": cache_stats
        }

    except Exception as e:
        return {
            "status": "error",
            "file": input_file,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch Convergence Analyzer with Caching",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", help="Directory containing .txt conversation files")
    input_group.add_argument("--files", nargs="+", help="List of specific files to process")

    parser.add_argument("--output-dir", default="./batch_results", help="Output directory for results")
    parser.add_argument("--starts_with", default="AI", choices=["AI", "HUMAN"], help="Who starts conversations")
    parser.add_argument("--lang", help="Force language (en/es, otherwise auto-detect)")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    parser.add_argument("--parallel", action="store_true", help="Process files in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='[%(levelname)s] %(message)s'
    )

    # Get list of files to process
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {args.input_dir}")
            return
        files = list(input_dir.glob("*.txt"))
    else:
        files = [Path(f) for f in args.files]

    if not files:
        print("No .txt files found to process")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(files)} conversation files...")
    print(f"Output directory: {output_dir}")
    print(f"Cache: {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    print()

    results = []

    if args.parallel:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    str(f),
                    str(output_dir),
                    args.starts_with,
                    args.lang,
                    not args.no_cache
                ): f for f in files
            }

            with tqdm(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

                    if result["status"] == "success":
                        pbar.set_postfix({"Last": Path(result["file"]).name[:20]})
    else:
        # Sequential processing with progress bar
        for file in tqdm(files, desc="Processing files"):
            result = process_single_file(
                str(file),
                str(output_dir),
                args.starts_with,
                args.lang,
                not args.no_cache
            )
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\nSuccessful: {len(successful)} / {len(results)}")
    print(f"Failed: {len(failed)} / {len(results)}")

    if successful:
        print("\n✓ Processed files:")
        for r in successful:
            print(f"  - {Path(r['file']).name}: {r['turns']} turns, D_t={r['final_D_t']:.3f}, lang={r['language']}")
            if r.get("cache_stats"):
                print(f"    Cache: {r['cache_stats']['hit_rate']} hit rate")

    if failed:
        print("\n✗ Failed files:")
        for r in failed:
            print(f"  - {Path(r['file']).name}: {r['error']}")

    # Save summary
    summary_file = output_dir / "batch_summary.csv"
    pd.DataFrame(results).to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
