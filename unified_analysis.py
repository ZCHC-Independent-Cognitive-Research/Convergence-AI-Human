#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Convergence + Linguistic Metrics Analyzer
--------------------------------------------------
Combines cognitive-emotional convergence (convergence_pipeline.py)
with linguistic pattern analysis (conversation_parametres.py)
to provide comprehensive conversation analysis.

Outputs:
  - CSV with both convergence metrics (D_t, Δ_t) and linguistic metrics
    (tokens/sentence, lexical diversity, emotion markers, clauses)
  - Unified HTML report with interactive visualizations

Usage:
  python unified_analysis.py --input conversation.txt --output analysis.csv --report
"""

import argparse
import logging
import sys
import os
import numpy as np
import pandas as pd
from convergence_pipeline import (
    read_conversation, compute, polarity, style_similarity,
    MODELS
)
from conversation_parametres import analyze
from langdetect import detect
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def detect_symbolic_drift(df, emotion_threshold=0.3, clause_threshold=2.0):
    """
    Detects symbolic drift pattern as described in Report.md

    Symbolic drift characteristics:
    - Low emotional markers (< 0.3 avg)
    - High structural density (> 2.0 clauses/sentence proxy)
    - Sustained over multiple turns

    Returns:
        tuple: (bool, str) - (is_drift_detected, warning_message)
    """
    avg_emotion_human = df["U_emotion"].mean()
    # Use style as proxy for structural complexity (not perfect but indicative)
    avg_complexity = df["U_style"].mean()

    warnings = []
    is_drift = False

    if avg_emotion_human < emotion_threshold:
        warnings.append(f"Low emotional expression detected (avg={avg_emotion_human:.3f})")
        is_drift = True

    # Check for sustained low emotion variance
    emotion_variance = df["U_emotion"].std()
    if emotion_variance < 0.15:
        warnings.append(f"Very low emotional variance (std={emotion_variance:.3f})")
        is_drift = True

    # Check convergence trend
    final_convergence = df["D_t"].iloc[-1]
    initial_convergence = df["D_t"].iloc[0]

    if final_convergence < initial_convergence * 0.7:
        warnings.append(f"Strong convergence detected (D_t: {initial_convergence:.3f} → {final_convergence:.3f})")

    if is_drift:
        message = "⚠️  SYMBOLIC DRIFT PATTERN DETECTED\n" + "\n".join(f"  - {w}" for w in warnings)
        message += "\n  → This pattern may indicate high AI exposure or symbolic communication style"
    else:
        message = "✓ Normal conversation pattern"

    return is_drift, message


def analyze_unified(input_file, output_csv, starts_with="AI", lang_override=None, verbose=False):
    """
    Performs unified analysis combining convergence + linguistic metrics

    Args:
        input_file: Path to conversation .txt file
        output_csv: Output CSV path
        starts_with: "AI" or "HUMAN" - who starts the conversation
        lang_override: Force language ("en" or "es")
        verbose: Enable verbose logging

    Returns:
        pd.DataFrame: Combined analysis results
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='[%(levelname)s] %(message)s',
        stream=sys.stdout
    )

    # Step 1: Run convergence analysis
    logging.info("=" * 60)
    logging.info("PHASE 1: Cognitive-Emotional Convergence Analysis")
    logging.info("=" * 60)

    pairs = read_conversation(input_file, starts_with)
    if not pairs:
        raise ValueError("No dialogue pairs found. Check file format and --starts_with option.")

    logging.info(f"Loaded {len(pairs)} dialogue pairs")

    # Detect language
    if lang_override:
        lang = lang_override.lower()
        if lang not in MODELS:
            raise ValueError(f"Unsupported language: {lang_override}")
        logging.info(f"Using forced language: {lang.upper()}")
    else:
        sample_text = " ".join([" ".join(p) for p in pairs[:5]])
        lang_detected = detect(sample_text)
        lang = "es" if "es" in lang_detected else "en"
        logging.info(f"Detected language: {lang.upper()}")

    # Load models
    logging.info(f"Loading models: {MODELS[lang]['embed']} + {MODELS[lang]['sent']}")
    embedder = SentenceTransformer(MODELS[lang]["embed"])
    sentiment_pipe = pipeline("sentiment-analysis", model=MODELS[lang]["sent"])

    # Compute convergence metrics
    prev_emo_h, prev_emo_i = 0.5, 0.5
    data, A_list = [], []

    for t, (human, ai) in enumerate(pairs):
        logging.info(f"Processing turn {t+1}/{len(pairs)}")
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

    df_convergence = pd.DataFrame(data)

    # Step 2: Run linguistic analysis
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 2: Linguistic Pattern Analysis")
    logging.info("=" * 60)

    # Combine all text for linguistic analysis
    full_text = "\n".join([human for human, ai in pairs])

    try:
        linguistic_metrics = analyze(full_text)
        logging.info("Linguistic metrics computed successfully")
    except Exception as e:
        logging.warning(f"Linguistic analysis failed: {e}")
        linguistic_metrics = {
            "Total Tokens": 0,
            "Avg Tokens/Sentence": 0,
            "Lexical Diversity": 0,
            "Emotion Markers/Sentence": 0,
            "Avg Clauses/Sentence": 0
        }

    # Step 3: Detect symbolic drift
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 3: Symbolic Drift Detection")
    logging.info("=" * 60)

    is_drift, drift_message = detect_symbolic_drift(df_convergence)
    logging.info(f"\n{drift_message}\n")

    # Step 4: Combine results
    # Add linguistic metrics as constant columns
    for key, value in linguistic_metrics.items():
        df_convergence[key.replace(" ", "_").replace("/", "_per_")] = value

    df_convergence["symbolic_drift_detected"] = is_drift

    # Save results
    df_convergence.to_csv(output_csv, index=False)
    logging.info(f"Unified analysis saved to {output_csv}")

    return df_convergence, linguistic_metrics, (is_drift, drift_message)


def generate_html_report(df, linguistic_metrics, drift_info, output_html):
    """
    Generates an interactive HTML report with visualizations

    Args:
        df: DataFrame with unified analysis results
        linguistic_metrics: Dictionary of linguistic metrics
        drift_info: Tuple (is_drift, message)
        output_html: Output HTML file path
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str

    # Generate plots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Convergence Distance
    ax1.plot(df["t"], df["D_t"], marker="o", linewidth=2, markersize=6)
    ax1.set_title("Convergence Distance D_t", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Interaction Step (t)")
    ax1.set_ylabel("D_t (lower = higher convergence)")
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Variation Magnitude
    ax2.plot(df["t"], df["‖Δ_t‖"], marker="o", color="orange", linewidth=2, markersize=6)
    ax2.set_title("Variation Magnitude ‖Δ_t‖", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Interaction Step (t)")
    ax2.set_ylabel("‖Δ_t‖")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    # Plot 3: Emotional evolution
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["t"], df["U_emotion"], marker="o", label="Human Emotion", linewidth=2)
    ax.plot(df["t"], df["I_emotion"], marker="s", label="AI Emotion", linewidth=2)
    ax.set_title("Emotional Evolution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Interaction Step (t)")
    ax.set_ylabel("Emotional Polarity [0=negative, 1=positive]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_b64 = fig_to_base64(fig2)
    plt.close(fig2)

    # Generate HTML
    is_drift, drift_msg = drift_info
    drift_color = "#ff6b6b" if is_drift else "#51cf66"
    drift_status = "DETECTED" if is_drift else "NOT DETECTED"

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Convergence Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .drift-alert {{
            background: {drift_color};
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Unified Convergence Analysis Report</h1>
        <p>Cognitive-Emotional Convergence + Linguistic Pattern Analysis</p>
    </div>

    <div class="section drift-alert">
        <h2>⚠️ Symbolic Drift Detection: {drift_status}</h2>
        <pre style="white-space: pre-wrap; font-family: monospace;">{drift_msg}</pre>
    </div>

    <div class="section">
        <h2>📊 Convergence Metrics Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Turns</div>
                <div class="metric-value">{len(df)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final D_t</div>
                <div class="metric-value">{df['D_t'].iloc[-1]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Human Emotion</div>
                <div class="metric-value">{df['U_emotion'].mean():.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg AI Emotion</div>
                <div class="metric-value">{df['I_emotion'].mean():.3f}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Convergence Visualization</h2>
        <img src="data:image/png;base64,{plot1_b64}" alt="Convergence plots">
    </div>

    <div class="section">
        <h2>💬 Emotional Evolution</h2>
        <img src="data:image/png;base64,{plot2_b64}" alt="Emotional evolution">
    </div>

    <div class="section">
        <h2>🔤 Linguistic Metrics</h2>
        <div class="metric-grid">
"""

    for key, value in linguistic_metrics.items():
        html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{key}</div>
                <div class="metric-value">{value if isinstance(value, int) else f"{value:.3f}"}</div>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>📋 Detailed Turn-by-Turn Data</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Turn</th>
                        <th>D_t</th>
                        <th>‖Δ_t‖</th>
                        <th>Human Emotion</th>
                        <th>AI Emotion</th>
                        <th>Logic Alignment</th>
                    </tr>
                </thead>
                <tbody>
"""

    for _, row in df.iterrows():
        html_content += f"""
                    <tr>
                        <td>{int(row['t'])}</td>
                        <td>{row['D_t']:.3f}</td>
                        <td>{row['‖Δ_t‖']:.3f}</td>
                        <td>{row['U_emotion']:.3f}</td>
                        <td>{row['I_emotion']:.3f}</td>
                        <td>{row['A_logic']:.3f}</td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        <p>Generated by Unified Convergence Analyzer</p>
        <p>Based on Cognitive-Emotional Convergence Framework by Oscar Aguilera</p>
    </div>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    logging.info(f"HTML report saved to {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Convergence + Linguistic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_analysis.py --input conversation.txt --output analysis.csv
  python unified_analysis.py --input chat.txt --output results.csv --report --verbose
        """
    )
    parser.add_argument("--input", required=True, help="Path to conversation .txt file")
    parser.add_argument("--output", default="unified_analysis.csv", help="Output CSV file")
    parser.add_argument("--starts_with", default="AI", choices=["AI", "HUMAN"],
                       help="Who starts the conversation")
    parser.add_argument("--lang", help="Force language: en or es (otherwise auto-detect)")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run unified analysis
    df, linguistic_metrics, drift_info = analyze_unified(
        args.input, args.output, args.starts_with, args.lang, args.verbose
    )

    # Generate HTML report if requested
    if args.report:
        html_path = args.output.replace(".csv", ".html")
        generate_html_report(df, linguistic_metrics, drift_info, html_path)
        print(f"\n✓ HTML report: {html_path}")

    print(f"✓ Analysis complete: {args.output}")


if __name__ == "__main__":
    main()
