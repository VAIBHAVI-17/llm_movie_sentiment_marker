import streamlit as st
import pandas as pd
import time
from sentiment_llm import analyze_review

st.set_page_config(page_title="Movie LLM Sentiment Marker", layout="centered")
st.title("🎬 Movie LLM Sentiment Marker")

# --- in-memory cache keyed by (text, strict_flag, temperature) ---
_cache = {}

def cached_analyze(text: str, strict: bool, temperature: float):
    key = (text.strip(), bool(strict), float(temperature))
    if key not in _cache:
        _cache[key] = analyze_review(text, strict=strict, temperature=temperature)
    return _cache[key]

# ---------------- UI Controls ----------------
strict_mode = st.radio("Interpretation Mode:", ["Strict", "Lenient"]) == "Strict"
mode = st.radio("Choose input:", ["Single Review", "Dataset (CSV)"])

# ---------------- SINGLE REVIEW MODE ----------------
if mode == "Single Review":
    review = st.text_area("Enter a movie review:", height=180)
    if st.button("Analyze"):
        if not review.strip():
            st.warning("⚠️ Please enter a review.")
        else:
            start = time.time()
            try:
                # creative mode → high temp
                result = cached_analyze(review, strict=strict_mode, temperature=0.9)
                st.markdown(f"## 🎯 Label: **{result['label']}**")
                st.markdown(f"**Confidence:** {result['confidence']:.2f}")
                st.markdown(f"**Rationale:** {result['explanation']}")
                if result.get("evidence_phrases"):
                    st.markdown("**Evidence phrases:**")
                    st.write(result["evidence_phrases"])
                with st.expander("🔍 Full JSON"):
                    st.json(result)
            except Exception as e:
                st.error(f"Error analyzing review: {e}")
            finally:
                st.info(f"⏱ {time.time() - start:.2f}s")

# ---------------- DATASET MODE ----------------
else:
    st.write("Upload a CSV with columns: review_id, review_text, sentiment")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        required = {"review_id", "review_text", "sentiment"}
        if not required.issubset(df.columns):
            st.error(f"CSV must contain columns: {required}")
        else:
            batch_size = st.number_input("Batch size", min_value=5, max_value=40, value=5, step=5)

            if st.button("Analyze Dataset"):
                start = time.time()
                preds, confidences, explanations, evidence = [], [], [], []

                total = len(df)
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    for start_idx in range(0, total, batch_size):
                        batch = df.iloc[start_idx:start_idx + batch_size]
                        for idx, row in batch.iterrows():
                            review_text = str(row["review_text"]) if pd.notna(row["review_text"]) else ""
                            if not review_text.strip():
                                raise ValueError(f"Empty review_text at row {idx}, review_id={row.get('review_id')}")

                            res = cached_analyze(review_text, strict=strict_mode, temperature=0.2)

                            preds.append(res["label"])
                            confidences.append(res.get("confidence", 0.0))
                            explanations.append(res.get("explanation", ""))
                            evidence.append("|".join(res.get("evidence_phrases", [])))

                            status_text.text(f"Processed {idx+1}/{total} reviews")
                            progress_bar.progress((idx+1) / total)
                            time.sleep(4.5)

                    df["predicted_sentiment"] = preds
                    df["predicted_confidence"] = confidences
                    df["predicted_explanation"] = explanations
                    df["predicted_evidence"] = evidence

                    # case-normalized accuracy
                    ground = df["sentiment"].astype(str).str.strip().str.lower()
                    pred = df["predicted_sentiment"].astype(str).str.strip().str.lower()
                    accuracy = (ground == pred).mean() * 100.0

                    # --- Per-class counts ---
                    actual_counts = ground.value_counts().rename("Actual")
                    predicted_counts = pred.value_counts().rename("Predicted")

                    # Combine into one table
                    counts_df = pd.concat([actual_counts, predicted_counts], axis=1).fillna(0).astype(int)

                    elapsed = time.time() - start
                    st.success(f"✅ Dataset processed in {elapsed:.2f}s")
                    st.write(f"📊 Accuracy: **{accuracy:.2f}%**")

                    # Show per-class counts table
                    st.markdown("### 📊 Per-Class Counts (Actual vs Predicted)")
                    st.table(counts_df)

                    # preview + download
                    st.dataframe(df.head(5))
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download results", data=csv, file_name="dataset_with_predictions.csv")

                except Exception as e:
                    st.error(str(e))
