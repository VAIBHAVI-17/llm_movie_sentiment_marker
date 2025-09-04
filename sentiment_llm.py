"""
Gemini wrapper with few-shot prompt that returns strict JSON:
{
  "label": "Positive | Negative | Neutral",
  "confidence": 0.00,
  "explanation": "Short reason grounded in the text",
  "evidence_phrases": ["phrase1", "phrase2"]
}

- explanation is enforced to be 1-2 short sentences (short rationale).
"""

import os
import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import google.generativeai as genai

# ---------- Setup ----------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Put it in your .env as GEMINI_API_KEY=...")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

# ---------- Helpers ----------
_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _clean_text_preserve_json(raw: str) -> str:
    """
    Remove triple-backtick and surrounding text, return a substring that contains the JSON block if possible.
    """
    if raw is None:
        return ""

    text = raw.strip()

    # Remove any leading/trailing triple backticks and an optional 'json' tag
    if text.startswith("```"):
        # remove leading ``` and trailing ```
        text = text.lstrip("`").strip()
        # If the first word is 'json', strip it
        if text[:4].lower() == "json":
            text = text[4:].strip()
        # remove trailing code fence if any
        if text.endswith("```"):
            text = text[:-3].strip()

    m = _JSON_BLOCK.search(text)
    if m:
        return m.group(0)

    # Fallback
    return text


def _shorten_explanation(text: Optional[str], max_sentences: int = 2) -> str:
    """
    Keep only up to max_sentences sentences for a short rationale.
    Splits on sentence boundaries (.,!,?).
    """
    if not text:
        return ""
    # Clean whitespace
    s = str(text).strip()
    # Split into sentences
    parts = re.split(r'(?<=[.!?])\s+', s)
    chosen = parts[:max_sentences]
    short = " ".join(chosen).strip()
    # Final trim: if too long, reduce to 240 chars
    if len(short) > 240:
        short = short[:240].rsplit(" ", 1)[0] + "..."
    return short


def _extract_json(raw_text: str) -> Optional[Any]:
    """
    Extract a JSON object """
    cleaned = _clean_text_preserve_json(raw_text)

    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None
 

def _normalize_label(label_raw: Any) -> str:
    """
    Normalize label variants to EXACT 'Positive' | 'Negative' | 'Neutral'.
    Raise ValueError if it can't be normalized.
    """
    if label_raw is None:
        raise ValueError("Label missing from model output")

    lab = str(label_raw).strip().lower()
    if lab in {"positive", "pos", "p"}:
        return "Positive"
    if lab in {"negative", "neg", "n"}:
        return "Negative"
    if lab in {"neutral", "neu", "ntrl", "neut"}:
        return "Neutral"
    # If model returned something else, it's an explicit error
    raise ValueError(f"Invalid label from model: '{label_raw}'")


def _coerce_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given parsed JSON (possibly with varied keys), coerce to the exact schema:
    {label, confidence, explanation, evidence_phrases}
    Also ensure explanation is short.
    """
    # Label: accept fields like label
    label = parsed.get("label") or parsed.get("sentiment") or parsed.get("prediction")
    label = _normalize_label(label)

    # Confidence:
    conf_raw = parsed.get("confidence") or parsed.get("score") or 0.0
    try:
        confidence = float(conf_raw)
    except Exception:
        confidence = 0.0
    if confidence < 0.0 or confidence > 1.0:
        confidence = 0.0

    # Explanation: prefer keys 'explanation', 'rationale', 'reason', 'justification'
    explanation_raw = (
        parsed.get("explanation")
        or parsed.get("rationale")
        or parsed.get("reason")
        or parsed.get("justification")
        or ""
    )
    explanation = _shorten_explanation(explanation_raw, max_sentences=2)

    # Evidence phrases: expect list; coerce or empty list
    ev = parsed.get("evidence_phrases") or parsed.get("evidence") or parsed.get("highlights") or []
    if not isinstance(ev, list):
        # split a single string by ; or |
        if isinstance(ev, str):
            ev_list = [s.strip() for s in re.split(r"[|;]\s*", ev) if s.strip()]
        else:
            ev_list = []
    else:
        ev_list = [str(x).strip() for x in ev if isinstance(x, (str, int, float))]

    # trim to max 3 items
    evidence_phrases = ev_list[:3]

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "evidence_phrases": evidence_phrases,
    }


# ---------- Few-shot prompt with rules ----------
_FEW_SHOT = [
    # ------- Clear Positive ------
    {
        "review": "Loved the soundtrack and the performances, even though the story drags in the middle.",
        "json": {
            "label": "Positive",
            "confidence": 0.86,
            "explanation": "Praise for soundtrack and performances outweighs the pacing complaint.",
            "evidence_phrases": ["Loved the soundtrack", "performances", "story drags"]
        }
    },

    # ------- Clear Negative ------
    {
        "review": "Terrible pacing and wooden acting. Do not recommend.",
        "json": {
            "label": "Negative",
            "confidence": 0.94,
            "explanation": "Strong negative language about pacing and acting.",
            "evidence_phrases": ["Terrible pacing", "wooden acting", "Do not recommend"]
        }
    },

    # -------- Mixed review → STRICT mode = Neutral --------
    {
        "review": "Great acting, but the story was boring.",
        "json": {
            "label": "Neutral",
            "confidence": 0.70,
            "explanation": "The review praises the acting but criticizes the story, making the sentiment balanced.",
            "evidence_phrases": ["Great acting", "story was boring"]
        },
    },

    # -------- Mixed review → LENIENT mode = Positive --------
    {
        "review": "Great acting, but the story was boring.",
        "json": {
            "label": "Positive",
            "confidence": 0.75,
            "explanation": "The review is mostly positive about the acting, but notes the story as a drawback.",
            "evidence_phrases": ["Great acting", "story was boring"]
        },
    },

    # -------- Mixed review → LENIENT mode = Positive --------
    {
        "review": "The cinematography was stunning, though the pacing was a bit slow.",
        "json": {
            "label": "Positive",
            "confidence": 0.75,
            "explanation": "The review is mostly positive about cinematography, but mentions slow pacing as a drawback.",
            "evidence_phrases": ["cinematography was stunning", "pacing was a bit slow"]
        },
    },

    # -------- Mixed review → LENIENT mode = Negative --------
    {
        "review": "The plot was messy and confusing, though the soundtrack was nice.",
        "json": {
            "label": "Negative",
            "confidence": 0.78,
            "explanation": "The review is mostly negative about the plot, but notes the soundtrack positively.",
            "evidence_phrases": ["plot was messy and confusing", "soundtrack was nice"]
        },
    },

    # -------- Purely Neutral factual review --------
    {
        "review": "The movie releases next week and stars two actors.",
        "json": {
            "label": "Neutral",
            "confidence": 0.70,
            "explanation": "Factual description without a clear opinion.",
            "evidence_phrases": []
        }
    }
]


_PROMPT_INSTRUCTIONS = """
You are a precise movie-review sentiment classifier.

Return ONLY one compact JSON object and nothing else. No commentary, no markdown.
Schema (exact):
{
  "label": "Positive | Negative | Neutral",
  "confidence": 0.00,
  "explanation": "Short reason grounded in the text (1-2 short sentences).",
  "evidence_phrases": ["phrase1", "phrase2"]
}

Rules:
- Base judgment only on the provided review text.
- Sarcasm: if sarcasm exists but target is ambiguous -> Neutral (Strict).
- Comparisons ("better than the last one"): use relative tone; if overall favorable -> Positive.
- Third-party quotes without reviewer stance -> Neutral unless reviewer endorses/opposes.
- Mixed opinions ("great acting, boring plot"):
  - STRICT mode: If review contains both positives and negatives, output "Neutral" in label unless one side is overwhelmingly dominant.
  - LENIENT mode: If review contains both positives and negatives, choose the stronger side (Positive or Negative) as the label. Always mention the weaker side in the explanation.

Return valid JSON that follows the schema exactly. Keep "explanation" short (one or two short sentences).
"""


def analyze_review(review: str, *, strict: bool = True, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Analyze a review and return a validated JSON dict with keys:
    label (Positive|Negative|Neutral), confidence (0..1), explanation (short), evidence_phrases (list).

    - strict: if True, prompt uses STRICT mode rules; if False, use LENIENT mode rules.
    - temperature: model temperature (use low for dataset, higher for single review).
    """
    review = (review or "").strip()
    if not review:
        # Return a valid JSON structure for empty inputs
        return {"label": "Neutral", "confidence": 0.0, "explanation": "No review text provided.", "evidence_phrases": []}

    # Build prompt (few-shot + mode instruction)
    examples = "\n\n".join([f'Review: "{ex["review"]}"\nJSON: {json.dumps(ex["json"], ensure_ascii=False)}' for ex in _FEW_SHOT])
    if strict:
        mode_inst = (
            "STRICT mode active: For mixed reviews (both good and bad), "
            "output must be 'Neutral' unless one side is extremely dominant."
        )
    else:
        mode_inst = (
            "LENIENT mode active: For mixed reviews (both good and bad), "
            "pick the dominant sentiment (Positive or Negative) as the label. "
            "Always mention the weaker side in the explanation."
        )
    prompt = f"{_PROMPT_INSTRUCTIONS}\nMode instruction: {mode_inst}\n\n{examples}\n\nReview: \"{review}\"\nReturn the JSON now."

    # Calling Gemini
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": float(temperature), "max_output_tokens": 256},
    )

    raw = getattr(resp, "text", None) or str(resp)
    parsed = _extract_json(raw)
    if not parsed:
        raise ValueError(f"Model did not return parseable JSON. Raw output:\n{raw}")

    return _coerce_result(parsed)
