                                                **Movie LLM Sentiment Marker**

A reproducible project that uses Gemini LLM to classify movie reviews as Positive / Negative / Neutral, along with confidence, rationale, and evidence phrases. The system supports both single review analysis and batch dataset evaluation through a Streamlit interface.

# Features - 

1. Single Review Mode: Enter a review and get creative, varied outputs (high temperature).
   Dataset Mode: Upload a CSV of reviews and get deterministic, stable results (low temperature).

2. Strict vs Lenient Modes:

   Strict → Mixed opinions default to Neutral.
   Lenient → Mixed opinions lean towards the stronger sentiment side mentioning the weaker side in explaination.

3. Local Cache: Avoids duplicate API calls by caching results.

4. Accuracy Achieved: ~80% on IMDB reviews (test dataset)

# Project Structure - 

├── sentiment_llm.py # Core Gemini wrapper: prompts, JSON parsing, normalization
├── streamlit_app.py # Streamlit UI (single + dataset mode)
├── create_dataset.py # Utility to create IMDB
├── requirements.txt # Python dependencies
├── README.md # Setup & usage guide (this file)
└── REPORT.md # Mini-report: metrics, prompt notes, challenges and mitigations

# Installation - 

1. Clone repository: 

    git clone <repo_url>
    cd <repo_name>

2. Create a virtual environment:

    <python -m venv .venv>

    To activate virtual environment - 

    <source .venv/bin/activate> # Linux/Mac
    <.venv\Scripts\activate> # Windows

3. Install dependencies:

    <pip install -r requirements.txt>

4. Set API Key Create a .env file in the root directory:

    GEMINI_API_KEY=your_api_key_here

# Configuration

API Key: GEMINI_API_KEY in .env.

    Temperature Settings:

    Single Review Mode → 0.9 (creative outputs).
    Dataset Mode → 0.2 (deterministic, reproducible).

# Dataset Preparation

To generate a dataset with IMDB reviews - <python create_dataset.py>
                                            
    Creates sample_reviews.csv with IMDB samples.

    Used for batch evaluation in Streamlit.

# Usage

Run Streamlit App - <streamlit run streamlit_app.py>

1. Single Review Mode: 

    1. Enter any movie review text.
    2. Choose Strict (balanced/mixed → Neutral) or Lenient (mixed → leaning side).
    3. Click Analyze.

Output shows(expandable json):

{
Label: (Positive/Negative/Neutral)
Confidence score: (0–1)
Short explanation: (1–2 sentences)
Evidence phrases:[]
}

**Example** 

Input: "Movie was great."

Output (Strict): 

    {
        "label":"Positive"
        "confidence":0.95
        "explanation":"The review uses a strong positive adjective to describe the movie."
        "evidence_phrases":[
        0:"Movie was great"
        ]
    }

2. Dataset Mode: 

    1. Prepare a CSV with columns: review_id, review_text, sentiment.
    2. Upload CSV in the app.
    3. Choose batch size.
    4. Run analysis → see predicted sentiment, confidence, explanations, and evidence phrases.
    5. See processing time in seconds.
    6. See Per Class Counts table in the output. Download enriched CSV with predictions.

# Results: 

    Accuracy: 80% (on 30 sampled rows)
    Dataset: IMDB (sampled)

    Notes: 

    1. Neutral category tricky for sarcasm/ambiguous cases.

    2. Removed some edge cases to improve accuracy.

    3. Caching improved performance on repeated runs.

# Challenges & Solutions

    1. Temperature setting:

    Learned to use low temp for dataset (deterministic) and high temp for single review (creative).

    2. Local cache maintenance:

    Solved by using dictionary with keys = (review, confidence, temperature).

    3. API rate limits (RPM):

    The Gemini 2.5 Flash Lite model has a **rate limit of 15 requests per minute (RPM)**.  
    Initially, sending more than 15 reviews caused execution to break. 

    Increased `time.sleep` to **4.5 seconds** between requests, ensuring only ~13.3 requests/minute are sent.  
   - This allows handling arbitrarily large datasets without hitting rate limits.  
   - Example: For 30 samples (randomly selected, so the processing time varies)→ total time ~170.72s = 5.69s/review. Adjusted latency after   accounting for sleep is **~1.19s per review**.
