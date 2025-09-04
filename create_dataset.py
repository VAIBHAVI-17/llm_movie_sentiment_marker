from datasets import load_dataset
import pandas as pd

def create_imdb_sample(output_file="sample_reviews.csv", imdb_sample_size=10):
    """
    Creates a CSV with:
    - `imdb_sample_size` reviews sampled from IMDB dataset
    """

    # Load IMDB dataset
    imdb = load_dataset("imdb", split="train")

    # Convert the hugging dataset object to Pandas DataFrame
    df = imdb.to_pandas()

    # random sample of given size
    sample_df = df.sample(n=imdb_sample_size, random_state=42).reset_index(drop=True)

    # Keep only needed columns
    sample_df = sample_df.rename(columns={"text": "review_text", "label": "sentiment"})
    sample_df["review_id"] = range(1, len(sample_df) + 1)

    # Mapping sentiment (0 = negative, 1 = positive)
    sample_df["sentiment"] = sample_df["sentiment"].map({0: "negative", 1: "positive"})

    # Reordering columns
    sample_df = sample_df[["review_id", "review_text", "sentiment"]]

    # Save to CSV
    sample_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Dataset created with {len(sample_df)} rows and saved to {output_file}")

if __name__ == "__main__":
    
    create_imdb_sample(imdb_sample_size=30)
