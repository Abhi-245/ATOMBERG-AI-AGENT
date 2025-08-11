# file: sov_pipeline.py
import os, math, time
import pandas as pd
import tweepy
from serpapi import GoogleSearch
import re
import openai
from openai import OpenAI


client = OpenAI(api_key="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")  # Replace with your real key or environment variable
OPENAI_KEY = os.getenv('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')
SERPAPI_KEY = "b92b741dde0eb95e29d58b04b60bbe3fa30eaba11dd3091a259ccc5b4f50ab7c"


def fetch_google_serp(query, num=50, gl='in', hl='en', freshness="d"):
    """
    freshness: 'd' = last 24h, 'w' = last week, 'm' = last month
    """
    tbs_map = {"d": "qdr:d", "w": "qdr:w", "m": "qdr:m"}
    params = {
        'engine': 'google',
        'q': query,
        'num': min(num, 50),
        'gl': gl,
        'hl': hl,
        'tbs': tbs_map.get(freshness, ""),  # freshness filter
        'api_key': SERPAPI_KEY
    }
    search = GoogleSearch(params)
    res = search.get_dict()
    results = res.get('organic_results', [])
    rows = []
    for r in results:
        rows.append({
            'id': r.get('link'),
            'title': r.get('title'),
            'snippet': r.get('snippet'),
            'source': 'google',
            'engagement': 0
        })
    return pd.DataFrame(rows)


# === TwitterAPI.io-based fetch function ===
import requests

def fetch_x_twitterapiio(query: str, api_key: str, limit: int = 10):
    """
    Fetch tweets using TwitterAPI.io (Latest type)
    """
    base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"x-api-key": api_key}
    params = {
        "query": query,
        "queryType": "Latest"
    }

    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    tweets = data.get("tweets", [])[:limit]

    # Convert to DataFrame in same format as fetch_x() for pipeline
    rows = []
    for t in tweets:
        public_metrics = t.get("public_metrics", {}) or {}
        rows.append({
            "id": t.get("id"),
            "text": t.get("text"),
            "likes": public_metrics.get("like_count", 0),
            "retweets": public_metrics.get("retweet_count", 0),
            "replies": public_metrics.get("reply_count", 0),
            "date": t.get("created_at"),
            "source": "x"
        })
    return pd.DataFrame(rows)




import pandas as pd
from numpy import dot
from numpy.linalg import norm
import json

# 3) Brand detection with embeddings

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-10)  # added small epsilon for stability

# 4) sentiment via OpenAI Chat (simple wrapper)
def analyze_sentiment(text):
    prompt = (
        f"Classify sentiment of the following text on a scale -1 (very negative) to +1 (very positive). "
        f"Return only a JSON like {{\"score\":0.34, \"label\":\"positive\"}}. "
        f"Text: '''{text}'''"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    out = resp.choices[0].message.content
    try:
        j = json.loads(out)
        return float(j.get('score', 0)), j.get('label', 'neutral')
    except Exception:
        return 0.0, 'neutral'

# === CHANGES START ===
# Map multiple aliases to canonical brand names
brand_aliases_map = {
    "Atomberg": "Atomberg",
    "Atom Berg": "Atomberg",
    "Atom-Berg": "Atomberg",
    "Crompton": "Crompton",
    "Havells": "Havells",
    "Usha": "Usha",
    "Orient": "Orient",
    "Bajaj": "Bajaj",
    "BLDC":"BLDC"

}
canonical_brands = list(set(brand_aliases_map.values()))

def process_batch(df, brand_aliases):
    brand_embeds = {b: get_embedding(b) for b in canonical_brands}
    results = []

    for _, row in df.iterrows():
        text_raw = (str(row.get('title', '')) + ' ' +
                    str(row.get('snippet', '')) + ' ' +
                    str(row.get('text', ''))).strip()
        # Basic cleanup example
        text_clean = re.sub(r"http\S+|nan", "", text_raw, flags=re.IGNORECASE).strip()
        text_lower = text_clean.lower()

        matched_brand = None
        for alias, canonical in brand_aliases_map.items():
            if re.search(rf"\b{re.escape(alias.lower())}\b", text_lower):
                matched_brand = canonical
                break

        if matched_brand is None:
            emb = get_embedding(text_clean)
            sims = {b: cosine(emb, be) for b, be in brand_embeds.items()}
            sorted_sims = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            print(f"Similarity scores: {sorted_sims} for text: {text_clean[:100]}...")

            best_brand, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0

            # Stricter threshold and margin check
            if best_sim >= 0.4 and (best_sim - second_sim) > 0.05:
                matched_brand = best_brand
            else:
                matched_brand = 'Unknown'
                best_sim = 0.0
        else:
            best_sim = 1.0

        sentiment_score, label = analyze_sentiment(text_clean)

        if row.get('source') == 'google':
            eng = 1
        else:
            eng = row.get('likes', 0) + 2 * row.get('retweets', 0) + row.get('replies', 0)

        mention = matched_brand != 'Unknown'
        results.append({
            **row,
            'best_brand': matched_brand,
            'brand_sim': best_sim,
            'mention': mention,
            'sentiment': sentiment_score,
            'engagement': eng
        })

        print(f"Brand: {matched_brand}, Similarity: {best_sim}")

    return pd.DataFrame(results)


# 6) Compute SoV with filtering Unknown brand and normalized composite SoV
def compute_sov(df, brands, weights=(0.25, 0.5, 0.25)):
    # Filter out Unknown brand rows
    df = df[df['best_brand'] != 'Unknown']

    # aggregate metrics for mentioned rows
    agg = df[df['mention'] == True].groupby('best_brand').agg(
        mentions=('id', 'count'),
        total_engagement=('engagement', 'sum'),
        pos_engagement=('engagement', lambda s: s[df.loc[s.index, 'sentiment'] > 0].sum())
    ).reindex(brands).fillna(0)

    # Normalize total engagement to sum to 1
    agg['eng_norm'] = agg['total_engagement'] / (agg['total_engagement'].sum() + 1e-9)
    total_mentions = agg['mentions'].sum() + 1e-9

    sov_count = (agg['mentions'] / total_mentions).to_dict()
    sov_eng = agg['eng_norm'].to_dict()

    # Normalize positive engagement to sum to 1
    pos_sum = agg['pos_engagement'].sum() + 1e-9
    spv = (agg['pos_engagement'] / pos_sum).to_dict()

    # Composite SoV combines the three with given weights, then re-normalized
    w_c, w_e, w_s = weights
    composite_raw = {b: w_c * sov_count.get(b, 0) + w_e * sov_eng.get(b, 0) + w_s * spv.get(b, 0) for b in brands}

    # Normalize composite so sum to 1
    total_comp = sum(composite_raw.values()) + 1e-9
    composite = {b: val / total_comp for b, val in composite_raw.items()}

    return {
        'sov_count': sov_count,
        'sov_eng': sov_eng,
        'spv': spv,
        'composite': composite,
        'table': agg
    }

# === Runnable script block ===
if __name__ == "__main__":
    brands = canonical_brands

    print("Fetching Google SERP results...")
    df_google = fetch_google_serp("smart ceiling fan", num=25)  # limit to small number for test
    print(f"Got {len(df_google)} Google results")


    print("Fetching Tweets via TwitterAPI.io...")
    TWITTERAPIIO_KEY = "dd3f962354c240dfb54bab67a61fb9f6"  # your real key
    df_x = fetch_x_twitterapiio("smart ceiling fan", api_key=TWITTERAPIIO_KEY, limit=40)
    print(f"Got {len(df_x)} tweets")
    # Combine datasets
    df_all = pd.concat([df_google, df_x], ignore_index=True)

    print("Processing batch...")
    df_processed = process_batch(df_all, brand_aliases_map)

    print("Computing SoV...")
    sov_results = compute_sov(df_processed, brands)

    print("=== Share of Voice Results ===")
    print("Count-based SoV:", sov_results["sov_count"])
    print("Engagement-based SoV:", sov_results["sov_eng"])
    print("Share of Positive Voice:", sov_results["spv"])
    print("Composite SoV:", sov_results["composite"])

    print("\nDetailed table:")
    print(sov_results["table"])
