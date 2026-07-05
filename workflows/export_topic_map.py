#!/usr/bin/env python3
"""
Export a 2D topic map of the corpus for the dashboard.

Doc-level embeddings (mean-pooled from each document's chunk vectors in
LanceDB) are reduced to 2D via UMAP, clustered via KMeans (k chosen by
silhouette score), and each cluster is given a short LLM-generated topic
label. Each document also gets a nearest-neighbor cosine similarity against
every other document — the near-duplicate threshold is the corpus's own 95th
percentile of that distribution (data-driven, not guessed), which a manual
spot-check confirmed catches genuine redundancy (the same doc scraped as both
HTML and PDF, reissues across catalog years, re-hosted copies with a
different title prefix). Per-cluster redundancy % and the top duplicate
pairs overall are exported for the dashboard's "where is this corpus bloated
with copies" view.

Run (needs the project's real ML env — lancedb/sentence-transformers/torch/
sklearn/umap-learn, not the dashboard-only env):
    /opt/homebrew/anaconda3/envs/whisper-310/bin/python workflows/export_topic_map.py

Writes: dashboard_data/topic_map.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

from device_config import config
from db_vector_lance import LanceVectorDB
from llm_client import GeminiClient

NEAR_DUP_PERCENTILE = 95  # data-driven threshold, not a fixed guess — see module docstring
TOP_DUP_PAIRS = 25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT, 'dashboard_data', 'topic_map.json')

_STOPWORDS = {
    'the', 'a', 'an', 'and', 'of', 'for', 'in', 'to', 'on', 'with', 'is',
    'are', 'pdf', 'html', 'audio', 'from', 'how', 'what', 'why',
}


def doc_level_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean-pool each document's chunk vectors into one normalized doc vector."""
    vecs = np.stack(df['vector'].to_numpy())
    rows = []
    for content_id, idx in df.groupby('content_id').groups.items():
        idx = np.asarray(idx)
        mean_vec = vecs[idx].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        g = df.loc[idx]
        rows.append({
            'content_id': str(content_id),
            'title': g['title'].iloc[0],
            'content_type': g['content_type'].iloc[0],
            'source_name': g['source_name'].iloc[0],
            'chunk_count': int(len(idx)),
            'vector': mean_vec,
        })
    return pd.DataFrame(rows)


def pick_k(vectors: np.ndarray, k_range=range(6, 15)) -> int:
    """Choose cluster count by silhouette score rather than guessing."""
    best_k, best_score = list(k_range)[0], -1.0
    for k in k_range:
        if k >= len(vectors):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vectors)
        score = silhouette_score(vectors, km.labels_)
        print(f"  k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _fallback_label(titles: list) -> str:
    """Frequent non-trivial word pair from titles — used only if the LLM call fails."""
    words = {}
    for t in titles:
        cleaned = str(t).lower().replace('[pdf]', '').replace('[html]', '').replace('[audio]', '')
        for w in cleaned.split():
            w = w.strip('.,:;()[]"\'')
            if len(w) > 3 and w not in _STOPWORDS:
                words[w] = words.get(w, 0) + 1
    top = sorted(words.items(), key=lambda x: -x[1])[:2]
    return " / ".join(w.title() for w, _ in top) if top else "Uncategorized"


def label_clusters(doc_df: pd.DataFrame) -> dict:
    labels = {}
    client = None
    try:
        client = GeminiClient(model=config.LLM_MODEL)
    except Exception as e:
        print(f"  No LLM client available ({e}); using frequent-word fallback labels")

    for cid, g in doc_df.groupby('cluster'):
        titles = g['title'].head(15).tolist()
        label = None
        if client is not None:
            try:
                prompt = (
                    "Here are document titles from one cluster of a manufacturing/"
                    "industrial-AI knowledge corpus:\n\n"
                    + "\n".join(f"- {t}" for t in titles)
                    + "\n\nRespond with ONLY a short topic label (2-5 words, title case) "
                      "that captures what this cluster is about. No punctuation, no explanation."
                )
                raw = client.generate(
                    prompt, "You are labeling document clusters for a dashboard.", temperature=0.2,
                )
                label = raw.strip().strip('"').strip("'")[:60]
            except Exception as e:
                print(f"  LLM labeling failed for cluster {cid} ({e}); using fallback")
        labels[cid] = label or _fallback_label(titles)
    return labels


def compute_near_duplicates(doc_df: pd.DataFrame, doc_vectors: np.ndarray):
    """Nearest-neighbor cosine similarity per doc, plus a data-driven near-dup threshold
    (the corpus's own 95th percentile) and the top overall duplicate-candidate pairs."""
    sim = doc_vectors @ doc_vectors.T
    np.fill_diagonal(sim, -1.0)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim.max(axis=1)
    threshold = float(np.percentile(nn_sim, NEAR_DUP_PERCENTILE))
    print(f"  near-dup threshold (p{NEAR_DUP_PERCENTILE} of nearest-neighbor similarity): {threshold:.4f}")

    doc_df = doc_df.copy()
    doc_df['nn_similarity'] = nn_sim
    doc_df['nn_content_id'] = doc_df['content_id'].to_numpy()[nn_idx]
    doc_df['nn_title'] = doc_df['title'].to_numpy()[nn_idx]
    doc_df['is_near_dup'] = doc_df['nn_similarity'] >= threshold

    # Top overall pairs, deduplicated (i,j) == (j,i), highest similarity first
    seen = set()
    pairs = []
    order = np.argsort(-nn_sim)
    for i in order:
        j = nn_idx[i]
        key = tuple(sorted((int(i), int(j))))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            'title_a': doc_df['title'].iloc[i],
            'title_b': doc_df['title'].iloc[j],
            'content_id_a': doc_df['content_id'].iloc[i],
            'content_id_b': doc_df['content_id'].iloc[j],
            'similarity': float(nn_sim[i]),
        })
        if len(pairs) >= TOP_DUP_PAIRS:
            break

    return doc_df, threshold, pairs


def main():
    print("Loading LanceDB corpus vectors...")
    # corpus_vectors was built with nomic-embed-text-v1.5 (768-dim) — must match
    # exactly, not LanceVectorDB's all-MiniLM-L6-v2 default (that's for signal_vectors).
    vdb = LanceVectorDB(
        config.LANCE_VECTOR_PATH,
        embedding_dim=768,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )
    tbl = vdb._get_table()
    df = tbl.to_pandas()
    print(f"  {len(df)} chunks loaded")

    print("Aggregating to document level (mean-pooled chunk vectors)...")
    doc_df = doc_level_aggregate(df)
    print(f"  {len(doc_df)} documents")
    doc_vectors = np.stack(doc_df['vector'].to_numpy())

    print("Selecting cluster count via silhouette score...")
    k = pick_k(doc_vectors)
    print(f"  chosen k={k}")

    print("Clustering (KMeans on 768-dim doc vectors, not the 2D projection)...")
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(doc_vectors)
    doc_df['cluster'] = km.labels_

    print("Labeling clusters...")
    cluster_labels = label_clusters(doc_df)
    doc_df['cluster_label'] = doc_df['cluster'].map(cluster_labels)
    for cid, label in sorted(cluster_labels.items()):
        n = int((doc_df['cluster'] == cid).sum())
        print(f"  cluster {cid} ({n} docs): {label}")

    print("Computing nearest-neighbor similarity / near-duplicate candidates...")
    doc_df, dup_threshold, dup_pairs = compute_near_duplicates(doc_df, doc_vectors)
    print(f"  {int(doc_df['is_near_dup'].sum())}/{len(doc_df)} docs flagged as near-duplicate candidates")

    print("Fitting UMAP on document vectors...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    doc_xy = reducer.fit_transform(doc_vectors)
    doc_df['x'] = doc_xy[:, 0]
    doc_df['y'] = doc_xy[:, 1]

    cluster_dup_stats = (
        doc_df.groupby('cluster')['is_near_dup']
        .agg(near_dup_count='sum', doc_count='count')
        .reset_index()
    )
    cluster_dup_stats['near_dup_pct'] = (
        cluster_dup_stats['near_dup_count'] / cluster_dup_stats['doc_count'] * 100
    ).round(1)

    out = {
        'docs': [
            {
                'content_id': r['content_id'], 'title': r['title'],
                'content_type': r['content_type'], 'source_name': r['source_name'],
                'chunk_count': r['chunk_count'], 'cluster': int(r['cluster']),
                'cluster_label': r['cluster_label'], 'x': float(r['x']), 'y': float(r['y']),
                'nn_similarity': float(r['nn_similarity']), 'is_near_dup': bool(r['is_near_dup']),
            }
            for r in doc_df.to_dict(orient='records')
        ],
        'clusters': [
            {
                'id': int(cid), 'label': label,
                'doc_count': int((doc_df['cluster'] == cid).sum()),
                'near_dup_pct': float(cluster_dup_stats.loc[cluster_dup_stats['cluster'] == cid, 'near_dup_pct'].iloc[0]),
            }
            for cid, label in sorted(cluster_labels.items())
        ],
        'duplicate_pairs': dup_pairs,
        'near_dup_threshold': dup_threshold,
        'near_dup_percentile': NEAR_DUP_PERCENTILE,
        'generated_at': pd.Timestamp.utcnow().isoformat(),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
