"""
Build a similarity network of the accepted papers and render it as an interactive map.

Pipeline:
  1. Read title/authors/forum url from <papers>/index.csv and pull the abstract
     out of the first page(s) of each PDF (via `pdftotext`).
  2. Embed "title. abstract" with a sentence-transformer (local, no API needed).
  3. Cosine similarity -> k-nearest-neighbour graph (edge weight = similarity).
  4. Louvain communities on that graph = topic groups; each group is labelled with its
     most distinctive terms (class-based TF-IDF).
  5. Force-directed layout (similar papers pull together) -> plotly HTML with hover
     abstracts, plus a static PNG.

Usage:  python paper_map.py --papers SocialSim2026 [--title "Social Sim '26"] [--k 5] [--resolution 1.0] [--model all-mpnet-base-v2]
Needs:  sentence-transformers, scikit-learn, networkx, plotly, matplotlib, poppler (pdftotext)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))

# Words that appear in nearly every paper of this workshop and say nothing about the topic.
DOMAIN_STOP = {
    "llm", "llms", "language", "model", "models", "large", "simulation", "simulations",
    "simulated", "simulate", "social", "agent", "agents", "based", "using", "paper",
    "results", "show", "propose", "work", "study", "findings", "approach", "framework",
    "human", "humans", "lm", "lms", "ai", "use", "new", "across", "evaluate", "evaluation",
    "llm-based", "does", "generated", "produce", "single", "increasingly", "finding", "find",
    "test", "tests", "used", "multiple", "large-scale", "real", "realistic",
}


# ---------------------------------------------------------------- data loading

def extract_abstract(pdf_path):
    """Return the abstract text from the first two pages of a paper PDF ('' if not found)."""
    try:
        text = subprocess.run(["pdftotext", "-l", "2", pdf_path, "-"],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    m = re.search(r"\bAbstract\b\s*\n(.*?)(?:\n\s*1\.?\s*\n+\s*[A-Z][^\n]{2,60}\n|\n\s*1\.?\s+Introduction)",
                  text, flags=re.S | re.I)
    abstract = m.group(1) if m else text[:2000]
    abstract = re.sub(r"-\n(\w)", r"\1", abstract)   # de-hyphenate line breaks
    abstract = re.sub(r"\s+", " ", abstract).strip()
    abstract = re.split(r"(?:\s\d{1,3}){3,}\s", abstract)[0]     # cut at line-number gutters
    abstract = re.sub(r"(\s\d{1,3})+$", "", abstract)              # trailing page/line numbers
    abstract = re.sub(r"\b([A-Z]) ([A-Z]{2,})", r"\1\2", abstract)  # small-caps artefacts: "P ERSUASION"
    return abstract


def load_papers(paper_dir):
    papers = []
    with open(os.path.join(paper_dir, "index.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["file"]:
                continue
            row["abstract"] = extract_abstract(os.path.join(paper_dir, row["file"]))
            papers.append(row)
    return papers


# ---------------------------------------------------------------- analysis

def similarity_graph(emb, k):
    """kNN graph: connect each paper to its k most similar papers (mutual edges merged)."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    G = nx.Graph()
    G.add_nodes_from(range(len(emb)))
    for i in range(len(emb)):
        for j in np.argsort(-sim[i])[:k]:
            G.add_edge(i, int(j), weight=float(sim[i, j]))
    return G, sim


def label_clusters(texts, clusters, emb, encoder, n_terms=3):
    """Label each group with phrases that are shared by its papers and semantically central to it.

    Candidates: 1-3-word phrases (TF-IDF over terms used by >= 2 papers) occurring in at least
    30% of the group's papers (relaxed for broad groups) and over-represented relative to the rest. They are then ranked by
    embedding similarity to the group's centroid minus similarity to the closest other centroid.
    """
    vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP),
                          ngram_range=(1, 3), min_df=2, max_df=0.5, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b")
    X = vec.fit_transform(texts).toarray()
    vocab = np.array(vec.get_feature_names_out())
    centroids = np.stack([emb[m].mean(0) for m in clusters])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    labels = []
    for c, members in enumerate(clusters):
        inside = np.zeros(len(texts), bool)
        inside[members] = True
        contrast = X[inside].mean(0) - X[~inside].mean(0)
        coverage = (X[inside] > 0).mean(0)
        # require the phrase in >= 30% of the group (relaxed for broad groups), and >= 2 papers
        for min_cov in (0.3, 0.2, 0.0):
            cand = np.where((coverage >= min_cov) & (coverage * len(members) >= 2) & (contrast > 0))[0]
            if len(cand) >= n_terms:
                break
        cand = cand[np.argsort(-contrast[cand])][:60]
        if len(cand) == 0:
            labels.append(f"group {c + 1}")
            continue
        term_emb = encoder.encode(list(vocab[cand]), normalize_embeddings=True, show_progress_bar=False)
        sims = term_emb @ centroids.T
        others = np.delete(sims, c, axis=1).max(1) if len(clusters) > 1 else 0
        score = sims[:, c] - 0.5 * others
        terms = []
        for t in vocab[cand[np.argsort(-score)]]:
            # skip phrases overlapping one already chosen
            stems = {w[:6] for w in t.split()}
            if any(stems & {w[:6] for w in s.split()} for s in terms):
                continue
            terms.append(t)
            if len(terms) == n_terms:
                break
        labels.append(" · ".join(terms))
    return labels


# ---------------------------------------------------------------- plotting

def plot_interactive(papers, G, pos, comm_of, labels, palette, out_html, venue):
    fig = go.Figure()

    # edges, opacity by similarity
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    lo, hi = w.min(), w.max()
    for u, v, d in G.edges(data=True):
        a = 0.3 + 0.5 * (d["weight"] - lo) / (hi - lo + 1e-9)
        fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]],
                                 mode="lines", line=dict(width=1 + 3 * (a - 0.3), color=f"rgba(60,60,60,{a:.2f})"),
                                 hoverinfo="skip", showlegend=False))

    # nodes, one trace per cluster so the legend toggles groups
    for c, label in enumerate(labels):
        idx = [i for i in G.nodes if comm_of[i] == c]
        hover = [
            f"<b>#{papers[i]['number']} {'<br>'.join(textwrap.wrap(papers[i]['title'], 70))}</b>"
            f"<br><i>{textwrap.shorten(papers[i]['authors'], 90)}</i><br><br>"
            + "<br>".join(textwrap.wrap(textwrap.shorten(papers[i]["abstract"], 600), 90))
            for i in idx
        ]
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in idx], y=[pos[i][1] for i in idx],
            mode="markers+text", text=[papers[i]["number"] for i in idx],
            textposition="middle center", textfont=dict(size=9, color="white"),
            marker=dict(size=[14 + 2 * G.degree(i) for i in idx], color=palette[c],
                        line=dict(width=1, color="white")),
            name=f"{c + 1}. {label} ({len(idx)})",
            hovertext=hover, hoverinfo="text",
            customdata=[papers[i]["forum_url"] for i in idx],
        ))
        cx, cy = np.mean([pos[i] for i in idx], axis=0)
        fig.add_annotation(x=cx, y=cy, text="<b>" + label.replace(" · ", "<br>") + "</b>", showarrow=False,
                           yshift=70, font=dict(size=30, color=palette[c]), bgcolor="rgba(255,255,255,0.75)")

    fig.update_layout(
        title=f"{venue} @ COLM — accepted papers by abstract similarity "
              "(hover for abstract, click to open on OpenReview)",
        template="plotly_white", hovermode="closest", height=900,
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        legend=dict(title="Topic groups", itemclick="toggle"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    # click a node -> open its OpenReview forum
    post_script = """
    document.querySelector('.plotly-graph-div').on('plotly_click', function(d){
        var url = d.points[0].customdata; if (url) window.open(url, '_blank');
    });"""
    fig.write_html(out_html, include_plotlyjs="cdn", post_script=post_script)


def group_hull(points, pad):
    """Rounded convex hull around a group's nodes (each point inflated to a circle of radius pad)."""
    from scipy.spatial import ConvexHull
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1) * pad
    cloud = (points[:, None, :] + ring[None]).reshape(-1, 2)
    return cloud[ConvexHull(cloud).vertices]


def plot_static(papers, G, pos, comm_of, labels, palette, out_png, venue):
    """Slide-ready 16:9 figure: network on the left, large group labels in a column on the right."""
    fig = plt.figure(figsize=(20, 11.25))
    ax = fig.add_axes([0.01, 0.02, 0.62, 0.88])
    lax = fig.add_axes([0.64, 0.02, 0.35, 0.88])
    lax.set_axis_off()

    # shaded region per group
    for c in range(len(labels)):
        pts = np.array([pos[i] for i in G.nodes if comm_of[i] == c])
        ax.fill(*group_hull(pts, 0.07).T, color=palette[c], alpha=0.12, lw=0, zorder=0)

    # links: darker and thicker for more similar pairs
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    t = (w - w.min()) / (w.max() - w.min() + 1e-9)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2 + 3.0 * t,
                           edge_color=[(0.25, 0.25, 0.25, 0.35 + 0.5 * x) for x in t])
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[palette[comm_of[i]] for i in G.nodes],
                                   node_size=[450 + 60 * G.degree(i) for i in G.nodes],
                                   edgecolors="white", linewidths=1.5)
    nodes.set_zorder(3)
    for txt in nx.draw_networkx_labels(G, pos, {i: papers[i]["number"] for i in G.nodes},
                                       font_size=11, font_weight="bold", font_color="white", ax=ax).values():
        txt.set_zorder(4)
    ax.set_axis_off()
    ax.margins(0.03)

    # label column
    sizes = [sum(comm_of[i] == c for i in G.nodes) for c in range(len(labels))]
    slot = 1.0 / len(labels)
    for c, label in enumerate(labels):
        y = 1 - (c + 0.5) * slot
        lax.scatter([0.04], [y], s=900, color=palette[c], edgecolors="white", linewidths=1.5)
        lax.text(0.04, y, str(sizes[c]), ha="center", va="center", fontsize=14, weight="bold", color="white")
        lax.text(0.11, y, label.replace(" · ", "\n"), ha="left", va="center", fontsize=26,
                 weight="bold", linespacing=1.05, color=palette[c])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)

    fig.suptitle(f"{venue}: accepted papers by abstract similarity", fontsize=30, weight="bold", y=0.975)
    fig.text(0.5, 0.905, "node = paper #  ·  links = most similar abstracts  ·  "
             "colour = topic group (count in legend dot)", ha="center", fontsize=16, color="#555")
    fig.savefig(out_png, dpi=160)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", default=os.path.join(HERE, "SocialSim2026"),
                    help="folder with the PDFs and index.csv")
    ap.add_argument("--out", help="output folder (default: <papers>/paper_map)")
    ap.add_argument("--title", default="Social Sim '26", help="workshop name used in figure titles")
    ap.add_argument("--model", default="all-mpnet-base-v2", help="sentence-transformers model")
    ap.add_argument("--k", type=int, default=5, help="neighbours per paper in the graph")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain resolution: higher -> more, smaller groups")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.papers, "paper_map")
    os.makedirs(args.out, exist_ok=True)

    papers = load_papers(args.papers)
    missing = [p["number"] for p in papers if len(p["abstract"]) < 200]
    print(f"Loaded {len(papers)} papers" + (f"; abstract extraction looks weak for #{missing}" if missing else ""))

    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    encoder = SentenceTransformer(args.model)
    emb = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    G, sim = similarity_graph(emb, args.k)
    communities = nx.community.louvain_communities(G, weight="weight", resolution=args.resolution, seed=args.seed)
    communities = sorted((sorted(c) for c in communities), key=len, reverse=True)
    comm_of = {i: c for c, members in enumerate(communities) for i in members}
    labels = label_clusters(texts, communities, emb, encoder)

    # spring layout on (similarity - baseline) so only *relative* similarity drives attraction
    base = min(d["weight"] for *_, d in G.edges(data=True))
    for _, _, d in G.edges(data=True):
        d["pull"] = (d["weight"] - base + 0.05) ** 2
    pos = nx.spring_layout(G, weight="pull", k=1.2 / np.sqrt(len(G)), iterations=500, seed=args.seed)

    palette = (plt.cm.tab10.colors + plt.cm.Dark2.colors)
    palette = ["#%02x%02x%02x" % tuple(int(255 * x) for x in col) for col in palette]

    plot_interactive(papers, G, pos, comm_of, labels, palette, os.path.join(args.out, "paper_map.html"), args.title)
    plot_static(papers, G, pos, comm_of, labels, palette, os.path.join(args.out, "paper_map.png"), args.title)

    # tabular outputs for reuse (e.g. session planning)
    with open(os.path.join(args.out, "clusters.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "cluster_label", "number", "title", "authors", "forum_url", "x", "y"])
        for i, p in enumerate(papers):
            w.writerow([comm_of[i] + 1, labels[comm_of[i]], p["number"], p["title"], p["authors"],
                        p["forum_url"], *np.round(pos[i], 4)])
    np.save(os.path.join(args.out, "embeddings.npy"), emb)
    with open(os.path.join(args.out, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=1, ensure_ascii=False)

    print(f"\n{len(communities)} groups (k={args.k}, resolution={args.resolution}):")
    for c, members in enumerate(communities):
        print(f"\n[{c + 1}] {labels[c]}")
        for i in members:
            print(f"    #{papers[i]['number']:>3}  {textwrap.shorten(papers[i]['title'], 90)}")
    print(f"\nWrote {args.out}/paper_map.html, paper_map.png, clusters.csv, embeddings.npy, papers.json")


if __name__ == "__main__":
    main()
