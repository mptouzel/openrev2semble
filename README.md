# openrev2semble

Download the accepted papers of an OpenReview venue (e.g. a COLM workshop) and map them by
abstract similarity: papers are grouped into topics and drawn as a network. You get a
slide-ready PNG and an interactive HTML map. Hover a node to see its abstract, and click it
to open the paper on OpenReview.

## Setup

```bash
pip install -r requirements.txt
# abstracts are read from the PDFs with pdftotext (poppler):
sudo apt install poppler-utils      # macOS: brew install poppler
```

Put your OpenReview login in a `.env` file next to the scripts (it is git-ignored):

```
OPENREVIEW_USERNAME=you@example.com
OPENREVIEW_PASSWORD=...
```

If there's no `.env`, the script asks for your login when it runs.

## 1. Download accepted papers

Find the venue id in the URL of the venue page, `https://openreview.net/group?id=<VENUE_ID>`.

```bash
python download_accepted.py --venue-id colmweb.org/COLM/2026/Workshop/Social_Sim --out SocialSim2026
python download_accepted.py --venue-id colmweb.org/COLM/2025/Workshop/Social_Sim --out SocialSim2025
```

This saves one PDF per accepted paper plus an `index.csv` (number, title, authors, forum URL,
file). The script first looks for papers stamped with the venue id. If it finds none (because
post-decision processing hasn't run yet), it reads the accept decisions instead. Add
`--supplementary` to also download supplementary material.

## 2. Build the paper map

```bash
python paper_map.py --papers SocialSim2026 --title "Social Sim '26"
python paper_map.py --papers SocialSim2025 --title "Social Sim '25"
```

Outputs go to `<papers>/paper_map/`:

| file | contents |
|---|---|
| `paper_map.png` | 16:9 static figure for slides |
| `paper_map.html` | interactive map (hover for abstracts, click to open on OpenReview, toggle groups in the legend) |
| `clusters.csv` | group number and label, plus title, authors, URL and layout position for each paper |
| `papers.json` | metadata plus the extracted abstracts |
| `embeddings.npy` | normalised abstract embeddings |

### How it works

1. Pulls the abstract from the first two pages of each PDF.
2. Embeds "title. abstract" with a local sentence-transformer (`all-mpnet-base-v2` by
   default), so no API key is needed.
3. Builds a k-nearest-neighbour graph from cosine similarity. Each paper links to its `k`
   most similar papers, and the edge weight is the similarity.
4. Finds topic groups with Louvain community detection. Each group is labelled with 1–3-word
   phrases that are common inside the group, rare outside it, and close to the group's centroid.
5. Lays out the graph with a force-directed (spring) layout, so similar papers pull together.
   Node size is the paper's number of links (degree).

### Options

| flag | default | effect |
|---|---|---|
| `--k` | 5 | neighbours per paper in the graph |
| `--resolution` | 1.0 | Louvain resolution: higher gives more, smaller groups |
| `--model` | `all-mpnet-base-v2` | any sentence-transformers model |
| `--seed` | 0 | seed for community detection and layout |
| `--out` | `<papers>/paper_map` | output folder |
