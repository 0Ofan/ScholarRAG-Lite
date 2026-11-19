# ScholarRAG Lite

Source repository: [0Ofan/ScholarRAG-Lite](https://github.com/0Ofan/ScholarRAG-Lite).
The complete application uses 12,000 papers with both Fixed-length and Sentence-aware retrieval. Source code and small Dashboard results are stored in Git; model weights, the unchanged FAISS indexes and original chunks are distributed as checksum-verified assets of the same repository's `runtime-v1` Release. The final report, presentation, credentials, caches and virtual environment are excluded.

**Publication status:** source code and the complete [runtime-v1 Release](https://github.com/0Ofan/ScholarRAG-Lite/releases/tag/runtime-v1) are published. All five uploaded assets match their pinned SHA-256 checksums; anonymous range download was verified. A live Community Cloud URL is not available yet; cloud installation and browser operation still require acceptance.

## Download and run the complete application

Use Python 3.13 on Windows or Linux. Download the repository ZIP through **Code → Download ZIP** and extract it, or clone it with Git. Allow approximately 10 GiB free disk space for the runtime download/extraction, plus the virtual environment. The runtime download is about 2.32 GiB. Existing models and indexes are reused; no dataset processing is required.

Windows PowerShell, from the extracted repository root:

```powershell
# For a new user's checkout only. Do not recreate an existing project environment.
python -m venv .venv
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"
.\.venv\Scripts\python.exe -m pip install -r community/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run community/app.py
```

Linux, from the repository root:

```bash
python3.13 -m venv .venv
PIP_CACHE_DIR="$PWD/.cache/pip" .venv/bin/python -m pip install -r community/requirements.txt
.venv/bin/python -m streamlit run community/app.py
```

Before starting, copy `.env.example` to `.env` and set your own `APP_PASSWORD`. Open `http://localhost:8501` and enter that password. On first authorized startup the app downloads all five Release parts, checks SHA-256 and extracts the original runtime artifacts into this checkout. Later starts reuse the files. All computation runs on the machine hosting Streamlit, without accessing the original author's computer.

For generated answers, fill in your own DeepSeek API key in the local `.env`. After password login, retrieval, method comparison and the Dashboard work without an API key. API usage is paid by the owner of the configured key. Never commit `.env` or Streamlit secrets. On Community Cloud set `APP_PASSWORD` in Secrets; an unconfigured or incorrect password blocks the app before data loading and generation. Passwords are shared with your invited users, not published with source code.

The portable entrypoint uses one CPU encoder, one resident original FAISS index, and disk-backed chunk records. It preserves original vector scores, paper aggregation and evidence text. Both 12,000-paper methods are available, but changing methods reloads the selected index; concurrent retrieval is serialized to bound memory. Original local GPU execution remains available through root `app.py`.

For Community Cloud deployment and GitHub Release publication, see [community/README.md](community/README.md). Optional Docker server instructions remain in [deployment/README.md](deployment/README.md).

## Project overview

ScholarRAG Lite is a local academic retrieval-augmented generation (RAG) system built on the CORD-19 research collection. It uses TREC-COVID topics and relevance judgments to study document preprocessing, retrieval quality, scalability, and computational cost. Its central question is: **How does document preprocessing optimization affect the retrieval quality, scalability, and cost efficiency of an academic RAG system as the dataset grows?**

The implemented comparison is fixed-length versus sentence-aware chunking on frozen sets of 2,000, 6,000, and 12,000 papers. An interactive application retrieves paper evidence and streams a DeepSeek answer with citations. The word *improved* names the proposed sentence-aware strategy; the experiments do not show that it performs better.

## Key features and architecture

- Academic paper text processing with document, section, chunk, and source-span mapping.
- Fixed-length and sentence-aware chunking; BGE-small-en-v1.5 embeddings; normalized FAISS IndexFlatIP search and paper-level max-chunk-score aggregation.
- Evidence-supported DeepSeek Flash answers, citation IDs, source checks, and a three-page Streamlit interface.
- Frozen retrieval comparisons, an overlap ablation, resource measurements, and separate offline retrieval/evidence experiments.

**Offline:** CORD-19 text → cleaning and section assembly → chunking → BGE embeddings → FAISS index. Chunk and Document IDs retain a path back to the source paper.

**Online default:** Research question → BGE query embedding → FAISS chunk search → paper-level Top-10 ranking → one best chunk from each of the top three papers → DeepSeek Flash streaming → answer with evidence citations. Structural citation checks verify IDs and source mapping; they do not establish that every claim is semantically supported.

## Dataset and experimental protocol

The frozen corpus is drawn from the **2020-07-16 CORD-19 snapshot**. All six formal retrieval runs use the same 50 TREC-COVID topics and official qrels after the frozen Document ID normalization. The 2,000-paper Pilot is nested within the 6,000-paper Medium set, which is nested within the 12,000-paper Large set. Each includes the same 956 known-positive anchor papers. See `experiments/configs/phase27_experiment_config.json`, `experiments/configs/phase33_runs.json`, and the manifests under `data/processed/`.

| Strategy | Frozen configuration |
| --- | --- |
| Fixed-length | Target 180 BGE tokens, 20-token overlap, maximum 240 |
| Sentence-aware | Target 180, soft maximum 220, hard maximum 240; no fixed overlap in the six-run comparison |
| Shared retrieval | BAAI/bge-small-en-v1.5, L2-normalized 384-dimensional vectors, FAISS IndexFlatIP, paper-level maximum chunk score |

The Pilot overlap ablation separately tested sentence-aware chunking with 20-token overlap; it is not one of the six scale-comparison runs. See `experiments/results/phase34_ablation/comparison.json`.

## Current application configuration

| Component | Stable default | Available alternative |
| --- | --- | --- |
| Corpus and retrieval | Large (12,000 papers), Fixed-length BGE + FAISS | Large Sentence-aware in the UI |
| Candidate ranking | Top-10 distinct papers, ordered by maximum chunk score | No online BM25/RRF/reranker |
| Evidence | Best original chunk from each of the top three papers | UI allows 4 or 5 chunks from the corresponding top papers |
| Generation | DeepSeek Flash, non-thinking streaming, V1 Stable Default | V2 Experimental with source-role hints and local verification details |

The independent BM25, RRF, Query Decomposition, and BGE reranker implementations are **offline research modules**, not components of the default application. Neither V2 nor the offline experiments replace the V1 default. DeepSeek is called only for answer generation on Search & Answer; retrieval and the dashboard remain usable without an API key.

## Web interface

- **Search & Answer:** Enter a research question, select Large Fixed-length or Sentence-aware, choose 3/4/5 evidence chunks, and select V1 or experimental V2. Inspect streamed answers, citation IDs, original paper chunks, and, in V2, verification hints. `Unverified` is not a confirmation of correctness or an error.
- **Method Comparison:** Search the same question with both chunking indexes and inspect common and distinct papers, scores, and evidence. This page does not generate two LLM answers.
- **Experiment Dashboard:** Explore saved Pilot/Medium/Large retrieval metrics, processing/resource measurements, and the Pilot A/B/C chunking ablation. Viewing the dashboard does not rerun experiments.

## Experimental results

All retrieval metrics below are 50-topic macro averages. Recall uses **known positive papers within the current frozen corpus** as its denominator, not every positive paper in full TREC-COVID. Recall@5 was derived from the saved Top-100 rankings; the other quality metrics are cross-checked historical results. `final_metrics_summary.json` contains each topic, source path, and source-file SHA-256.

| Papers | Chunking | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | Cleaning + chunking (s) | Embedding + vector write (s) | Documents/s during embedding | Warm FAISS + paper aggregation mean (ms) | Peak embedding RAM (GiB) | Peak embedding VRAM reserved (GiB) | FAISS file (GiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | Fixed | .0894 | .1610 | .5736 | .4212 | 59.3 | 169.2 | 11.82 | 3.65 | 1.57 | .49 | .127 |
| 2,000 | Sentence | .0876 | .1597 | .5623 | .4114 | 63.9 | 175.8 | 11.38 | 4.32 | 1.55 | .70 | .130 |
| 6,000 | Fixed | .0643 | .1080 | .4351 | .2835 | 242.9 | 654.6 | 9.17 | 9.46 | 1.86 | .49 | .487 |
| 6,000 | Sentence | .0620 | .1044 | .4335 | .2691 | 244.0 | 673.6 | 8.91 | 9.36 | 1.84 | .70 | .496 |
| 12,000 | Fixed | .0472 | .0824 | .3620 | .2103 | 477.9 | 1356.6 | 8.85 | 17.29 | 2.39 | .49 | .992 |
| 12,000 | Sentence | .0436 | .0756 | .3576 | .1926 | 497.3 | 1364.1 | 8.80 | 17.33 | 2.38 | .68 | 1.010 |

Stage durations use historical three-repeat medians; the FAISS latency column is the mean of per-topic medians across three runs and excludes query encoding. RAM/VRAM values are **embedding-process peaks**, not Streamlit resident usage. The paper-level index build times for Fixed/Sentence were 0.082/0.173 s (Pilot), 0.362/0.327 s (Medium), and 0.825/0.856 s (Large), excluding index-file writes. Document-reading throughput and warm sequential FAISS-only queries/s, with measurement definitions, are in `experiments/results/final_gap_completion/final_metrics_summary.json`; they are not end-to-end or concurrent throughput.

In a separate **Large Fixed V1, three-evidence** experiment, 20 real DeepSeek calls returned 13,116 input and 2,488 output tokens. The historical public-price estimate was **CNY 0.046136 total**, not a provider bill. Mean loaded-query local retrieval was 37.5 ms and mean time to a complete answer was 2.074 s in that run; these timings are not measurements for the other five retrieval configurations. Source: `experiments/results/phase36/aggregate.json`.

## Main findings and Big Data characteristics

- Fixed-length had higher macro nDCG@10 and Recall@5/10/100 at all three sizes. Sentence-aware generally produced more chunks, used more index space, and took longer to embed, without a demonstrated macro retrieval-quality gain under this protocol. The sentence-aware overlap ablation and other offline alternatives remain separate experiments.
- As the corpus grew, vector counts, embedding time, storage, and retrieval latency increased while known-positive retrieval metrics decreased. The positive anchors remained fixed as additional papers were added, so this trend must not be generalized to collections whose relevant-document count grows with corpus size.
- Increasing evidence from Top-3 to Top-5/8 raised input tokens and estimated API cost in the separate Phase 4.18 study. Some questions gained useful evidence; a Top-8 answer also contained a confirmed denominator error. No reliable full-set semantic ranking establishes that a larger evidence budget is better.
- Offline BM25/RRF experiments showed query-dependent complementarity, while Query Decomposition scope control and the six-question BGE reranker pilot did not establish a stable answer-quality improvement. They remain experimental and do not change the application default. See `phase_4_24_2_1_summary.md` and `phase_4_24_3_summary.md`.

The measured **Volume** is three nested corpus sizes and their chunk/index growth. **Variety** covers research topics, paper lengths, and section structures within text, not verified multimodal retrieval. **Velocity** is historical batch processing and sequential query speed; continuous incremental indexing and concurrent serving were not measured. **Veracity** includes Document ID normalization, duplicate/conflicting-version audits, source spans, and citation checks, while semantic claim support remains only partly verified. Details and source boundaries: `experiments/results/final_gap_completion/final_4v_analysis.md`.

## Installation and local use

Use the existing project `.venv`; do not use global Python or recreate indexes to launch the current demonstration. In a Windows PowerShell terminal opened in the project root:

```powershell
$root = 'D:\VSproject\BD\Project'
$env:PIP_CACHE_DIR = "$root\.cache\pip"
$env:HF_HOME = "$root\.cache\huggingface"
$env:HF_HUB_CACHE = "$root\.cache\huggingface\hub"
$env:TORCH_HOME = "$root\.cache\torch"
$env:TEMP = "$root\.cache\tmp"
$env:TMP = "$root\.cache\tmp"
& "$root\.venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501` in a browser. Set `DEEPSEEK_API_KEY` in the existing root `.env` to enable generated answers; optional settings are `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` (`deepseek-flash`), `DEEPSEEK_MAX_TOKENS`, `DEEPSEEK_TEMPERATURE`, and `DEEPSEEK_TIMEOUT`. Never commit `.env` or print the key. If dependencies are missing, install only those needed into the project environment with `& .\.venv\Scripts\python.exe -m pip install -r requirements.txt` after setting the project-local cache variables. Retrieval can still run when the API key is unavailable.

For a command-line retrieval demonstration on the existing Pilot index:

```powershell
& .\.venv\Scripts\python.exe -m src.cli --examples
& .\.venv\Scripts\python.exe -m src.cli --query 'coronavirus origin' --top-k 10
```

## Reproducibility and project structure

```text
app.py             Streamlit application and navigation
src/               Retrieval, RAG, evaluation, and isolated experiment modules
data/              Frozen corpus manifests, processed chunks, indexes, qrels
data/indexes/      Persisted FAISS indexes
models/            Local model snapshots
experiments/       Frozen configs, per-topic results, timings, analyses
tests/             Incremental and historical checks
requirements.txt   Python dependencies
AGENTS.md          Project operation and isolation rules
```

Start with `experiments/configs/phase27_experiment_config.json` and `experiments/configs/phase33_runs.json` for the six-run protocol; the Pilot results are under `experiments/results/phase31_pilot/` and `phase32_improved/`, and Medium/Large results under `experiments/results/phase33/`. The Pilot overlap ablation is under `phase34_ablation/`. `experiments/results/final_gap_completion/` provides the final metric derivation, four-V analysis, proposal compliance check, and experimental review. Historical scripts such as `src/phase33_run.ps1` and `src/phase31_evaluate.py` are reproduction entry points for a separate fresh run; they are not required for ordinary application use and should not be run over frozen results. See `phase_3_3_summary.md` for the original run protocol.

## Limitations

TREC-COVID qrels leave many retrieved papers unjudged; unjudged does not mean irrelevant. Recall is conditional on known positives in each frozen corpus. There is no single measured full offline read-to-index throughput, concurrent QPS test, or six-configuration LLM-cost comparison. The 20-question answer evaluation and later claim audits are limited by non-blind or incomplete semantic review; valid citation IDs alone do not establish factual support. Experimental retrieval and evidence modules have not demonstrated a sufficiently stable benefit to replace the current V1/BGE/Fixed/Top-3 default. See `experiments/results/final_gap_completion/phase_final_gap_completion_summary.md` for the remaining proposal and rubric gaps.
