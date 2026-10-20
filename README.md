# TextLens — Similarity Workspace

A responsive, Turnitin-inspired **two-document similarity comparison tool**, built independently and not affiliated with Turnitin.

## Start locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Features

- Two text editors, sample input, word/character counts
- TF-IDF (unigrams + bigrams) with cosine similarity, calculated by Python/scikit-learn
- Exact shared four-word phrases (case-insensitive) and word coverage percentages
- Responsive dashboard, analysis report, and export to plain text
- API: `POST /api/compare` with JSON `{ "text_a": "...", "text_b": "..." }`

## Limitations / ethical use

This is a **pairwise text comparison tool**, not a plagiarism verdict. The score is not a probability of plagiarism, not directly comparable to a Turnitin originality score, and not an AI authorship detector. It does not check internet pages, journals, or institutional archives. It does not have the source databases, semantic retrieval pipelines, citation handling, or safeguards used by enterprise systems. A match can be legitimate quotation or standard technical language, and paraphrasing may not be picked up. The API accepts up to 100,000 characters per field. For internet deployment, disable Flask debug mode and add proper production configuration, reverse proxy, rate limits, authentication where appropriate, and a privacy policy.
