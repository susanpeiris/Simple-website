"""TextLens: educational text similarity checker.

TF-IDF downweights common terms and assigns higher weight to distinctive terms.
Cosine similarity measures the angle between two TF-IDF vectors; values near
1 indicate similar distributions of terms. This is NOT a plagiarism or AI detector.
"""
import re
from collections import Counter
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
MAX_CHARS = 100_000
WORD_RE = re.compile(r"\b[\w]+(?:['’][\w]+)?\b", re.UNICODE)


def tokenize(text):
    return [m.group(0).casefold() for m in WORD_RE.finditer(text)]


def overlapping_phrases(a, b, n=4, limit=40):
    """Find shared consecutive word n-grams and return unique matches."""
    aw, bw = tokenize(a), tokenize(b)
    bgrams = {tuple(bw[i:i+n]) for i in range(max(0, len(bw)-n+1))}
    found = []
    seen = set()
    for i in range(max(0, len(aw)-n+1)):
        gram = tuple(aw[i:i+n])
        if gram in bgrams and gram not in seen:
            seen.add(gram)
            found.append(' '.join(gram))
    return found[:limit]


def matched_word_coverage(a, b, n=4):
    """Fraction of words in each document covered by a matching n-gram."""
    aw, bw = tokenize(a), tokenize(b)
    if not aw or not bw:
        return 0, 0
    ag = {tuple(aw[i:i+n]) for i in range(max(0, len(aw)-n+1))}
    bg = {tuple(bw[i:i+n]) for i in range(max(0, len(bw)-n+1))}
    common = ag & bg
    def covered(words):
        indices = set()
        for i in range(max(0, len(words)-n+1)):
            if tuple(words[i:i+n]) in common:
                indices.update(range(i, i+n))
        return round(100 * len(indices) / len(words), 1)
    return covered(aw), covered(bw)


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/api/compare')
def compare():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Please provide a JSON object.'), 400
    a, b = data.get('text_a'), data.get('text_b')
    if not isinstance(a, str) or not isinstance(b, str):
        return jsonify(error='Both texts must be strings.'), 400
    if not a.strip() or not b.strip():
        return jsonify(error='Enter text in both fields to compare.'), 400
    if len(a) > MAX_CHARS or len(b) > MAX_CHARS:
        return jsonify(error=f'Each text must be under {MAX_CHARS:,} characters.'), 413
    # TF-IDF on unigrams and bigrams captures both individual terms and adjacent pairs.
    # A small sample of two documents cannot provide a comprehensive corpus IDF model.
    try:
        v = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents='unicode')
        vectors = v.fit_transform([a, b])
        similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except ValueError:
        similarity = 0.0  # e.g. content containing only punctuation
    phrases = overlapping_phrases(a, b)
    cov_a, cov_b = matched_word_coverage(a, b)
    return jsonify(
        similarity=round(similarity * 100, 1),
        overlap_phrases=phrases,
        overlap_count=len(phrases),
        coverage_a=cov_a,
        coverage_b=cov_b,
        words_a=len(tokenize(a)),
        words_b=len(tokenize(b)),
        methodology='TF-IDF (unigrams + bigrams) and cosine similarity',
        disclaimer='Compares only the two submitted texts. It does not search the web, detect AI writing, or establish plagiarism.'
    )


if __name__ == '__main__':
    app.run(debug=True)
