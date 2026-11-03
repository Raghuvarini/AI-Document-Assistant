
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def fuse(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[int(idx)] = (
                scores.get(int(idx), 0) + 1 / (k + rank)
            )

    return sorted(scores, key=scores.get, reverse=True)


class HybridIndex:

    def __init__(self, chunks, encoder):

        if not chunks:
            raise ValueError('No readable chunks.')

        if len(chunks) > 10000:
            raise ValueError(
                'Limit is 10,000 chunks; upload a smaller document set.'
            )

        self.chunks = chunks
        self.encoder = encoder

        self.vectors = np.asarray(
            encoder.encode(
                [c.text for c in chunks],
                normalize_embeddings=True
            )
        )

        self.bm25 = BM25Okapi(
            [tokenize(c.text) or ['_empty_'] for c in chunks]
        )

    def search(
        self,
        query,
        top_k=5,
        reranker=None,
        min_similarity=0.20
    ):

        q = np.asarray(
            self.encoder.encode(
                [query],
                normalize_embeddings=True
            )
        )[0]

        dense = self.vectors @ q

        lexical = self.bm25.get_scores(tokenize(query))

        n = min(24, len(self.chunks))

        dense_rank = np.argsort(-dense)[:n]

        lexical_rank = [
            i for i in np.argsort(-lexical)[:n]
            if lexical[i] > 0
        ]

        candidates = [
            i for i in fuse([dense_rank, lexical_rank])
            if dense[i] >= min_similarity or lexical[i] > 0
        ][:n]

        rerank_scores = {}

        if reranker is not None and candidates:

            values = reranker.predict(
                [(query, self.chunks[i].text) for i in candidates]
            )

            rerank_scores = dict(
                zip(candidates, map(float, np.atleast_1d(values)))
            )

            candidates.sort(
                key=rerank_scores.get,
                reverse=True
            )

        return [
            {
                'chunk': self.chunks[i],
                'cosine': float(dense[i]),
                'rerank': rerank_scores.get(i)
            }
            for i in candidates[:top_k]
        ]
