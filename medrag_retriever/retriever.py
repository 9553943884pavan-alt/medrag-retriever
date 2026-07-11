from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from pydantic import Field
from typing import List


class HybridMedRAGRetriever(BaseRetriever):
    client: object = Field(...,description= "Qdrant Client Instance for vector Database")
    dense_model: object = Field(..., description= "Sentence Transformers Model for Dense Embedding")
    sparse_model: object = Field(..., description= "Sparse Embeddings generator for BM25 Retriever")
    reranker: object = Field(..., description= "Rerank the Retrieved documents based How relevant to Query")
    collection_name: str = Field(default="medrag_textbooks")
    top_k_fetch: int = Field(default=20)
    top_n_final: int = Field(default=3)
    min_rerank_score: float = Field(default=0.25)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_dense = self.dense_model.encode(
            [query], normalize_embeddings=True
        )[0].astype("float32").tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=query_dense, using="dense", limit=self.top_k_fetch),
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=self.top_k_fetch
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.top_k_fetch,
            with_payload=True
        )

        candidates = [
            {"text": p.payload["text"], "source": p.payload["source"], "chunk_id": p.payload["chunk_id"]}
            for p in results.points
        ]
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        filtered = [(c, s) for c, s in ranked if float(s) >= self.min_rerank_score]
        top_results = filtered[: self.top_n_final]

        return [
            Document(
                page_content=cand["text"],
                metadata={"source": cand["source"], "chunk_id": cand["chunk_id"], "rerank_score": float(score)}
            )
            for cand, score in top_results
        ]
