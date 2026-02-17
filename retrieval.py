import os
from typing import List
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder

class LocalHybridRerankRetriever(BaseRetriever):
    vectorstore: Chroma
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 4
    
    _rerank_model: CrossEncoder = None
    _bm25_retriever: BM25Retriever = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rerank_model = CrossEncoder(self.rerank_model_name)
        self._initialize_bm25()

    def _initialize_bm25(self):
        all_docs = self.vectorstore.get()
        if all_docs and all_docs['documents']:
            documents = [
                Document(page_content=text, metadata=meta if meta is not None else {}) 
                for text, meta in zip(all_docs['documents'], all_docs['metadatas'])
            ]
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = 10
        else:
            print("Warning: No documents found in vectorstore to initialize BM25.")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        if not self._bm25_retriever:
            return vector_retriever.invoke(query)[:self.top_k]
            
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self._bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
        
        initial_docs = ensemble_retriever.invoke(query)
        if not initial_docs:
            return []

        pairs = [(query, doc.page_content) for doc in initial_docs]
        scores = self._rerank_model.predict(pairs)
        
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:self.top_k]]

def get_local_retriever(vectorstore: Chroma):
    return LocalHybridRerankRetriever(vectorstore=vectorstore)
