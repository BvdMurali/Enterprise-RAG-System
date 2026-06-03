import logging
import os
import uuid
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.vectorstore.chroma_store import ChromaStoreService

logger = logging.getLogger(__name__)


class RetrieverService:
    """
    Enterprise Retriever Service implementing:
    - Parent Document Retrieval (Phase 1)
    - Hybrid Search (BM25 + Dense) via Reciprocal Rank Fusion (Phase 2)
    - Local CPU-Optimized Re-ranking (Phase 2)
    - Lost-in-the-Middle Context Engineering (Phase 3)
    """

    def __init__(self, chroma_store: ChromaStoreService):
        self.settings = get_settings()
        self.chroma_store = chroma_store

        # 1. Parent store configurations
        self.parent_store_dir = str(self.settings.parent_store_path)
        self.fs = LocalFileStore(self.parent_store_dir)
        self.parent_store = create_kv_docstore(self.fs)

        # 2. Document splitters (child tokens match, parent provides context)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=30
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )

        # 3. CPU-optimized Cross-Encoder model (only 80MB footprint)
        logger.info(f"Loading CPU-optimized reranker: {self.settings.reranker_model_name}")
        self.reranker = CrossEncoder(
            self.settings.reranker_model_name, 
            device="cpu",
            automodel_args={
                "use_safetensors": False
            }
        )

        # 4. In-memory cache for BM25 retriever
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._bm25_fitted = False

    def invalidate_bm25_cache(self):
        """Invalidate BM25 cache when database modifications occur."""
        self._bm25_retriever = None
        self._bm25_fitted = False
        logger.info("BM25 cache invalidated.")

    def _get_bm25_retriever(self) -> Optional[BM25Retriever]:
        """Lazy loader for BM25 search. Fits index on all child chunks in database."""
        if self._bm25_fitted:
            return self._bm25_retriever

        try:
            logger.info("Fitting BM25 on current database child chunks...")
            collection_data = self.chroma_store.vector_store.get()
            contents = collection_data.get("documents", [])
            metadatas = collection_data.get("metadatas", [])
            ids = collection_data.get("ids", [])

            if not contents:
                logger.warning("No documents in database. BM25 index is empty.")
                return None

            # Reconstruct list of child Documents
            child_docs = []
            for doc_id, text, meta in zip(ids, contents, metadatas):
                child_docs.append(Document(page_content=text, metadata=meta or {}, id=doc_id))

            self._bm25_retriever = BM25Retriever.from_documents(child_docs)
            self._bm25_retriever.k = 15  # Fetch top 15 candidates
            self._bm25_fitted = True
            logger.info(f"BM25 index successfully fitted on {len(child_docs)} chunks.")
            return self._bm25_retriever
        except Exception as e:
            logger.error(f"Failed to fit BM25 index: {e}", exc_info=True)
            return None

    def add_documents(self, documents: List[Document]) -> int:
        """
        Ingest documents using Parent-Child strategy:
        1. Split pages into parent docs, generate parent IDs, save to local store.
        2. Split each parent into child docs, retain parent doc_id in metadata.
        3. Insert child docs into ChromaDB vector store.
        """
        if not documents:
            return 0

        logger.info(f"Ingesting {len(documents)} document pages with Parent-Child mapping...")
        
        # A. Split into Parent documents
        parent_docs = self.parent_splitter.split_documents(documents)
        
        # B. Map parent ID to document list
        parent_mappings = []
        child_docs = []

        for parent in parent_docs:
            parent_id = str(uuid.uuid4())
            parent.metadata["parent_id"] = parent_id
            
            # Map parent keys as strings for EncoderBackedStore
            parent_mappings.append((parent_id, parent))

            # C. Split parent into child chunks
            child_chunks = self.child_splitter.split_documents([parent])
            for child in child_chunks:
                child.metadata["parent_id"] = parent_id
                child.metadata["source"] = parent.metadata.get("source", "Unknown")
                child.metadata["page"] = parent.metadata.get("page", 0)
                child_docs.append(child)

        # D. Save parents to LocalFileStore via create_kv_docstore
        self.parent_store.mset(parent_mappings)
        logger.info(f"Persisted {len(parent_mappings)} parent documents to file store.")

        # E. Add children to Chroma vector DB
        inserted_ids = self.chroma_store.add_documents(child_docs)
        
        # Invalidate BM25 cache since we added chunks
        self.invalidate_bm25_cache()
        return len(inserted_ids)

    def delete_document_by_source(self, source_filename: str):
        """
        Delete a document's parent records and child vectors.
        """
        logger.info(f"Deleting documents associated with source: {source_filename}")
        
        # Fetch all child nodes to collect parent IDs and vector IDs
        collection_data = self.chroma_store.vector_store.get()
        ids = collection_data.get("ids", [])
        metadatas = collection_data.get("metadatas", [])
        
        ids_to_delete = []
        parents_to_delete = []
        
        for doc_id, meta in zip(ids, metadatas):
            if meta:
                meta_source = os.path.basename(meta.get("source", ""))
                if meta_source == source_filename:
                    ids_to_delete.append(doc_id)
                    parent_id = meta.get("parent_id")
                    if parent_id:
                        parents_to_delete.append(parent_id)

        if ids_to_delete:
            self.chroma_store.vector_store.delete(ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} vector child nodes.")

        if parents_to_delete:
            self.parent_store.mdelete(parents_to_delete)
            logger.info(f"Deleted {len(parents_to_delete)} parent documents.")

        self.invalidate_bm25_cache()

    def rrf(self, dense_results: List[Document], sparse_results: List[Document], c: int = 60) -> List[Document]:
        """Merge search results using Reciprocal Rank Fusion."""
        scores = {}
        unique_docs = {}

        for rank, doc in enumerate(dense_results):
            doc_text = doc.page_content
            scores[doc_text] = scores.get(doc_text, 0.0) + 1.0 / (rank + c)
            doc.metadata["dense_rank"] = rank
            unique_docs[doc_text] = doc

        for rank, doc in enumerate(sparse_results):
            doc_text = doc.page_content
            scores[doc_text] = scores.get(doc_text, 0.0) + 1.0 / (rank + c)
            doc.metadata["sparse_rank"] = rank
            if doc_text not in unique_docs:
                unique_docs[doc_text] = doc

        sorted_texts = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [unique_docs[text] for text in sorted_texts]

    def distribute_lost_in_middle(self, documents: List[Document]) -> List[Document]:
        """
        Distributes retrieved chunks so that the most relevant are placed 
        at the beginning and the end of the context (Lost-in-the-Middle mitigation).
        """
        sorted_docs = sorted(documents, key=lambda x: x.metadata.get("relevance_score", 0.0), reverse=True)
        distributed = [None] * len(sorted_docs)
        
        left = 0
        right = len(sorted_docs) - 1
        
        for idx, doc in enumerate(sorted_docs):
            if idx % 2 == 0:
                distributed[left] = doc
                left += 1
            else:
                distributed[right] = doc
                right -= 1
        return [d for d in distributed if d is not None]

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Helper to evaluate metadata filters (supporting $in, $and, and basename matching for files) in BM25."""
        if not filter_dict:
            return True

        def eval_single(meta_val: Any, filter_val: Any, key: str = "") -> bool:
            # Handle list membership filter ($in)
            if isinstance(filter_val, dict) and "$in" in filter_val:
                allowed = filter_val["$in"]
                return meta_val in allowed if isinstance(allowed, list) else meta_val == allowed
            
            # Normalize path comparison for "source" filter
            if key == "source" and isinstance(meta_val, str) and isinstance(filter_val, str):
                return os.path.basename(meta_val) == os.path.basename(filter_val)
                
            return meta_val == filter_val

        # Handle AND filters
        if "$and" in filter_dict:
            for sub_filter in filter_dict["$and"]:
                for k, v in sub_filter.items():
                    if not eval_single(metadata.get(k), v, k):
                        return False
            return True

        # Handle flat dict filters
        for k, v in filter_dict.items():
            if not eval_single(metadata.get(k), v, k):
                return False
        return True

    def retrieve(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Advanced hybrid retrieval pipeline:
        1. Dense retrieval (Chroma semantic query).
        2. Sparse retrieval (BM25 keyword search).
        3. Reciprocal Rank Fusion (RRF) to combine results.
        4. Cross-Encoder Re-ranking of top candidates on CPU.
        5. Map top child chunks to parent document contexts.
        """
        logger.info(f"RAG Retrieval initiated for query: '{query}'")

        try:
            # 1. Fetch dense search results from Chroma
            dense_results = []
            results = self.chroma_store.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=15,
                filter=filter_dict,
                score_threshold=self.settings.retrieval_score_threshold
            )
            for doc, score in results:
                doc.metadata["relevance_score"] = float(score)
                dense_results.append(doc)

            # 2. Fetch sparse search results from BM25
            sparse_results = []
            bm25 = self._get_bm25_retriever()
            if bm25:
                raw_bm25_results = bm25.invoke(query)
                if filter_dict:
                    for doc in raw_bm25_results:
                        if self._matches_filter(doc.metadata or {}, filter_dict):
                            sparse_results.append(doc)
                else:
                    sparse_results = raw_bm25_results

            if not dense_results and not sparse_results:
                logger.info("No dense or sparse results retrieved.")
                return []

            # 3. Reciprocal Rank Fusion (RRF)
            fused_children = self.rrf(dense_results, sparse_results, c=60)[:10]

            # 4. Cross-Encoder Re-ranking (runs on CPU)
            # Prepend source filename metadata so that queries with candidate names (or document context)
            # match body text chunks that do not repeat the name/context in every sentence.
            pairs = []
            for doc in fused_children:
                source_fn = os.path.basename(doc.metadata.get("source", "Unknown"))
                prepended_text = f"[Document: {source_fn}] {doc.page_content}"
                pairs.append([query, prepended_text])

            rerank_scores = self.reranker.predict(pairs)
            
            for idx, score in enumerate(rerank_scores):
                fused_children[idx].metadata["relevance_score"] = float(score)

            # Sort by re-ranker score descending
            fused_children = [doc for _, doc in sorted(zip(rerank_scores, fused_children), key=lambda x: x[0], reverse=True)]

            # Take top K configured retrievals
            top_children = fused_children[:self.settings.retrieval_top_k]

            # 5. Fetch Parent Documents and return them
            parent_documents = []
            seen_parents = set()

            for child in top_children:
                parent_id = child.metadata.get("parent_id")
                if not parent_id:
                    parent_documents.append(child)
                    continue

                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent_doc_list = self.parent_store.mget([parent_id])
                if parent_doc_list and parent_doc_list[0]:
                    parent_doc = parent_doc_list[0]
                    # Preserve child's page/relevance metadata
                    parent_doc.metadata["page"] = child.metadata.get("page", 0)
                    parent_doc.metadata["relevance_score"] = child.metadata.get("relevance_score", 0.0)
                    parent_documents.append(parent_doc)
                else:
                    parent_documents.append(child)

            # 6. Apply Lost-in-the-Middle sorting
            final_docs = self.distribute_lost_in_middle(parent_documents)
            logger.info(f"Successfully retrieved and sorted {len(final_docs)} parent documents.")
            return final_docs

        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

    def retrieve_formatted_context(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> str:
        """
        Retrieve chunks, map to parent context, format as XML tags, and mitigate Lost-in-the-Middle.
        """
        docs = self.retrieve(query, filter_dict)
        if not docs:
            return "No relevant context found in the uploaded documents."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            chunk_text = f"<Document ID={i} Source={os.path.basename(source)} Page={page}>\n{doc.page_content}\n</Document>"
            context_parts.append(chunk_text)

        return "\n\n".join(context_parts)
