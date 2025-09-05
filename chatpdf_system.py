"""
ChatPDF System - Tiered PDF Search with OpenAI Embeddings
A standalone system for processing PDFs and answering questions using tiered search.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import chromadb
import PyPDF2
import openai
import logging

logger = logging.getLogger(__name__)

class TieredPDFSearchSystem:
    def __init__(
        self, 
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-large",
        answer_model: str = "gpt-4o",
        chroma_persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the tiered PDF search system.
        
        Args:
            openai_api_key: OpenAI API key for embeddings, summarization and final answer generation
            embedding_model: OpenAI embedding model to use
            answer_model: OpenAI model for generating final answers
            chroma_persist_directory: Directory to persist ChromaDB data
        """
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.answer_model = answer_model
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        
        # Create collections
        try:
            self.doc_collection = self.chroma_client.get_collection("document_summaries")
        except:
            self.doc_collection = self.chroma_client.create_collection("document_summaries")
            
        try:
            self.page_collection = self.chroma_client.get_collection("document_pages")
        except:
            self.page_collection = self.chroma_client.create_collection("document_pages")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # OpenAI has a limit on batch size, so we might need to chunk
            embeddings = []
            batch_size = 100  # Conservative batch size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise e

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        Extract full text and page-by-page text from PDF.
        
        Returns:
            Tuple of (full_text, list_of_page_texts)
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    page_texts.append(page_text)
                    full_text += page_text + "\n"
                
                return full_text.strip(), page_texts
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return "", []

    def generate_document_summary(self, full_text: str, max_tokens: int = 500) -> str:
        """
        Generate a summary of the document using OpenAI.
        """
        try:
            # Truncate text if too long (rough token estimation: 1 token ≈ 4 chars)
            max_chars = 12000  # Roughly 3000 tokens, leaving room for prompt
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "..."
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for summaries
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that creates concise, informative summaries of documents. Focus on the main topics, key findings, and overall purpose of the document."
                    },
                    {
                        "role": "user", 
                        "content": f"Please provide a comprehensive summary of the following document:\n\n{full_text}"
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback: use first 500 words as summary
            words = full_text.split()
            return " ".join(words[:500]) + "..." if len(words) > 500 else full_text

    def generate_file_id(self, file_path: str) -> str:
        """Generate a unique ID for a file based on its path and modification time."""
        stat = os.stat(file_path)
        unique_string = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def get_processing_stats(self) -> Dict:
        """Get statistics about processed documents."""
        try:
            # Get document count
            doc_count = self.doc_collection.count()
            
            # Get page count
            page_count = self.page_collection.count()
            
            # Get list of processed files
            doc_results = self.doc_collection.get()
            processed_files = []
            
            if doc_results['metadatas']:
                processed_files = [
                    {
                        'file_name': meta['file_name'],
                        'total_pages': meta['total_pages']
                    }
                    for meta in doc_results['metadatas']
                ]
            
            return {
                'document_count': doc_count,
                'page_count': page_count,
                'processed_files': processed_files
            }
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {
                'document_count': 0,
                'page_count': 0,
                'processed_files': []
            }

    def process_pdf_folder(self, folder_path: str, force_reprocess: bool = False) -> Dict:
        """
        Process all PDFs in a folder and add them to ChromaDB collections.
        
        Args:
            folder_path: Path to folder containing PDFs
            force_reprocess: If True, reprocess even if document already exists
            
        Returns:
            Dictionary with processing results
        """
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        
        results = {
            'total_files': len(pdf_files),
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'errors': []
        }
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                file_id = self.generate_file_id(str(pdf_path))
                
                # Check if document already processed (unless force_reprocess)
                if not force_reprocess:
                    try:
                        existing = self.doc_collection.get(ids=[file_id])
                        if existing['ids']:
                            logger.info(f"Skipping {pdf_path.name} (already processed)")
                            results['skipped_files'] += 1
                            continue
                    except:
                        pass  # Document doesn't exist, continue processing
                
                logger.info(f"Processing {pdf_path.name}...")
                
                # Extract text
                full_text, page_texts = self.extract_text_from_pdf(str(pdf_path))
                
                if not full_text.strip():
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    results['failed_files'] += 1
                    results['errors'].append(f"No text extracted from {pdf_path.name}")
                    continue
                
                # Generate document summary
                summary = self.generate_document_summary(full_text)
                
                # Generate embeddings
                summary_embedding = self.generate_single_embedding(summary)
                
                # Generate page embeddings (batch process for efficiency)
                non_empty_pages = [page_text for page_text in page_texts if page_text.strip()]
                if non_empty_pages:
                    page_embeddings = self.generate_embeddings(non_empty_pages)
                else:
                    page_embeddings = []
                
                # Add document summary to collection
                self.doc_collection.upsert(
                    embeddings=[summary_embedding],
                    metadatas=[{
                        "file_path": str(pdf_path),
                        "file_name": pdf_path.name,
                        "file_id": file_id,
                        "total_pages": len(page_texts),
                        "summary": summary
                    }],
                    documents=[summary],
                    ids=[file_id]
                )
                
                # Add pages to collection
                page_ids = []
                page_metadatas = []
                page_documents = []
                page_embeddings_to_store = []
                
                embedding_index = 0
                for page_num, page_text in enumerate(page_texts, 1):
                    if page_text.strip():  # Only add non-empty pages
                        page_id = f"{file_id}_page_{page_num}"
                        page_ids.append(page_id)
                        page_metadatas.append({
                            "file_id": file_id,
                            "file_name": pdf_path.name,
                            "file_path": str(pdf_path),
                            "page_number": page_num,
                            "total_pages": len(page_texts)
                        })
                        page_documents.append(page_text)
                        page_embeddings_to_store.append(page_embeddings[embedding_index])
                        embedding_index += 1
                
                if page_ids:  # Only upsert if we have pages to add
                    self.page_collection.upsert(
                        embeddings=page_embeddings_to_store,
                        metadatas=page_metadatas,
                        documents=page_documents,
                        ids=page_ids
                    )
                
                logger.info(f"Successfully processed {pdf_path.name}: {len(page_ids)} pages")
                results['processed_files'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed_files'] += 1
                results['errors'].append(f"Error processing {pdf_path.name}: {str(e)}")
        
        return results

    def tiered_search(
        self, 
        query: str, 
        top_documents: int = 3, 
        pages_per_document: int = 5
    ) -> Dict:
        """
        Perform tiered search: first find relevant documents, then relevant pages.
        
        Args:
            query: User search query
            top_documents: Number of top documents to retrieve
            pages_per_document: Number of pages to retrieve per document
            
        Returns:
            Dictionary containing search results
        """
        # Generate query embedding
        query_embedding = self.generate_single_embedding(query)
        
        # Tier 1: Find relevant documents
        logger.info("Tier 1: Searching for relevant documents...")
        doc_results = self.doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_documents
        )
        
        if not doc_results['ids'][0]:
            return {"documents": [], "pages": [], "query": query}
        
        relevant_file_ids = [
            metadata["file_id"] 
            for metadata in doc_results["metadatas"][0]
        ]
        
        logger.info(f"Found {len(relevant_file_ids)} relevant documents")
        
        # Tier 2: Find relevant pages within those documents
        logger.info("Tier 2: Searching for relevant pages...")
        all_pages = []
        
        for file_id in relevant_file_ids:
            page_results = self.page_collection.query(
                query_embeddings=[query_embedding],
                where={"file_id": file_id},
                n_results=pages_per_document
            )
            
            # Add pages from this document
            for i in range(len(page_results["ids"][0])):
                all_pages.append({
                    "page_id": page_results["ids"][0][i],
                    "content": page_results["documents"][0][i],
                    "metadata": page_results["metadatas"][0][i],
                    "score": page_results["distances"][0][i]
                })
        
        # Sort all pages by relevance score
        all_pages.sort(key=lambda x: x["score"])
        
        # Prepare document information
        documents_info = []
        for i in range(len(doc_results["ids"][0])):
            documents_info.append({
                "doc_id": doc_results["ids"][0][i],
                "file_name": doc_results["metadatas"][0][i]["file_name"],
                "summary": doc_results["documents"][0][i],
                "score": doc_results["distances"][0][i]
            })
        
        return {
            "query": query,
            "documents": documents_info,
            "pages": all_pages,
            "total_documents_found": len(documents_info),
            "total_pages_found": len(all_pages)
        }

    def generate_answer(self, query: str, search_results: Dict, max_context_length: int = 12000) -> str:
        """
        Generate an answer to the query using the search results.
        
        Args:
            query: Original user query
            search_results: Results from tiered_search
            max_context_length: Maximum length of context to send to LLM
            
        Returns:
            Generated answer
        """
        if not search_results["pages"]:
            return "I couldn't find any relevant information in the documents to answer your query. Please try rephrasing your question or check if the relevant documents have been uploaded."
        
        # Prepare context from relevant pages
        context_parts = []
        current_length = 0
        
        for page in search_results["pages"]:
            page_context = f"[From {page['metadata']['file_name']}, Page {page['metadata']['page_number']}]\n{page['content']}\n"
            
            if current_length + len(page_context) > max_context_length:
                break
                
            context_parts.append(page_context)
            current_length += len(page_context)
        
        context = "\n".join(context_parts)
        
        # Generate answer using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=self.answer_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided document excerpts. Use only the information from the provided context to answer questions. If the context doesn't contain enough information to answer the question, say so clearly. Always cite which document and page number your information comes from when you reference specific facts or findings. Provide comprehensive answers when possible, but be concise and well-structured."
                    },
                    {
                        "role": "user",
                        "content": f"Context from documents:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above."
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"

    def search_and_answer(
        self, 
        query: str, 
        top_documents: int = 3, 
        pages_per_document: int = 5
    ) -> Dict:
        """
        Complete pipeline: search and generate answer.
        
        Returns:
            Dictionary with search results and generated answer
        """
        # Perform tiered search
        search_results = self.tiered_search(query, top_documents, pages_per_document)
        
        # Generate answer
        answer = self.generate_answer(query, search_results)
        
        return {
            "query": query,
            "answer": answer,
            "search_results": search_results
        }

