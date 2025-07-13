import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available. Basic text search will be used.")

class VectorMemoryStore:
    """Vector-based memory storage using Sentence Transformers and cosine similarity.
    
    This implementation provides a simple in-memory vector store with optional persistence.
    It uses scikit-learn's cosine similarity for efficient similarity search.
    """
    
    def __init__(self, persist_directory: str = "data/vector_store", model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector memory store with enhanced context retrieval.
        
        Args:
            persist_directory: Directory to store the vector store data
            model_name: Name of the sentence transformer model to use
        """
        self.persist_directory = Path(persist_directory)
        self.model_name = model_name
        
        # Initialize in-memory storage with enhanced metadata
        self.embeddings = []  # List of numpy arrays
        self.metadatas = []   # List of metadata dicts with timestamps
        self.documents = []   # List of document texts
        self.ids = []         # List of document IDs
        self.conversation_maps = defaultdict(list)  # conversation_id -> list of memory indices
        self.next_id = 0
        
        # Time decay parameters (in hours)
        self.time_decay_half_life = 24  # Half-life of memory relevance in hours
        
        # Initialize embedding model if available
        self._transformers_available = TRANSFORMERS_AVAILABLE
        if self._transformers_available:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Initialized embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {str(e)}")
                self._transformers_available = False
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing data
        self._load_from_disk()
    
    class SafeUnpickler(pickle.Unpickler):
        """Custom unpickler that can handle our custom pickling format."""
        def persistent_load(self, pid):
            # Handle our custom persistent IDs
            if pid[0] == 'NUMPY_ARRAY':
                return np.array(pid[1], dtype=np.float32)
            elif pid[0] == 'PYTHON_OBJECT':
                return str(pid[1])
            else:
                raise pickle.UnpicklingError(f"Unsupported persistent object: {pid}")
    
    def _load_from_disk(self):
        """Load data from disk if it exists, supporting both JSON and pickle formats."""
        data_file = self.persist_directory / "vector_store.pkl"
        if not data_file.exists():
            logger.info("No existing vector store found, starting fresh.")
            return False
            
        try:
            # First try to load as JSON
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Fall back to pickle if JSON fails
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
            
            # Check version for compatibility
            version = data.get('version', '0.0')
            
            # Load basic data
            self.documents = data.get('documents', [])
            self.ids = data.get('ids', [])
            self.next_id = data.get('next_id', 0)
            self.model_name = data.get('model_name', self.model_name)
            
            # Convert embeddings back to numpy arrays if they were saved as lists
            saved_embeddings = data.get('embeddings', [])
            self.embeddings = []
            for emb in saved_embeddings:
                if isinstance(emb, list):
                    self.embeddings.append(np.array(emb, dtype=np.float32))
                elif hasattr(emb, 'tolist'):  # Handle numpy arrays directly
                    self.embeddings.append(emb)
                else:
                    logger.warning(f"Unexpected embedding type: {type(emb)}")
                    self.embeddings.append(emb)
            
            # Load metadata
            self.metadatas = data.get('metadatas', [])
            
            # Rebuild conversation maps
            self.conversation_maps = defaultdict(list)
            for idx, meta in enumerate(self.metadatas):
                if isinstance(meta, dict) and 'conversation_id' in meta:
                    conv_id = meta['conversation_id']
                    self.conversation_maps[conv_id].append(idx)
        
            logger.info(f"Loaded {len(self.ids)} vectors from {data_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store from disk: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try to recover by moving the corrupted file
            try:
                import time
                corrupted_file = self.persist_directory / f"vector_store_corrupted_{int(time.time())}.pkl"
                data_file.rename(corrupted_file)
                logger.warning(f"Moved corrupted vector store to {corrupted_file}")
            except Exception as e:
                logger.error(f"Failed to move corrupted file: {str(e)}")
                
            # Reset to empty state
            self._reset_state()
            return False
    
    def _reset_state(self):
        """Reset the vector store to an empty state."""
        self.embeddings = []
        self.metadatas = []
        self.documents = []
        self.ids = []
        self.next_id = 0
        self.conversation_maps = defaultdict(list)
    
    def add_memory(self, conversation_id: str, text: str, metadata: Optional[Dict] = None) -> str:
        """Add a memory to the store with enhanced metadata.
        
        Args:
            conversation_id: ID of the conversation
            text: The text to store
            metadata: Optional metadata about the memory
            
        Returns:
            str: ID of the stored memory
        """
        # Generate a unique ID
        memory_id = f"{conversation_id}_{self.next_id}"
        self.next_id += 1
        
        # Prepare metadata with timestamp and conversation info
        if metadata is None:
            metadata = {}
            
        current_time = time.time()
        metadata.update({
            'conversation_id': conversation_id,
            'timestamp': current_time,
            'last_accessed': current_time,
            'access_count': 0,
            'type': metadata.get('type', 'generic')
        })
        
        # Generate embedding if possible
        embedding = None
        if self._transformers_available:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
        
        # Store the memory
        idx = len(self.ids)
        self.ids.append(memory_id)
        self.documents.append(text)
        self.metadatas.append(metadata)
        self.embeddings.append(embedding)
        self.conversation_maps[conversation_id].append(idx)
        
        # Persist to disk
        self._save_to_disk()
        
        return memory_id
    
    def _calculate_time_decay(self, timestamp: float, current_time: Optional[float] = None) -> float:
        """Calculate time decay factor for a memory based on its age.
        
        Args:
            timestamp: Timestamp of the memory
            current_time: Current time (defaults to now)
            
        Returns:
            float: Decay factor between 0 and 1
        """
        if current_time is None:
            current_time = time.time()
            
        # Calculate age in hours
        age_hours = (current_time - timestamp) / 3600
        
        # Exponential decay with half-life
        decay = 0.5 ** (age_hours / self.time_decay_half_life)
        return decay
    
    def _calculate_relevance_score(self, 
                                 similarity: float, 
                                 metadata: Dict, 
                                 current_time: Optional[float] = None) -> float:
        """Calculate combined relevance score with time decay and access patterns.
        
        Args:
            similarity: Semantic similarity score (0-1)
            metadata: Memory metadata
            current_time: Current timestamp
            
        Returns:
            float: Combined relevance score (0-1)
        """
        if current_time is None:
            current_time = time.time()
            
        # Time-based decay
        time_factor = self._calculate_time_decay(metadata['timestamp'], current_time)
        
        # Recency boost for recently accessed memories
        recency_factor = 0.0
        if 'last_accessed' in metadata:
            recency_factor = self._calculate_time_decay(metadata['last_accessed'], current_time)
        
        # Frequency boost for frequently accessed memories
        frequency_factor = 0.0
        if 'access_count' in metadata and metadata['access_count'] > 0:
            frequency_factor = 1.0 - (1.0 / (1.0 + metadata['access_count']))
        
        # Combine factors with weights
        # Weights can be tuned based on application needs
        weights = {
            'similarity': 0.6,
            'time': 0.2,
            'recency': 0.1,
            'frequency': 0.1
        }
        
        score = (
            weights['similarity'] * similarity +
            weights['time'] * time_factor +
            weights['recency'] * recency_factor +
            weights['frequency'] * frequency_factor
        )
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    def _update_access_stats(self, idx: int, current_time: Optional[float] = None) -> None:
        """Update access statistics for a memory.
        
        Args:
            idx: Index of the memory
            current_time: Current timestamp
        """
        if current_time is None:
            current_time = time.time()
            
        if 0 <= idx < len(self.metadatas):
            self.metadatas[idx]['last_accessed'] = current_time
            self.metadatas[idx]['access_count'] = self.metadatas[idx].get('access_count', 0) + 1
    
    def search_memories(self, 
                       query: str, 
                       conversation_id: Optional[str] = None, 
                       limit: int = 5,
                       min_score: float = 0.3) -> List[Dict]:
        """Search for similar memories with enhanced relevance scoring.
        
        Args:
            query: The search query
            conversation_id: Optional conversation ID to filter by
            limit: Maximum number of results to return
            min_score: Minimum relevance score (0-1) for results
            
        Returns:
            List of dictionaries containing memory text, metadata, and relevance score
        """
        if not self.embeddings:
            return []
            
        current_time = time.time()
        
        # Filter by conversation if specified
        indices = list(range(len(self.embeddings)))
        if conversation_id is not None:
            indices = self.conversation_maps.get(conversation_id, [])
            if not indices:
                return []
        
        results = []
        
        # If we have embeddings, use semantic search
        if self._transformers_available and self.embeddings[0] is not None:
            try:
                # Get query embedding
                query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
                
                # Calculate similarities and relevance scores
                for idx in indices:
                    if self.embeddings[idx] is not None:
                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            self.embeddings[idx].reshape(1, -1)
                        )[0][0]
                        
                        # Calculate combined relevance score
                        relevance = self._calculate_relevance_score(
                            similarity=similarity,
                            metadata=self.metadatas[idx],
                            current_time=current_time
                        )
                        
                        if relevance >= min_score:
                            results.append({
                                'text': self.documents[idx],
                                'metadata': self.metadatas[idx],
                                'similarity': similarity,
                                'relevance': relevance,
                                'index': idx
                            })
                
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                # Fall through to text-based search
        
        # Fallback to text-based search if no semantic results
        if not results:
            query = query.lower()
            for idx in indices:
                text = self.documents[idx].lower()
                if query in text:
                    # Simple relevance score based on query term frequency
                    term_freq = text.count(query)
                    similarity = min(1.0, term_freq * 0.1)  # Cap at 1.0
                    
                    relevance = self._calculate_relevance_score(
                        similarity=similarity,
                        metadata=self.metadatas[idx],
                        current_time=current_time
                    )
                    
                    if relevance >= min_score:
                        results.append({
                            'text': self.documents[idx],
                            'metadata': self.metadatas[idx],
                            'similarity': similarity,
                            'relevance': relevance,
                            'index': idx
                        })
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Update access stats for top results
        for result in results[:limit]:
            self._update_access_stats(result['index'], current_time)
        
        # Return results with limited fields
        return [{
            'text': r['text'],
            'metadata': r['metadata'],
            'score': r['relevance']
        } for r in results[:limit]]
    
    def get_conversation_memories(self, 
                                conversation_id: str, 
                                limit: int = 10,
                                recent_first: bool = True) -> List[Dict]:
        """Get all memories for a specific conversation with enhanced sorting.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of memories to return
            recent_first: Whether to return most recent memories first
            
        Returns:
            List of conversation memories with metadata and relevance scores
        """
        if conversation_id not in self.conversation_maps:
            return []
            
        current_time = time.time()
        results = []
        
        # Get all memory indices for this conversation
        for idx in self.conversation_maps[conversation_id]:
            if 0 <= idx < len(self.metadatas):
                metadata = self.metadatas[idx]
                
                # Calculate relevance based on recency and access patterns
                time_factor = self._calculate_time_decay(
                    metadata.get('timestamp', 0), 
                    current_time
                )
                
                # Boost for frequently accessed memories
                access_count = metadata.get('access_count', 0)
                frequency_factor = 1.0 - (1.0 / (1.0 + access_count))
                
                # Combine factors (adjust weights as needed)
                relevance = 0.7 * time_factor + 0.3 * frequency_factor
                
                results.append({
                    'text': self.documents[idx],
                    'metadata': metadata,
                    'relevance': relevance,
                    'index': idx
                })
        
        # Sort by relevance or timestamp
        if recent_first:
            results.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
        else:
            results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Update access stats for returned results
        for result in results[:limit]:
            self._update_access_stats(result['index'], current_time)
        
        # Return results with limited fields
        return [{
            'text': r['text'],
            'metadata': r['metadata'],
            'score': r['relevance']
        } for r in results[:limit]]
    
    def delete_conversation_memories(self, conversation_id: str) -> int:
        """Delete all memories for a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            int: Number of memories deleted
        """
        # Find indices of memories to delete (in reverse order to avoid index shifting)
        to_delete = []
        for i in reversed(range(len(self.ids))):
            if self.metadatas[i].get('conversation_id') == conversation_id:
                to_delete.append(i)
        
        # Delete from all lists
        for idx in to_delete:
            del self.ids[idx]
            del self.documents[idx]
            del self.metadatas[idx]
            if idx < len(self.embeddings):
                del self.embeddings[idx]
        
        # Rebuild FAISS index if needed
        if self.index is not None and self.embeddings:
            self._rebuild_faiss_index()
        
        # Persist changes
        self._save_to_disk()
        
        return len(to_delete)
    

    
    def persist(self):
        """Persist the vector store to disk."""
        self._save_to_disk()
    
    class SafePickler(pickle.Pickler):
        """Custom pickler that safely handles complex objects."""
        def persistent_id(self, obj):
            # Handle numpy arrays
            if hasattr(obj, 'tolist'):
                return ('NUMPY_ARRAY', obj.tolist())
            # Handle other non-serializable objects
            elif not isinstance(obj, (str, int, float, bool, list, dict, tuple, type(None))):
                return ('PYTHON_OBJECT', str(obj))
            else:
                return None
    
    def _make_serializable(self, obj, max_depth=10):
        """Recursively convert an object to a serializable format.
        
        Args:
            obj: The object to make serializable
            max_depth: Maximum recursion depth to prevent stack overflow
            
        Returns:
            A serializable version of the object
        """
        if max_depth < 0:
            return str(obj)
            
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple, set)):
            return [self._make_serializable(x, max_depth-1) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v, max_depth-1) 
                   for k, v in obj.items()}
        elif hasattr(obj, 'tolist'):  # For numpy arrays
            return obj.tolist()
        else:
            return str(obj)
    
    def _save_to_disk(self):
        """Save the current state to disk using a custom pickler for safety."""
        if not hasattr(self, 'persist_directory'):
            return False
        
        try:
            # Create a simplified data structure for serialization
            data_to_save = {
                'version': '3.0',  # New version for simplified serialization
                'ids': self.ids,
                'documents': self.documents,
                'next_id': self.next_id,
                'model_name': self.model_name,
                'metadatas': [],
                'embeddings': []
            }
            
            # Convert embeddings to lists
            for emb in self.embeddings:
                if hasattr(emb, 'tolist'):
                    data_to_save['embeddings'].append(emb.tolist())
                else:
                    data_to_save['embeddings'].append(emb)
            
            # Process metadata to ensure it's serializable
            for meta in self.metadatas:
                data_to_save['metadatas'].append(self._make_serializable(meta))
            
            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Use a temporary file for atomic writes
            temp_file = self.persist_directory / "vector_store.pkl.tmp"
            final_file = self.persist_directory / "vector_store.pkl"
            
            # Convert to JSON-serializable format first
            json_serializable = self._make_serializable(data_to_save)
            
            # Save as JSON instead of pickle for better compatibility
            with open(temp_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(json_serializable, f, ensure_ascii=False, indent=2)
            
            # Atomic rename on POSIX systems
            if temp_file.exists():
                if final_file.exists():
                    final_file.unlink()
                temp_file.rename(final_file)
            
            logger.info(f"Saved {len(self.ids)} vectors to {final_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store to disk: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Clean up any temporary files
            if 'temp_file' in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to clean up temp file: {str(e)}")
                    
            return False
