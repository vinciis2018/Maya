import os
import sys
import logging
from pathlib import Path

# Add the server directory to the path so we can import the vector store
sys.path.append(str(Path(__file__).parent / 'server'))
from services.vector_store import VectorMemoryStore

def test_vector_store():
    # Set up test data
    test_dir = "test_vector_store"
    conversation_id = "test_conversation_1"
    
    # Clean up any existing test data
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    # Initialize the vector store
    print("Initializing vector store...")
    vector_store = VectorMemoryStore(persist_directory=test_dir)
    
    # Add some test memories
    print("\nAdding test memories...")
    test_memories = [
        "The quick brown fox jumps over the lazy dog.",
        "I love programming in Python.",
        "Machine learning is fascinating.",
        "The weather is nice today.",
        "I need to buy groceries."
    ]
    
    for i, text in enumerate(test_memories):
        memory_id = vector_store.add_memory(
            conversation_id=conversation_id,
            text=text,
            metadata={"index": i, "source": "test"}
        )
        print(f"Added memory {i+1}: {text[:50]}... (ID: {memory_id})")
    
    # Test searching
    print("\nTesting search functionality...")
    queries = ["programming", "animals", "shopping"]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = vector_store.search_memories(query, conversation_id=conversation_id)
        
        if not results:
            print("  No results found.")
        else:
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['text']} (Score: {result['score']:.4f})")
    
    # Test getting all memories for the conversation
    print("\nTesting get_conversation_memories...")
    all_memories = vector_store.get_conversation_memories(conversation_id)
    print(f"Found {len(all_memories)} memories for conversation '{conversation_id}':")
    for i, memory in enumerate(all_memories, 1):
        print(f"  {i}. {memory['text']}")
    
    # Test persistence
    print("\nTesting persistence...")
    print("  Persisting data...")
    vector_store.persist()
    
    print("  Creating a new vector store instance...")
    new_vector_store = VectorMemoryStore(persist_directory=test_dir)
    new_memories = new_vector_store.get_conversation_memories(conversation_id)
    
    print(f"  Loaded {len(new_memories)} memories from disk.")
    print("  Original memory count:", len(all_memories))
    print("  Loaded memory count: ", len(new_memories))
    
    # Clean up
    print("\nCleaning up test data...")
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_vector_store()
