import sys
from pathlib import Path
import pickle
import numpy as np

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Load the vector store
vector_store_path = Path('data/vector_store/vector_store.pkl')
if not vector_store_path.exists():
    print("Vector store file not found!")
    sys.exit(1)

# Load the vector store
with open(vector_store_path, 'rb') as f:
    data = pickle.load(f)

print(f"Found {len(data['embeddings'])} vectors in the store")

# Print document metadata
print("\nDocument entries:")
for i, (text, meta) in enumerate(zip(data['documents'], data['metadatas'])):
    if meta.get('type') == 'document':
        print(f"\nDocument {i+1}:")
        print(f"Source: {meta.get('source')}")
        print(f"Text preview: {text[:200]}...")
        print(f"Metadata: {meta}")
