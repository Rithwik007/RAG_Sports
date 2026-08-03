from qdrant_client import QdrantClient

# Connect to in-memory Qdrant (same as app.py)
client = QdrantClient(location=":memory:")

# Get all collections
collections = client.get_collections()
print("Available collections:")
for collection in collections.collections:
    print(f"  - {collection.name}")

# Try to get info about enterprise_docs
try:
    info = client.get_collection("enterprise_docs")
    print(f"\nenterprise_docs info:")
    print(f"  Points: {info.points_count}")
    print(f"  Vector size: {info.config.params.vectors.size}")
except Exception as e:
    print(f"\nenterprise_docs not found: {e}")
