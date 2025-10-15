import math

def create_chunks(data: list[dict], workers: int) -> list[list[dict]]:
    if not data:
        return []

    chunk_size = max(1, len(data) // workers)
    chunks = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks