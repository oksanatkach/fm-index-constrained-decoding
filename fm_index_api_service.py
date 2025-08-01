from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import logging
import time
import os
from contextlib import asynccontextmanager

# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Your FM-Index import
from index import FMIndex

# Global variable to hold the loaded index
fm_index: Optional[FMIndex] = None


class SearchRequest(BaseModel):
    sequence: List[int]
    limit: Optional[int] = 100


class SearchResponse(BaseModel):
    count: int
    doc_indices: List[int]
    took_ms: float


class ContinuationRequest(BaseModel):
    sequence: List[int]


class ContinuationResponse(BaseModel):
    continuations: List[int]
    took_ms: float

class CountRequest(BaseModel):
    sub_sequence: List[int]

class CountResponse(BaseModel):
    count: int
    took_ms: float

class RangeRequest(BaseModel):
    sequence: List[int]

class RangeResponse(BaseModel):
    range: Tuple[int, int]
    took_ms: float

class DistinctRequest(BaseModel):
    lows: List[int]
    highs: List[int]

class DistinctResponse(BaseModel):
    distinct_list: List[Tuple[List[int], List[int]]]
    took_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the FM-Index
    global fm_index
    # index_path = os.getenv("FM_INDEX_PATH", "data/mac/enwiki.qwen3_8b.fm_index")
    index_path = os.getenv("FM_INDEX_PATH", "data/PAQ/PAQ_index/PAQ.fm_index")

    logger.info(f"Loading FM-Index from {index_path}...")
    start_time = time.time()

    try:
        fm_index = FMIndex.load(index_path)
        load_time = time.time() - start_time
        logger.info(f"FM-Index loaded successfully in {load_time:.2f}s")
        logger.info(f"Index contains {fm_index.n_docs} documents, {len(fm_index)} tokens")
    except Exception as e:
        logger.error(f"Failed to load FM-Index: {e}")
        raise

    yield

    # Shutdown: Clean up if needed
    logger.info("Shutting down FM-Index service")


app = FastAPI(
    title="FM-Index Search Service",
    description="High-performance substring search service using FM-Index",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")
    return {"status": "healthy", "n_docs": fm_index.n_docs, "n_tokens": len(fm_index)}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for documents containing the given sequence"""
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    start_time = time.time()

    try:
        # Get count
        count = fm_index.get_count(request.sequence)

        # Get document indices (limited)
        doc_indices = list(fm_index.get_doc_indices(request.sequence))
        if request.limit and len(doc_indices) > request.limit:
            doc_indices = doc_indices[:request.limit]

        took_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            count=count,
            doc_indices=doc_indices,
            took_ms=took_ms
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occurring_distinct")
async def get_occurring_distinct():
    """Get index statistics"""
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    return fm_index.occurring_distinct

@app.post("/get_count")
async def get_count(request: CountRequest):
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    start_time = time.time()
    try:
        count = fm_index.get_count(request.sub_sequence)
        took_ms = (time.time() - start_time) * 1000

        return CountResponse(
            count=count,
            took_ms=took_ms
        )
    except Exception as e:
        logger.error(f"Get count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_range")
async def get_range(request: RangeRequest):

    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    start_time = time.time()
    try:
        low, high = fm_index.get_range(request.sequence)
        took_ms = (time.time() - start_time) * 1000

        return RangeResponse(
            range=(low, high),
            took_ms=took_ms
        )
    except Exception as e:
        logger.error(f"Get count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_distinct_count_multi")
async def get_distinct_count_multi(request: DistinctRequest):

    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    start_time = time.time()
    try:
        distinct_list = fm_index.get_distinct_count_multi(request.lows, request.highs)
        took_ms = (time.time() - start_time) * 1000

        return DistinctResponse(
            distinct_list=distinct_list,
            took_ms=took_ms
        )
    except Exception as e:
        logger.error(f"Get count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continuations", response_model=ContinuationResponse)
async def get_continuations(request: ContinuationRequest):
    """Get possible continuations for the given sequence"""
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    start_time = time.time()

    try:
        continuations = fm_index.get_continuations(request.sequence)
        took_ms = (time.time() - start_time) * 1000

        return ContinuationResponse(
            continuations=continuations,
            took_ms=took_ms
        )

    except Exception as e:
        logger.error(f"Continuations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs/{doc_index}")
async def get_document(doc_index: int):
    """Retrieve a document by its index"""
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    if doc_index < 0 or doc_index >= fm_index.n_docs:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        doc = fm_index.get_doc(doc_index)
        return {"doc_index": doc_index, "tokens": doc, "length": len(doc)}

    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get index statistics"""
    if fm_index is None:
        raise HTTPException(status_code=503, detail="FM-Index not loaded")

    return {
        "n_docs": fm_index.n_docs,
        "n_tokens": len(fm_index),
        "occurring_tokens": len(fm_index.occurring),
        "has_labels": fm_index.labels is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
