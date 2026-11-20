# ==========================================================================================================
# Imports
# ==========================================================================================================

import torch

import numpy as np

from fastmcp import FastMCP
from langchain_chroma import Chroma
from PIL import Image
from transformers import CLIPModel, CLIPProcessor



# ==========================================================================================================
# Server declaration
# ==========================================================================================================

server = FastMCP(name="Image2Images-RAG_Server", instructions="Server for image-to-images retrieval of recipes.")



# ==========================================================================================================
# Embedding Models
# ==========================================================================================================

@torch.no_grad()
def embed_images(paths, batch_size=16):
    # ---- CLIP embedding model (512-d) for image + query text ----
    device = "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()

    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))

    return (np.vstack(vecs))



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

def _unwrap(res: dict):
    """
        Parse the output of Chromadb.retrieve:
        Chroma returns lists-of-lists; unwrap the first query.

        Args:
            res (dict): raw result from invoking Chromadb.retrieve().
        Returns:
            List[str]: ids
            List[str]: docs
            List[Dict]: metas
            List[float]: dists
    """
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists



# ==========================================================================================================
# MCP Tools
# ==========================================================================================================

@server.tool()
def retrieve_images_by_image(chromadb_dir: str, query_image_path: str, k: int = 5, where: dict | None = None):
    """
    Function to retrieve similar images from the image database based on a query image,
    with optional metadata filtering.

    Args:
        chromadb_dir (str): the path to the database.
        query_image_path (str): the file path to the query image.
        k (int): the number of top results to return.
        where (dict | None): Optional metadata filtering conditions. For example: where = {"cuisine": "Italian"}. If None, no filtering is applied.
    Returns:
        List[str]: ids
        List[str]: docs
        List[Dict]: metas
        List[float]: dists
    """

    chromadb = Chroma(
        collection_name="food_images",
        persist_directory=chromadb_dir,
    )

    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _unwrap(res)



# ==========================================================================================================
# Entry point
# ==========================================================================================================

# Run the Server with HTTP transport
if __name__ == "__main__":
    print("Starting HTTP MCP Server on http://127.0.0.1:8080")

    server.run(transport="http", host="127.0.0.1", port=8080)