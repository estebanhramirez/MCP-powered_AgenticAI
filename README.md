# AgenticAI-with-MCP
Code repository for an Agentic AI, consisting of an LLM with access to tools, prompts and resources, through the Model Context Protocol (MCP).
<br>

Particularly, we implement a (Multimodal - Image-oriented) RAG workflow that recommends users recipes, based on a given photo (image) of the dish that the user would like to cook.
<br>

If we would like to construct an indexing system (which is) capable of retrieving the recipe that produces the dish that "looks" the most similar to a given photo, then our indexing system must use the images of the dishes to compare recipes.

This is why we use the recipe names as `chromadb.documents` (since this is what we want to ultimately retrieve) and the embeddings of the images of dishes as `chromadb.embeddings` (since this is what we want the system to use to perform similarity search).

We store alongside each record in the vector index metadata, such as "cuisine type", "ingredients" and "cooking time", to enable constrained retrieval throught filtering, in addition to similarity search.

# Appendix
Repository with the Python code used for implementing a Multimodal (AI-powered) recommendation system.

![MCP_lifecycle](readme/MCP_lifecycle.jpeg)  
