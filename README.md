# AgenticAI-with-MCP
Code repository for an Agentic AI, consisting of an LLM with access to tools, prompts and resources, through the Model Context Protocol (MCP).
<br>

Particularly, we implement a (multimodal - image-oriented) RAG workflow that recommends recipes to users, based on a given photo (image) of the dish that the user would like to replicate. This system returns the recipe that produces the dish that looks the most similar to the dish in the photo. The output recipe contains ingredients, cooking instructions, cooking time, among other information. We feed this information to an (open-source) LLM that synthetizes it, and provides a clear user-friendly step-by-step guide on how to cook something the most similar to what is shown in the photo.

This system tries to provide a recipe that results in the exact desired result. However, when it is not possible to provide the exact same recipe, the system will explain why the output recipe produces a dish that might result in what the user wants: for example, the recipe uses the same ingredients that are shown in the picture, or have the same preparation method.
<br>

Recipes consist of a list of `ingredients`, `instructions`, among other information, which includes an image of the resulting dish produced by following each recipe.

If we would like to construct an indexing system (which is) capable of retrieving the recipe that produces the dish that "looks" the most similar to a given photo, then our indexing system must store the `ingredients`, `instructions` and other useful information about recipes, but specifically use the images of the dishes to compare recipes.

This is why we use the recipe names as `chromadb.documents` (since this is what we want to ultimately retrieve) and the embeddings of the images of dishes as `chromadb.embeddings` (since this is what we want the system to use to perform similarity search).

Now, in order for our system to be able to provide an explanation of why the output recipe produces something similar to what is shown in the photo, we must provide the synthetizer LLM with a textual description of the output recipe, as well as a textual description of the input photo. The LLM must use these descriptions to compare both results in terms of ingredients, looks, and colors (in general, in terms of the elements that can be seen in the image, and which characterize it).

We store alongside each record in the vector index metadata, such as "cuisine type", "ingredients" and "cooking time", to enable constrained retrieval throught filtering, in addition to similarity search.

# Appendix
Repository with the Python code used for implementing a Multimodal (AI-powered) recommendation system.

![MCP_lifecycle](readme/MCP_lifecycle.jpeg)  
