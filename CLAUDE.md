# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval Augmented Generation) application that indexes GitHub repositories and allows natural language queries against the codebase. It uses:
- **LlamaIndex** for document loading, indexing, and retrieval
- **Google Gemini** for embeddings (via `GoogleEmbedding`)
- **FAISS** for local vector storage

## Environment Setup

1. Create and activate the virtual environment (already created as `vrag`):
   ```bash
   source vrag/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install llama-index llama-index-readers-github llama-index-embeddings-google llama-index-vector-stores-faiss faiss-cpu nest-asyncio python-dotenv
   ```

3. Configure environment variables in [`.env`](.env):
   - `GITHUB_TOKEN` - GitHub personal access token for repository access
   - `GOOGLE_API_KEY` - Google AI API key for Gemini embeddings
   - `FAISS_INDEX_PATH` - Local path for FAISS vector store (default: `./vector_store/RAG1.faiss`)

## Running the Application

```bash
python main.py
```

The script will:
1. Prompt for a GitHub repository URL
2. Download and index repository files (`.py`, `.js`, `.ts`, `.md`)
3. Store embeddings in FAISS locally
4. Run an interactive query loop

## Architecture

**main.py** flow:
1. Loads environment variables via `python-dotenv`
2. Uses `GithubRepositoryReader` (LlamaIndex) to fetch repository contents
3. Creates embeddings with `GoogleEmbedding` (768-dimensional)
4. Stores vectors in `FaissVectorStore` for fast similarity search
5. Builds `VectorStoreIndex` and exposes as a query engine

The FAISS index is persisted locally at `FAISS_INDEX_PATH`, so subsequent runs can reuse the cached embeddings (this would require modifying the code to check for existing index).
