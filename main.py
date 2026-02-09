import os
import re
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
import faiss
import textwrap

# =============================
# Load environment variables
# =============================
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")

if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN not found in .env")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

# =============================
# Configure Settings globally (must be done early)
# =============================
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=GOOGLE_API_KEY,
)

Settings.llm = GoogleGenAI(
    model="models/gemini-flash-latest",
    api_key=GOOGLE_API_KEY,
)

# =============================
# Helpers
# =============================
def parse_github_url(url):
    # Remove .git suffix and any trailing characters (#, etc)
    url = url.replace(".git", "").rstrip("#/")
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    if match:
        owner, repo = match.groups()
        return owner, repo.rstrip("#/")
    return (None, None)

# =============================
# GitHub client
# =============================
github_client = GithubClient(GITHUB_TOKEN)

# =============================
# Load repository
# =============================
github_url = input("Enter GitHub repository URL: ")
owner, repo = parse_github_url(github_url)

if not owner or not repo:
    raise ValueError("Invalid GitHub URL")

print(f"Loading repository: {owner}/{repo}")

loader = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    filter_file_extensions=(
        [".py", ".md"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    verbose=True,
    concurrent_requests=1,
)

# Auto-detect default branch
import requests
branch_response = requests.get(
    f"https://api.github.com/repos/{owner}/{repo}",
    headers={"Authorization": f"token {GITHUB_TOKEN}"}
)
default_branch = branch_response.json().get("default_branch", "main")
print(f"Using branch: {default_branch}")

docs = loader.load_data(branch=default_branch)

print(f"Loaded {len(docs)} documents")

if len(docs) == 0:
    print("No documents found. Try a repository with Python files.")
    exit(0)

# =============================
# FAISS Vector Store - Load or Create
# =============================
if os.path.exists(FAISS_INDEX_PATH):
    print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load existing index
    index = VectorStoreIndex.from_documents(
        [],  # Empty - loading from existing index
        storage_context=storage_context,
    )
else:
    # Get embedding dimension dynamically
    test_embedding = Settings.embed_model.get_text_embedding("test")
    dimension = len(test_embedding)
    print(f"Embedding dimension: {dimension}")

    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
    )

    # Save FAISS index to disk
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH) if os.path.dirname(FAISS_INDEX_PATH) else ".", exist_ok=True)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print(f"FAISS index created and saved to {FAISS_INDEX_PATH}")

# =============================
# Query Engine
# =============================
query_engine = index.as_query_engine()

# Test question
intro_question = "What is the AI model used for the Agent?"
print("\nTest question:", intro_question)
print("=" * 60)
answer = query_engine.query(intro_question)
print(textwrap.fill(str(answer), 100))

# Interactive loop
while True:
    user_question = input("\nAsk a question (or 'exit'): ")
    if user_question.lower() == "exit":
        print("Exiting.")
        break

    print("=" * 60)
    answer = query_engine.query(user_question)
    print(textwrap.fill(str(answer), 100))
