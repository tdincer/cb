"""
Ingest Web Pages into Chroma using Nodes

- Uses BeautifulSoupWebReader to load HTML content.
- Splits content into nodes using SentenceSplitter.
- Embeds nodes with Ollama embeddings.
- Stores everything in a local Chroma vector store.
"""

import pickle
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    HTMLNodeParser,
    HierarchicalNodeParser,
    TokenTextSplitter,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from chromadb import PersistentClient
from llama_index.llms.ollama import Ollama
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.schema import MetadataMode


class Models:
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    request_timeout: int = 120

    def nomic_embedding(cls):
        return OllamaEmbedding(
            model_name="nomic-embed-text:latest",
            base_url=cls.OLLAMA_BASE_URL,
            request_timeout=cls.request_timeout,
            # max_length=2048,
        )

    def gemma_embedding(cls):
        return OllamaEmbedding(
            model_name="embeddinggemma:latest",
            base_url=cls.OLLAMA_BASE_URL,
            request_timeout=cls.request_timeout,
            # max_length=2048,
        )

    def deepseek_r1(cls):
        return Ollama(
            model="deepseek-r1:1.5b",
            base_url=cls.OLLAMA_BASE_URL,
            request_timeout=cls.request_timeout,
        )

    def llama32(cls):
        return Ollama(
            model="llama3.2:latest",
            base_url=cls.OLLAMA_BASE_URL,
            request_timeout=cls.request_timeout,
        )


SOCIAL_MEDIA_KEYWORDS = [
    "twitter",
    "linkedin",
    "github",
    "facebook",
    "instagram",
    "youtube",
    "follow us",
    "flickr",
]


def clean_text(text: str) -> str:
    """
    Clean web page text:
    1. Remove lines that look like social media links (case-insensitive).
    2. Collapse 3+ consecutive newlines → 2 newlines (one blank line between paragraphs).
    3. Collapse multiple spaces/tabs → single space.
    4. Strip leading/trailing whitespace.
    """
    # Split text into lines and remove social media lines
    lines = text.split("\n")
    filtered_lines = [
        line
        for line in lines
        if not any(keyword.lower() in line.lower() for keyword in SOCIAL_MEDIA_KEYWORDS)
    ]
    text = "\n".join(filtered_lines)

    # Collapse 3+ newlines → 2 newlines (paragraph preservation)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs → single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()


def universal_extractor(soup: BeautifulSoup, **kwargs):
    """
    Extracts and cleans text from a BeautifulSoup object.
    """
    # Extract text using the default method
    text = soup.get_text(separator="\n")

    # Clean the extracted text
    cleaned_text = clean_text(text)

    # Return the cleaned text and an empty metadata dictionary
    return cleaned_text, {}


class WebDocs:
    def __init__(self, urls=list):
        self.urls = urls
        self.documents = None
        self.custom_extractors = {
            urlparse(url).netloc: universal_extractor for url in urls
        }

    def save_pickle(self, filename: str = "documents.pkl"):
        """Save the documents list as a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self.documents, f)

    def load_pickle(self, filename: str = "documents.pkl"):
        """Try to load documents from a pickle file into self.documents."""
        import pickle

        try:
            with open(filename, "rb") as f:
                self.documents = pickle.load(f)
                print(f"Loaded documents from {filename}.")
        except FileNotFoundError:
            self.documents = None
            print(f"Pickle file not found: {filename}.")
        except (pickle.UnpicklingError, EOFError) as e:
            self.documents = None
            print(f"Error loading pickle ({filename}): {e}")

    def save_txt(self, filename: str = "documents.txt"):
        """Save a plain text version of all documents for human inspection."""
        if not self.documents:
            return
        with open(filename, "w", encoding="utf-8") as f:
            for i, doc in enumerate(self.documents, start=1):
                f.write(f"\n\n===== Document {i} =====\n")
                f.write(doc.text or "")

    def load(self, pickle_file: str = "documents.pkl"):
        """Load from cache if exists, else scrape from the web."""
        try:
            self.documents = self.load_pickle(pickle_file)
            if self.documents:
                print("Loaded cached documents.")
                return  # no return value — data is stored in self.documents
            raise FileNotFoundError
        except FileNotFoundError:
            print("Cache not found — fetching fresh data.")

        reader = BeautifulSoupWebReader(website_extractor=self.custom_extractors)
        self.documents = reader.load_data(self.urls)

        self.save_pickle(pickle_file)
        self.save_txt()
        print("Documents fetched and cached.")


urls: list = [
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control",
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control/accounts-payable",
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control/grants-and",
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control/faqs",
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control/payroll",
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control/cashier",
    "https://www.colgate.edu/about/offices-centers-institutes/finance-and-administration/accounting-and-control/student-accounts",
    "https://www.colgate.edu/about/campus-services-and-resources/travel-expenses-and-reimbursement",
    "https://www.colgate.edu/about/campus-services-and-resources/gate-card",
    "https://www.colgate.edu/about/campus-services-and-resources/travel-policy",
    "https://www.colgate.edu/about/campus-services-and-resources/employee-travel",
]

model = Models()
web_docs = WebDocs(urls)
web_docs.load()

documents = web_docs.documents

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(documents)


# 4️⃣ Setup Chroma persistence
persist_dir: str = "./chroma_bs_nodes_db"
collection_name: str = "bs_nodes"

client = PersistentClient(path=persist_dir)
chroma_store = ChromaVectorStore(chroma_collection=collection_name, client=client)
storage_context = StorageContext.from_defaults(vector_store=chroma_store)

# 5️⃣ Build and persist index
index = VectorStoreIndex(
    nodes,
    embed_model=model.gemma_embedding(),
    show_progress=True,
)
index.storage_context.persist(persist_dir)

query_engine = index.as_query_engine(
    llm=model.llama32(), response_mode="tree_summarize", similarity_top_k=10
)

questions = [
    "prohibition in air travel?",
    "use of personal car for travel?",
    "Are there any constraints on using personal car for travel?",
    "What is the rental car policy?",
    "What is the mileage rate for 2025?",
    "What is it required to use a rental car?",
    "Who are the CBT's designated full service advisors?",
    "What is Allison Holms's email from CBT?",
    "What is Allison Holms's phone number?",
    "Who are the representatives of CBT?",
    "What is Kari Fichter's phone number?",
    "What is the CBT's contact information?",
    "mileage rate effective 2025, personal car, travel policy",
]
r = query_engine.query(questions[0])
print(r.response)
#############

persist_dir: str = "./chroma_bs_nodes_db"
collection_name: str = "bs_nodes"
embedding_model: str = "nomic-embed-text:latest"
llm_model = (
    "deepseek-r1:1.5b"  # "tinyllama:latest"  # "deepseek-r1:1.5b"  # "llama3.2:latest"
)
ollama_base_url: str = "http://host.docker.internal:11434"
urls = ["https://eugeneyan.com/writing/llm-patterns/"]

# LLM Model
llm = Ollama(model=llm_model, base_url=ollama_base_url)

# Embedding Model
embed_model = OllamaEmbedding(
    model_name=embedding_model, base_url=ollama_base_url, max_length=2048
)

# Documents
reader = BeautifulSoupWebReader()
documents = reader.load_data(urls)
print(f"Loaded {len(documents)} documents")

# Node parser
node_parser = TokenTextSplitter(separator=" ", chunk_size=256, chunk_overlap=128)

# Extractor
extractors_1 = [
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=2
    ),
]

# Original Nodes
orig_nodes = node_parser.get_nodes_from_documents(documents)
print(orig_nodes[0].get_content(metadata_mode="all"))

pipeline = IngestionPipeline(transformations=[node_parser, *extractors_1])

nodes_1 = pipeline.run(nodes=orig_nodes, in_place=False, show_progress=True)

# Ingest nodes
index = VectorStoreIndex(nodes_1, embed_model=embed_model)


# RAG
retriever = index.as_retriever(similarity_top_k=5)
from llama_index.core.response_synthesizers import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    llm=llm,  # the same model you used for extractors, or a larger one
    response_mode="compact",  # "tree_summarize" for long contexts
)
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)
question = "What problems does LlamaIndex solve?"
response = query_engine.query(question)
print(response)

# Inspect RAG
retrieved = retriever.retrieve(question)
for node in retrieved:
    print(f"\n=== Node ===\n{node.text[:400]}...\n")

# Query rewrite
from llama_index.core.chat_engine import CondenseQuestionChatEngine

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    llm=llm,  # your existing LLM (OllamaLLM, OpenAI, etc.)
)
chat_engine.reset()  # clear memory before new session

response_1 = chat_engine.chat("What is LlamaIndex?")
print(response_1)

response_2 = chat_engine.chat("How does it handle document ingestion?")
print(response_2)

response_3 = chat_engine.chat("and its performance?")
print(response_3)


def ingest_bs_nodes_to_chroma(
    urls: list[str],
    persist_dir: str = "./chroma_bs_nodes_db",
    collection_name: str = "bs_nodes",
    embedding_model: str = "nomic-embed-text:latest",
    ollama_base_url: str = "http://host.docker.internal:11434",
):
    """
    Ingest web pages using BeautifulSoupWebReader, split into sentence nodes,
    embed with Ollama, and store in Chroma.
    """

    # 1️⃣ Load web pages
    reader = BeautifulSoupWebReader()
    documents = reader.load_data(urls)
    print(f"Loaded {len(documents)} documents")
    return documents
    # 2️⃣ Create nodes using pipeline
    pipeline = IngestionPipeline(
        transformations=[
            HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        ]
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    # print(f"Created {len(nodes)} nodes")

    # # 3️⃣ Setup Ollama embeddings
    # embed_model = OllamaEmbedding(model_name=embedding_model, base_url=ollama_base_url)

    # # 4️⃣ Setup Chroma persistence
    # client = PersistentClient(path=persist_dir)
    # chroma_store = ChromaVectorStore(chroma_collection=collection_name, client=client)
    # storage_context = StorageContext.from_defaults(vector_store=chroma_store)

    # # 5️⃣ Build and persist index
    # index = VectorStoreIndex.from_documents(
    #     nodes, storage_context=storage_context, embed_model=embed_model
    # )
    # index.storage_context.persist(persist_dir)

    # print(f"✅ Ingested {len(nodes)} nodes into Chroma collection '{collection_name}'")
    # return index


result = ingest_bs_nodes_to_chroma(
    urls=[
        "https://docs.llamaindex.ai/en/stable/",
        "https://llamaindex.ai/",
    ],
    persist_dir="./chroma_bs_nodes_db",
    collection_name="bs_nodes",
)

# if __name__ == "__main__":
#     ingest_bs_nodes_to_chroma(
#         urls=[
#             "https://docs.llamaindex.ai/en/stable/",
#             "https://llamaindex.ai/",
#         ],
#         persist_dir="./chroma_bs_nodes_db",
#         collection_name="bs_nodes",
#     )

# OLLAMA_KEEP_ALIVE=60 OLLAMA_LOAD_TIMEOUT=10 ollama start nomic-embed-text:latest
