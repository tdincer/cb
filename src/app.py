from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Change the embeddings model to match with the model.
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://host.docker.internal:11434")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

loader = UnstructuredURLLoader(urls=["https://www.colgate.edu/about/campus-services-and-resources/university-purchases"])
docs = loader.load()

vector_store.add_documents(documents=docs)
results = vector_store.similarity_search("What are the ways to acquire goods or services?")

system_rules = """
You are a precise technical assistant specializing on the context.
Follow these rules:
1. Only answer from retrieved context.
2. Prefer short, factual explanations — do not invent content.
3. If unsure, clearly state that you cannot answer the question based on your knowledge base.
4. Cite document sources when relevant.
5. If there are any specific conditions mentioned in the context, add them in your answer as well.
"""

llm = OllamaLLM(model="llama3.2:latest", base_url="http://host.docker.internal:11434", system=system_rules)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
#    return_source_documents=True,
    output_key="answer"
)


# ---- 7. Run a query ----
query = "What are all the ways to acquire goods or services?"
result = qa_chain.invoke({"question": query})

print("\n--- Answer ---")
print(result["answer"])

query = "What is the capital of Turkey?"
result = qa_chain.invoke({"question": query})

print("\n--- Answer ---")
print(result["answer"])