from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_confidence(answer: str, documents: list, embeddings) -> float:
    """Estimate grounding confidence between answer and retrieved context."""
    if not documents:
        return 0.0

    doc_text = " ".join([d.page_content for d in documents])
    vecs = embeddings.embed_documents([answer, doc_text])
    return cosine_similarity([vecs[0]], [vecs[1]])[0][0]

def make_prompt():
    """Create a contextual RAG conversational prompt with dynamic questioning."""
    return PromptTemplate.from_template("""
    You are a precise technical retrieval-augmented conversational assistant.
    Follow these rules:
    1. Only answer from retrieved context.
    2. Prefer short, factual explanations — do not invent content.
    3. If unsure, clearly state that you cannot answer the question based on your knowledge base.
    4. Cite document sources when relevant.
    5. If there are any specific conditions mentioned in the context, add them in your answer as well.

    Conversation History:
    {chat_history}

    Retrieved Context:
    {context}

    User Question:
    {question}

    If the retrieved context is limited or your confidence is low,
    end with a thoughtful follow-up question that relates to the topic.

    Final Answer:
    """)

def build_confident_rag_chain(llm, retriever, memory, embeddings, threshold=0.7):
    """Builds a RAG chain that appends a follow-up question when confidence < threshold."""
    prompt = make_prompt()
    base_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        #return_source_documents=True,
        output_key="answer",
    )

    def invoke_with_confidence(inputs):
        result = base_chain.invoke(inputs)
        answer = result["answer"]
        docs = result.get("source_documents", [])

        confidence = compute_confidence(answer, docs, embeddings)

        if confidence < threshold:
            answer = f"{answer}\n\nWhat aspect of this topic would you like to explore next?"
        result["confidence"] = confidence
        result["answer"] = answer
        return result

    return invoke_with_confidence

# Change the embeddings model to match with the model.
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://host.docker.internal:11434")
llm = OllamaLLM(model="llama3.2:latest", base_url="http://host.docker.internal:11434")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

loader = UnstructuredURLLoader(urls=["https://www.colgate.edu/about/campus-services-and-resources/university-purchases"])
docs = loader.load()

vector_store.add_documents(documents=docs)
# results = vector_store.similarity_search("What are the ways to acquire goods or services?")

# system_rules = """
# You are a precise technical assistant specializing on the context.
# Follow these rules:
# 1. Only answer from retrieved context.
# 2. Prefer short, factual explanations — do not invent content.
# 3. If unsure, clearly state that you cannot answer the question based on your knowledge base.
# 4. Cite document sources when relevant.
# 5. If there are any specific conditions mentioned in the context, add them in your answer as well.
# """



retriever = vector_store.as_retriever(search_kwargs={"k": 5})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = build_confident_rag_chain(llm, retriever, memory, embeddings, threshold=0.75)


# ---- 7. Run a query ----
query = "What are all the ways to acquire goods or services?"
result = qa_chain({"question": query})

print("\n--- Answer ---")
print(result["answer"])

query = "What is the capital of Turkey?"
result = qa_chain.invoke({"question": query})

print("\n--- Answer ---")
print(result["answer"])