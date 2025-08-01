# rag_backend.py

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Step 1: Load and process uploaded file
def process_and_store(file_path):
    loader = UnstructuredLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small",
        model_kwargs={"device": "cpu"}
    )
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local("faiss_index")
    return vectordb

# Step 2: Load FAISS vector store
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# Step 3: Build QA Chain
def get_qa_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="tinyllama", base_url="http://localhost:11434")

    prompt_template = PromptTemplate.from_template("""
You are a helpful AI assistant. Use ONLY the following context to answer the user's question.
If the answer is not in the context, say "I don't know."

Context:
{summaries}

Question: {question}

Answer:
""")

    qa_chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt_template
    )

    def rag_chain(question):
        docs = retriever.get_relevant_documents(question)
        response = qa_chain.run(input_documents=docs, question=question)
        return {"result": response, "source_documents": docs}

    return rag_chain