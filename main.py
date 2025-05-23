
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

def get_embeddings():
    """Initialize and return the embedding model"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

def get_vectorstore(embeddings, collection_name="hack"):
    """Initialize and return the vector store"""
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        api_key=QDRANT_API_KEY,
        url=QDRANT_URL,
        retrieval_mode=RetrievalMode.DENSE,
    )

def get_llm():
    """Initialize and return the LLM"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        max_tokens=512,
        model_name="llama-3.3-70b-versatile"
    )

def get_retrieval_chain():
    """Create and return the RAG chain using runnables"""
    # Initialize components
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    
    # Define the prompt template
    prompt = PromptTemplate.from_template(
        """You are an assistant who only gives answers based on the context provided.give me little bit long answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
    )
    
    # Create the RAG chain using runnables
    rag_chain = (
        # Define inputs/outputs for the chain
        {
            "context": retriever, 
            "question": RunnablePassthrough()
        }
        # Pipe the inputs through the prompt template
        | prompt
        # Send the formatted prompt to the LLM
        | llm
        # Extract the string output from the LLM response
        | StrOutputParser()
    )
    
    return rag_chain

def query_answer(question: str) -> str:
    """Process a question through the RAG chain and return the answer"""
    chain = get_retrieval_chain()
    response = chain.invoke(question)
    return response