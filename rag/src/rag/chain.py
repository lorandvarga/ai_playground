from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from config import Config


def format_docs(docs: List) -> str:
    """Format documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChainWrapper:
    """Wrapper to hold both the chain and retriever"""
    def __init__(self, chain, retriever):
        self.chain = chain
        self.retriever = retriever

    def invoke(self, question: str) -> str:
        """Invoke the chain with a question"""
        return self.chain.invoke(question)


def create_rag_chain(vectorstore: FAISS) -> RAGChainWrapper:
    """
    Create a Retrieval QA chain for question answering using LCEL.

    Args:
        vectorstore: FAISS vector store for document retrieval

    Returns:
        RAGChainWrapper instance containing chain and retriever

    Raises:
        Exception: If there's an error creating the chain
    """
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Create LLM
        llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            api_key=Config.OPENAI_API_KEY,
            temperature=0
        )

        # Create custom prompt template
        template = """Use the following context to answer the question.
If you cannot find the answer in the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Create RAG chain using LCEL
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Return wrapped chain with retriever
        return RAGChainWrapper(rag_chain, retriever)

    except Exception as e:
        raise Exception(f"Error creating RAG chain: {str(e)}")


def query(qa_chain: RAGChainWrapper, question: str) -> Dict[str, Any]:
    """
    Query the RAG chain with a question.

    Args:
        qa_chain: RAGChainWrapper instance
        question: User's question

    Returns:
        Dictionary containing 'result' (answer) and 'source_documents'

    Raises:
        Exception: If there's an error processing the query
    """
    try:
        # Get the answer
        answer = qa_chain.invoke(question)

        # Get source documents separately
        source_documents = qa_chain.retriever.invoke(question)

        # Return in expected format
        return {
            "result": answer,
            "source_documents": source_documents
        }

    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")
