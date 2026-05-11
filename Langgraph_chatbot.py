from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import TypedDict, Annotated,Dict,Optional, Any

import requests
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

# Environment Setup

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STOCK_API_KEY = os.getenv("STOCK_API_KEY")

# Models and Embeddings

llm = ChatGroq(model="llama-3.1-8b-instant")
embeddings= HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

# In-memory Storage

_THREAD_RETRIVERS: Dict[str,Any] = {}
_THREAD_METADATA: Dict[str,dict] = {}

# Helper Fucntion

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a threda if available"""
    if thread_id and thread_id in _THREAD_RETRIVERS:
        return _THREAD_RETRIVERS[thread_id]
    return None

    

def ingest_pdf(file_bytes:bytes, thread_id:str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    # Prevent processing if the thread already has a retriever
    if thread_id in _THREAD_RETRIVERS:
        return _THREAD_METADATA.get(str(thread_id), {})
    
    # Handle file-like objects
    if hasattr(file_bytes, "read"):
        file_bytes = file_bytes.read()

    if not file_bytes:
        raise ValueError("No bytes received for ingestion")

    # Save Uploaded bytes to a temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        # load pdf
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("PDF is empty or contains no extractable text")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
        
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text chunks extracted from PDF. The PDF may be scanned or image-based.")
        
        # Create Vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k":4}
        )

        # Store retriever and metadata
        _THREAD_RETRIVERS[str(thread_id)]= retriever

        _THREAD_METADATA[str(thread_id)]= {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunk": len(chunks),
        }
        return{
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunk": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass



#Tools

search_tool = TavilySearchResults(max_results=3)


@tool
def calculator(first_num: float, second_num: float, operation:str) -> dict:
    """
    Perform a basic arthimetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return{"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return{'error': f"Unsupported operation '{operation}'"}
        return{"first_num": first_num, "second_num": second_num, "operation": operation, "result":result}
    except Exception as e:
        return{'error': str(e)}
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    Using Alpha vantage with API key.
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": STOCK_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            quote = data["Global Quote"]
            return {
                "symbol": symbol,
                "price": quote.get("05. price", "N/A"),
                "change": quote.get("09. change", "N/A"),
                "change_percent": quote.get("10. change percent", "N/A")
            }
        else:
            return {"error": f"Could not fetch price for symbol: {symbol}"}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def rag_tool(query:str, thread_id: Optional[str]= None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """ 
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return{
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return{
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id),{}).get("filename"),
    }

# Tool configuration
 
tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)
    
# LangGraph State

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages ]
    thread_id: str

# Graph Nodes
def chat_node(state: ChatState):
    messages = state['messages']

    response = llm_with_tools.invoke(messages)

    return{'messages': [response]}

# SQlite Setup
conn = sqlite3.connect(database='Nexus.db', check_same_thread=False)
conn.execute("""
             Create Table If not Exists thread_names(
             thread_id TEXT PRIMARY KEY,
             name TEXT)
""")
conn.commit()

checkpointer = SqliteSaver(conn=conn)

# Graph Definition

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge('chat_node', END)
chatbot = graph.compile(checkpointer=checkpointer)

# Thread Mangement Functions
def retrieve_all_threads():
    rows = conn.execute("SELECT thread_id, name FROM thread_names").fetchall()
    return {r[0]: r[1] for r in rows}

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIVERS

def thread_document_metadata(thread_id:str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

def save_thread_name(thread_id, name):
    conn.execute("INSERT or REPLACE INTO thread_names VALUES(?,?)",(thread_id, name))
    conn.commit()   
