# 🤖 NexusChat

A multi-tool AI chatbot built with **LangGraph**, **LangChain**, and **Streamlit** — featuring persistent memory, PDF-based RAG, web search, stock prices, and a calculator, all powered by Groq's lightning-fast LLaMA 3.1.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Conversational Memory** | Per-thread SQLite checkpointing via LangGraph — conversations persist across sessions |
| 📄 **PDF RAG** | Upload a PDF and ask questions about it; FAISS vector store with HuggingFace embeddings |
| 🌐 **Web Search** | Real-time search powered by Tavily |
| 📈 **Stock Prices** | Live stock quotes via Alpha Vantage API |
| 🧮 **Calculator** | Basic arithmetic (add, subtract, multiply, divide) as an LLM-callable tool |
| 💬 **Multi-thread Chats** | Create, switch, and name multiple independent conversations from the sidebar |

---

## 🏗️ Architecture

```
User Input (Streamlit UI)
        │
        ▼
  LangGraph Agent
  ┌─────────────┐
  │  chat_node  │◄──── LLaMA 3.1 8B (via Groq)
  └──────┬──────┘
         │  tools_condition
         ▼
  ┌─────────────┐
  │  tool_node  │──── [Web Search | Stock Price | Calculator | RAG]
  └──────┬──────┘
         │
         ▼
  SQLite Checkpointer (Nexus.db)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for **Groq**, **Tavily**, and **Alpha Vantage**

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nexuschat.git
cd nexuschat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
STOCK_API_KEY=your_alpha_vantage_api_key
TAVILY_API_KEY=your_tavily_api_key
```

> Get your keys at:
> - Groq → https://console.groq.com
> - Tavily → https://tavily.com
> - Alpha Vantage → https://www.alphavantage.co/support/#api-key

### 4. Run the app

```bash
streamlit run main_app.py
```

---

## 📁 Project Structure

```
nexuschat/
├── Langgraph_chatbot.py   # LangGraph agent, tools, PDF ingestion, graph definition
├── main_app.py            # Streamlit UI — chat interface, sidebar, session state
├── Nexus.db               # Auto-created SQLite database for thread persistence
├── .env                   # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-groq
langchain-huggingface
langgraph
faiss-cpu
pypdf
python-dotenv
requests
sentence-transformers
tavily-python
```

> Generate the exact file with `pip freeze > requirements.txt` after installing.

---

## 🛠️ How It Works

### Agent Loop
NexusChat uses a **ReAct-style LangGraph agent**. The LLM decides whether to answer directly or call a tool. After a tool executes, the result flows back to the LLM to form the final response.

### PDF RAG
When you upload a PDF:
1. It is split into chunks using `RecursiveCharacterTextSplitter`
2. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`
3. A FAISS vector store is built and tied to the current thread
4. The `rag_tool` performs similarity search and returns relevant context to the LLM

### Thread Persistence
Each conversation has a `thread_id`. LangGraph's `SqliteSaver` checkpoints the full message history so conversations survive app restarts.

---
