# Gymnastics AI Assistant

A RAG-based AI assistant built on the **FIG Women's Artistic Gymnastics Code of Points 2025–2028 (Hungarian version)**.

Developed as a university project to explore whether Retrieval-Augmented Generation (RAG) can improve the accuracy of AI-generated answers in a specialized, rule-based domain.

### Features

- **RAG Assistant:** Answers gymnastics-related questions using retrieved rulebook content, with source references.
- **Baseline LLM:** Answers the same questions without document retrieval for comparison.
- **Chat interface:** Supports follow-up questions and conversation history.
- **Evaluation:** Manual assessment of answer correctness, relevance, retrieved chunks, groundedness and citation accuracy.
- **Evaluation Dashboard:** Visualizes performance metrics and compares both approaches.

### Tech Stack

Python · Streamlit · Hugging Face / Sentence Transformers · multilingual-e5-small · Pinecone · Groq API (GPT-OSS 120B)

### Live Application

[Open Gymnastics AI Assistant on Streamlit Cloud](https://gyamnasticsragassistant-ynyxgb8r5fapprhwapp2hyqw.streamlit.app/)

### Run Locally

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/aszenas/gyamnastics_rag_assistant.git
cd gyamnastics_rag_assistant
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=gymnastics-rag
```

3. Start the application:

```bash
streamlit run app/app.py
```

The Pinecone index must already contain the processed rulebook embeddings.

**Note:** Never commit your `.env` file or API keys to GitHub.

### Author

Szénás Anna Lilla
