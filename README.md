# SciBot
LLM Research Assistant for Academic Papers

This project implements a Retrieval Augmented Generation (RAG) system
for exploring academic papers in psychology.

The system indexes research papers using embeddings and allows users
to ask questions about the literature using an LLM.

Motivation:
Researchers often spend a large amount of time searching through papers
to understand existing work. This tool was built to accelerate early-stage
literature exploration.

Main components:
- Paper ingestion and chunking
- Embedding generation
- Vector search retrieval
- LLM-based answer generation

# Create
1) Clone repo
2) Install Ollama
3) `ollama pull llama3.2`
4) `ollama pull nomic-embed-text`
5) `ollama serve`
6) `conda env create -f environment.yml`
7) If needed, change in file ~\anaconda3\envs\scibot\Lib\site-packages\paperqa\utils.py:
   
`def strip_citations(text):`

    if text is None:
    
        return "" 
        
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    
    return re.sub(citation_regex, "", text, flags=re.MULTILINE)`
    
    
8) `streamlit run web_test.py`

# Run (local - Rodrigo)
1) `conda activate scibot`
2) `cd repositories/SciBot`
3) `streamlit run web_test.py`
