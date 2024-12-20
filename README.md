# SciBot

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
