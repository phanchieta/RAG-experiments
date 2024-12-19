# SciBot

1) Clone repo
2) Install Ollama
3) `ollama pull llama3.2`
4) `ollama pull nomic-embed-text`
5) `ollama serve`
6) `conda env create -f environment.yml`
7) If needed, change:
   
`def strip_citations(text):
    if text is None:
        return ""  # or return an empty string or some other default behavior
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    return re.sub(citation_regex, "", text, flags=re.MULTILINE)`
9) `streamlit run web_test.py`
