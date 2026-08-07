import streamlit as st
import os
from paperqa import Settings, ask
from paperqa.settings import AgentSettings
from pathlib import Path

# Set OpenAI API Key (Ollama API) - not used, just populated
os.environ['OPENAI_API_KEY'] = "ollama"

# Directory to store uploaded papers
papers_directory = "papers"
os.makedirs(papers_directory, exist_ok=True)

#constants to be used throughout the code
MODEL = "ollama/llama3.2" 
EMBEDDING = "ollama/nomic-embed-text"
OLLAMA_SERVER = "http://localhost:11434"

# Configure Ollama in local settings for paper-qa
local_llm_config = {
    "model_list": [
        {
            "model_name": MODEL,
            "litellm_params": {
                "model": MODEL,
                "api_base": OLLAMA_SERVER,  
            }
        }
    ]
}


st.set_page_config(
        page_title="SciBot",
        page_icon=":page_facing_up:",
        layout="wide",
    )

# Streamlit page title
st.title("Chat with Your Papers :books:")

# Sidebar for file upload
uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
# Function to save uploaded PDFs to the "papers" directory

def save_uploaded_files(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(papers_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    return file_paths

# Display the files already in the "papers" directory (if any)
existing_files = os.listdir(papers_directory)

if existing_files:
    st.sidebar.write("Existing files in 'papers' directory:")
    selected_files = st.sidebar.multiselect(
        "Select files to preview or delete", existing_files
    )
    
    if selected_files:
        st.sidebar.write("Selected files:")
        for selected_file in selected_files:
            st.sidebar.text(selected_file)
        
        # File preview (show the content of the first selected file)
        if len(selected_files) == 1:
            selected_file_path = os.path.join(papers_directory, selected_files[0])
            with open(selected_file_path, "rb") as file:
                st.sidebar.download_button(
                    label="Download the file",
                    data=file,
                    file_name=selected_files[0],
                    mime="application/pdf"
                )

        # Delete the selected files
        if st.sidebar.button("Delete Selected Files"):
            for file in selected_files:
                os.remove(os.path.join(papers_directory, file))
                st.sidebar.write(f"Deleted {file}")

else:
    st.sidebar.write("No existing files found in 'papers' directory.")

# Save the uploaded files when needed
if uploaded_files:
    saved_files = save_uploaded_files(uploaded_files)
    st.sidebar.write("Uploaded files saved:")
    for saved_file in saved_files:
        st.sidebar.text(saved_file)

        
# Function to query using paper-qa and Ollama
def query_paperqa(question):
    # Ask the question using paper-qa
    answer = ask(
        question,
        settings=Settings(
            llm=MODEL,
            llm_config=local_llm_config,
            summary_llm=MODEL,
            summary_llm_config=local_llm_config,
            embedding=EMBEDDING,
            agent=AgentSettings(
                agent_llm=MODEL,
                agent_llm_config=local_llm_config
            ),
            paper_directory=papers_directory  # Use the papers directory where PDFs are stored
        ),
    )
    return answer

# Display the PDF upload section
if uploaded_files:
    if st.sidebar.button("Process PDFs"):
        # Save uploaded files to the "papers" directory
        file_paths = save_uploaded_files(uploaded_files)
        st.sidebar.success("Documents uploaded successfully!")

# Input field for user question
user_input = st.text_input("Ask a question about the documents:")

if st.button("Submit Question"):
    if user_input:
        # Query paper-qa and Ollama with the user's question
        response = query_paperqa(user_input)
        formatted_response = response.model_dump()['session']['formatted_answer']

        # Display the response from paper-qa and Ollama
        st.markdown(f"### {formatted_response}")
