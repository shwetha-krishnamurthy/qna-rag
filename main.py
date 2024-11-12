# https://medium.com/@kv742000/creating-a-python-server-for-document-q-a-using-langchain-31a123b67935
from flask import Flask, request, jsonify

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.output_parsers import RegexParser
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

load_dotenv()

if 'document_metadata' not in st.session_state:
    st.session_state.document_metadata = {}

def render_document_manager():
    """Render the document management interface"""
    st.sidebar.header("📁 Document Manager")
    
    # Document upload
    uploaded_files = st.sidebar.file_uploader("Upload PDF Document", accept_multiple_files=True, type="pdf")
    for uploaded_file in uploaded_files:
        try:
            if not os.path.exists("docs"):
                os.makedirs("docs")
            
            file_path = os.path.join("docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add metadata
            description = st.sidebar.text_area(
                "Add document description (optional):",
                key=f"desc_{uploaded_file.name}"
            )
            
            st.session_state.document_metadata[uploaded_file.name] = {
                'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': description
            }
            
            st.sidebar.success(f"Uploaded {uploaded_file.name} successfully!")
            
            # Reset processor initialization flag to reprocess documents
            st.session_state.processor_initialized = False
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error uploading file: {str(e)}")

def render_document_list():
    """Render the list of uploaded documents"""
    st.sidebar.header("📚 Uploaded Documents")
    
    if not os.path.exists("docs"):
        st.sidebar.warning("No documents directory found.")
        return
    
    pdf_files = [f for f in os.listdir("docs") if f.endswith('.pdf')]
    
    if not pdf_files:
        st.sidebar.warning("No documents uploaded yet.")
        return
    
    for pdf_file in pdf_files:
        with st.sidebar.expander(f"📄 {pdf_file}"):
            metadata = st.session_state.document_metadata.get(pdf_file, {})
            st.write(f"📅 Uploaded: {metadata.get('upload_date', 'Unknown')}")
            st.write(f"📝 Description: {metadata.get('description', 'No description provided')}")
            
            if st.button(f"Delete {pdf_file}", key=f"del_{pdf_file}"):
                try:
                    os.remove(os.path.join("docs", pdf_file))
                    if pdf_file in st.session_state.document_metadata:
                        del st.session_state.document_metadata[pdf_file]
                    st.session_state.processor_initialized = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting file: {str(e)}")

@st.cache_resource
def set_up_embeddings():
    loader = DirectoryLoader(f'docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    chunk_size_value = 1000
    chunk_overlap=100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap,length_function=len)
    texts = text_splitter.split_documents(documents)
    docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
    docembeddings.save_local("llm_faiss_index")
    docembeddings = FAISS.load_local("llm_faiss_index",OpenAIEmbeddings(), allow_dangerous_deserialization=True)


    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    This should be in the following format:

    Question: [question here]
    Helpful Answer: [answer here]
    Score: [score between 0 and 100]

    Begin!

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Helpful Answer:"""
    output_parser = RegexParser(
        regex=r"(.*?)\nScore: (.*)",
        output_keys=["answer", "score"],
    )

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser
    )

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)
    return docembeddings, chain

def get_answer(query, docembeddings, chain):
    try:
        relevant_chunks = docembeddings.similarity_search_with_score(query,k=2)
        chunk_docs=[]

        for chunk in relevant_chunks:
            chunk_docs.append(chunk[0])

        results = chain.invoke({"input_documents": chunk_docs, "question": query})

        print("Results are:", results, type(results), results["input_documents"])

        text_reference=""
        for i in range(len(results["input_documents"])):
            text_reference+=results["input_documents"][i].page_content
        output={"Answer":results["output_text"],"Reference":text_reference}
        return output

    except ValueError as e:
        return {
            "Answer": f"{str(e)[25:]}",
            "Reference": "The LLM returned the answer that didn't fit the format hence we don't have any reference",
        }
    

# Streamlit UI
def main():
    st.title("📝 Q&A RAG System")
    st.write("Upload any documents and ask questions to get answers")
    st.write("Please reload the page after uploading documents to see the changes")

    # Render document management interface
    render_document_manager()
    render_document_list()

    # Input section
    query = st.text_input("Query")
    
    openai_api_key = st.text_input("OpenAI API Key")

    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if st.button("Get Answer"):
        if query:
            try:
                with st.spinner("Writing answer..."):
                    # Get both answer and references
                    docembeddings, chain = set_up_embeddings()
                    formatted_transcript = get_answer(query, docembeddings, chain)
                    
                    # Display Answer
                    st.subheader("Answer")
                    st.text_area("", formatted_transcript["Answer"], height=400) 

                    # Display References
                    st.subheader("References")
                    st.text_area("", formatted_transcript["Reference"], height=400)       
                    
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Please enter a YouTube URL")

if __name__ == "__main__":
    main()
