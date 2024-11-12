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


# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain
# from langchain.output_parsers import RegexParser
# from langchain.docstore.document import Document
# from dotenv import load_dotenv
# import os
# import hashlib
# from datetime import datetime
# import json

# # Load environment variables
# load_dotenv()

# # Initialize session state
# if 'processor_initialized' not in st.session_state:
#     st.session_state.processor_initialized = False
# if 'docembeddings' not in st.session_state:
#     st.session_state.docembeddings = None
# if 'document_metadata' not in st.session_state:
#     st.session_state.document_metadata = {}

# @st.cache_resource
# def get_embeddings():
#     """Cache the embeddings model"""
#     return OpenAIEmbeddings()

# @st.cache_resource
# def get_llm():
#     """Cache the language model"""
#     return OpenAI(temperature=0)

# def save_document_metadata():
#     """Save document metadata to a JSON file"""
#     with open('docs/metadata.json', 'w') as f:
#         json.dump(st.session_state.document_metadata, f)

# def load_document_metadata():
#     """Load document metadata from JSON file"""
#     try:
#         with open('docs/metadata.json', 'r') as f:
#             st.session_state.document_metadata = json.load(f)
#     except FileNotFoundError:
#         st.session_state.document_metadata = {}

# @st.cache_resource
# def load_single_document(file_path):
#     """Cache loading of a single document"""
#     loader = PyPDFLoader(file_path)
#     return loader.load()

# def get_docs_hash():
#     """Generate a hash of all documents in the docs directory"""
#     hash_str = ""
#     for file in os.listdir('docs'):
#         if file.endswith('.pdf'):
#             file_path = os.path.join('docs', file)
#             with open(file_path, 'rb') as f:
#                 hash_str += hashlib.md5(f.read()).hexdigest()
#     return hash_str

# @st.cache_resource
# def split_documents(_documents, chunk_size_value, chunk_overlap):
#     """Cache document splitting"""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size_value,
#         chunk_overlap=chunk_overlap,
#         length_function=len
#     )
#     return text_splitter.split_documents(_documents)

# @st.cache_resource(hash_funcs={str: lambda x: get_docs_hash()})
# def create_embeddings_index(_texts):
#     """Cache the FAISS index creation"""
#     embeddings = get_embeddings()
#     return FAISS.from_documents(_texts, embeddings)

# def load_documents():
#     """Load all documents with metadata"""
#     documents = []
#     for filename in os.listdir('docs'):
#         if filename.endswith('.pdf'):
#             file_path = os.path.join('docs', filename)
#             docs = load_single_document(file_path)
            
#             # Add metadata to each document
#             for doc in docs:
#                 doc.metadata.update({
#                     'source': filename,
#                     'upload_date': st.session_state.document_metadata.get(filename, {}).get('upload_date', ''),
#                     'description': st.session_state.document_metadata.get(filename, {}).get('description', '')
#                 })
#             documents.extend(docs)
#     return documents

# def initialize_processor():
#     """Initialize the document processor and embeddings"""
#     try:
#         # Load documents with metadata
#         documents = load_documents()
        
#         if not documents:
#             st.warning("No documents found in the docs directory.")
#             return False
        
#         # Split documents with caching
#         chunk_size_value = 1000
#         chunk_overlap = 100
#         texts = split_documents(documents, chunk_size_value, chunk_overlap)
        
#         # Create embeddings with caching
#         st.session_state.docembeddings = create_embeddings_index(texts)
        
#         st.session_state.processor_initialized = True
#         return True
#     except Exception as e:
#         st.error(f"Error initializing processor: {str(e)}")
#         return False

# @st.cache_resource
# def setup_qa_chain():
#     """Cache the QA chain setup"""
#     prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

#     This should be in the following format:

#     Question: [question here]
#     Helpful Answer: [answer here]
#     Source Documents: [list of source documents]
#     Score: [score between 0 and 100]

#     Begin!

#     Context:
#     ---------
#     {context}
#     ---------
#     Question: {question}
#     Helpful Answer:"""
    
#     output_parser = RegexParser(
#         regex=r"(.*?)\nScore: (.*)",
#         output_keys=["answer", "score"],
#     )
    
#     PROMPT = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"],
#         output_parser=output_parser
#     )
    
#     return load_qa_chain(
#         get_llm(),
#         chain_type="map_rerank",
#         return_intermediate_steps=True,
#         prompt=PROMPT
#     )

# @st.cache_resource
# def get_answer(query, _docembeddings):
#     """Cache query results"""
#     try:

#         relevant_chunks = _docembeddings.similarity_search_with_score(query,k=2)

#         chunk_docs=[]
#         for chunk in relevant_chunks:
#             chunk_docs.append(chunk[0])
#         chain = setup_qa_chain()
#         results = chain({"input_documents": chunk_docs, "question": query})

#         print(results)

#         text_reference=""
#         for i in range(len(results["input_documents"])):
#             text_reference+=results["input_documents"][i].page_content
#         output={"Answer":results["output_text"],"Reference":text_reference}
#         return output

#         # relevant_chunks = _docembeddings.similarity_search_with_score(query, k=3)
#         # chunk_docs = [chunk[0] for chunk in relevant_chunks]
        
#         # chain = setup_qa_chain()
#         # results = chain({"input_documents": chunk_docs, "question": query})
        
#         # # Collect unique source documents
#         # sources = set()
#         # for doc in results["input_documents"]:
#         #     if 'source' in doc.metadata:
#         #         sources.add(doc.metadata['source'])
        
#         # text_reference = "\n\n".join([
#         #     f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
#         #     for doc in results["input_documents"]
#         # ])
        
#         # return {
#         #     "Answer": results["output_text"],
#         #     "Reference": text_reference,
#         #     "Sources": list(sources)
#         # }

#     except Exception as e:
#         return {
#             "Answer": f"Error processing query: {str(e)}",
#             "Reference": "",
#             "Sources": []
#         }

# def render_document_manager():
#     """Render the document management interface"""
#     st.sidebar.header("📁 Document Manager")
    
#     # Document upload
#     uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type="pdf")
#     if uploaded_file:
#         try:
#             if not os.path.exists("docs"):
#                 os.makedirs("docs")
            
#             file_path = os.path.join("docs", uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             # Add metadata
#             description = st.sidebar.text_area(
#                 "Add document description (optional):",
#                 key=f"desc_{uploaded_file.name}"
#             )
            
#             st.session_state.document_metadata[uploaded_file.name] = {
#                 'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 'description': description
#             }
            
#             save_document_metadata()
#             st.sidebar.success(f"Uploaded {uploaded_file.name} successfully!")
            
#             # Reset processor initialization flag to reprocess documents
#             st.session_state.processor_initialized = False
#             st.rerun()
            
#         except Exception as e:
#             st.sidebar.error(f"Error uploading file: {str(e)}")

# def render_document_list():
#     """Render the list of uploaded documents"""
#     st.sidebar.header("📚 Uploaded Documents")
    
#     if not os.path.exists("docs"):
#         st.sidebar.warning("No documents directory found.")
#         return
    
#     pdf_files = [f for f in os.listdir("docs") if f.endswith('.pdf')]
    
#     if not pdf_files:
#         st.sidebar.warning("No documents uploaded yet.")
#         return
    
#     for pdf_file in pdf_files:
#         with st.sidebar.expander(f"📄 {pdf_file}"):
#             metadata = st.session_state.document_metadata.get(pdf_file, {})
#             st.write(f"📅 Uploaded: {metadata.get('upload_date', 'Unknown')}")
#             st.write(f"📝 Description: {metadata.get('description', 'No description provided')}")
            
#             if st.button(f"Delete {pdf_file}", key=f"del_{pdf_file}"):
#                 try:
#                     os.remove(os.path.join("docs", pdf_file))
#                     if pdf_file in st.session_state.document_metadata:
#                         del st.session_state.document_metadata[pdf_file]
#                     save_document_metadata()
#                     st.session_state.processor_initialized = False
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Error deleting file: {str(e)}")

# def main():
#     st.title("📚 Multi-Document Q&A System")
    
#     # Load document metadata
#     load_document_metadata()
    
#     # Render document management interface
#     render_document_manager()
#     render_document_list()
    
#     # Main interface
#     openai_api_key = st.text_input("OpenAI API Key")

#     os.environ["OPENAI_API_KEY"] = openai_api_key

#     if not st.session_state.processor_initialized:
#         st.warning("⚠️ Document processor not initialized. Please initialize first.")
#         if st.button("Initialize Document Processor"):
#             with st.spinner("Initializing... This may take a few minutes..."):
#                 if initialize_processor():
#                     st.success("✅ Processor initialized successfully!")
#                 else:
#                     st.error("❌ Failed to initialize processor")
#     else:
#         st.success("✅ Document processor is ready!")

#     # Query input
#     query = st.text_input("Enter your question:", placeholder="What would you like to know about the documents?")

#     # Process query
#     if query and st.session_state.processor_initialized:
#         with st.spinner("Processing your question..."):
#             result = get_answer(query, st.session_state.docembeddings)
            
#             # Display answer
#             st.header("Answer")
#             st.write(result["Answer"])
            
#             # # Display sources
#             # if result["Sources"]:
#             #     st.subheader("Sources Referenced")
#             #     for source in result["Sources"]:
#             #         st.write(f"- 📄 {source}")
            
#             # Display reference
#             with st.expander("View Detailed References"):
#                 st.write(result["Reference"])

# if __name__ == "__main__":
#     main()
