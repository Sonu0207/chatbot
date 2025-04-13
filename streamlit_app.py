import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import tempfile
import os
import torch


# Load LLaMA 3 model with token
@st.cache_resource
def load_model():
    model_id = "meta-llama/Meta-Llama-3-8B"
    access_token = "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"

    if not access_token:
        st.error("Hugging Face access token is not set. Please set it as an environment variable.")
        return None

    # Try to import BitsAndBytesConfig
    bnb_config = None
    use_quantization = False
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        use_quantization = True
        st.info("BitsAndBytesConfig loaded successfully. Will attempt 4-bit quantization.")
    except Exception as e:
        st.warning(f"BitsAndBytesConfig not available ({e}). Proceeding without quantization.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

    # Try to load the model
    try:
        if use_quantization:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
                token=access_token,
                trust_remote_code=True
            )
            st.success("Model loaded with 4-bit quantization. 🚀")
        else:
            raise Exception("Skipping quantization.")

    except Exception as e:
        st.warning(f"Failed to load 4-bit quantized model ({e}). Loading CPU model instead.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            token=access_token,
            trust_remote_code=True
        )
        st.success("Model loaded on CPU. 🖥️ (No quantization)")

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


# Helper to load documents
def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

# Preprocessing uploaded documents
def process_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    return documents

# Embed documents into ChromaDB
def embed_documents(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb

# Main Streamlit App
def main():
    st.title("📚 Document QA with LLaMA 3 (RAG)")

    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload your documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        all_docs = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            docs = load_document(file_path, uploaded_file.type.split('/')[-1])
            all_docs.extend(docs)

        st.success("Documents uploaded and loaded successfully! ✅")

        processed_docs = process_documents(all_docs)
        vectordb = embed_documents(processed_docs)

        llm = load_model()
        if llm:
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), chain_type="stuff")

            st.header("Ask a Question about Your Documents")
            query = st.text_input("Enter your question here:")

            if query:
                with st.spinner("Generating response..."):
                    answer = qa_chain.run(query)
                    st.write("### Response:")
                    st.write(answer)

if __name__ == "__main__":
    main()
