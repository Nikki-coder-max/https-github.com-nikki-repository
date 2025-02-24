import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pdfplumber
import pandas as pd
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Step 1: Extract text, images, and tables from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_images_from_pdf(pdf_path, output_folder):
    """Extract images from a PDF and save them to a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_path = os.path.join(output_folder, f"page_{page_num+1}_img_{img_index+1}.{base_image['ext']}")
            with open(img_path, "wb") as f:
                f.write(base_image["image"])
            image_paths.append(img_path)
    return image_paths

def perform_ocr_on_images(image_paths):
    """Perform OCR on extracted images."""
    ocr_texts = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            text = pytesseract.image_to_string(image).strip()
            if text:
                ocr_texts.append(text)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return ocr_texts

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF."""
    table_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                df = pd.DataFrame(table)
                table_texts.append(f"Table {table_index + 1} on Page {page_num}:\n{df.to_string(index=False, header=False)}\n")
    return table_texts

# Step 2: Clean and chunk text
def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines
    text = re.sub(r' {2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)  # Fix hyphenated words
    text = text.replace("\n\n", " ").replace("\n", " ")  # Fix broken lines
    text = re.sub(r' {2,}', ' ', text)  # Remove excessive spaces
    text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)  # Fix hyphenated word breaks
    text = text.replace("\n\n", " ")  # Fix broken line splits
    text = text.replace("\n", " ")  # Fix remaining broken lines
    return text.strip()

def chunk_text(text, chunk_size=600, overlap=100):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "?", ".", "!", "\n", " "]
    )
    return text_splitter.split_text(text)

# Step 3: Create Pinecone index and store embeddings
def create_pinecone_index(index_name, dimension=384):
    """Create a Pinecone index."""
    os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Pinecone index '{index_name}' created successfully!")

def check_index_exists(index_name):
    """Check if a Pinecone index exists."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = pc.list_indexes()
    return index_name in existing_indexes

def store_embeddings_in_pinecone(text_chunks, embedding_model, index_name):
    """Store embeddings in Pinecone."""
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embedding_model
    )
    return docsearch

# Step 4: Hybrid retrieval with BM25 and dense embeddings
def hybrid_retriever(query, bm25, embedder, retrieved_docs, top_k=5):
    """Perform hybrid retrieval using BM25 and dense embeddings."""
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    
    # BM25 scores
    bm25_scores = bm25.get_scores(word_tokenize(query.lower()))
    
    # Dense embeddings
    doc_embeddings = np.array(embedder.encode(retrieved_texts))
    query_embedding = embedder.encode(query).reshape(1, -1)
    
    # FAISS index for dense retrieval
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    _, dense_indices = index.search(query_embedding, k=len(retrieved_texts))
    dense_scores = np.exp(-dense_indices[0])  # Convert distances to scores
    
    # Normalize scores
    scaler = MinMaxScaler()
    bm25_scores_scaled = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
    dense_scores_scaled = scaler.fit_transform(np.array(dense_scores).reshape(-1, 1)).flatten()
    
    # Combine scores
    hybrid_scores = 0.5 * bm25_scores_scaled + 0.5 * dense_scores_scaled
    
    # Sort documents by hybrid scores
    return [doc for doc, _ in sorted(zip(retrieved_docs, hybrid_scores), key=lambda x: x[1], reverse=True)[:top_k]]

# Step 5: Rerank documents using a cross-encoder
def rerank_documents(query, retrieved_docs):
    """Rerank documents using a cross-encoder."""
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1]
    return [retrieved_docs[i] for i in ranked_indices[:2]]

# Step 6: Generate response using an LLM with few-shot examples and CoT reasoning
def generate_response(query, reranked_docs):
    """Generate a response using an LLM with few-shot examples and CoT reasoning."""
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=text_generator)

    # Define few-shot examples with CoT reasoning
    few_shot_examples = """
    Example 1:
    Context: Python is a high-level programming language.
    Q: What is Python?
    A: Let's think step-by-step:
       1. Python is described as a "high-level programming language."
       2. High-level languages are known for their simplicity and readability.
       3. Python is widely used in web development, data analysis, and AI.
       Therefore, Python is a high-level programming language known for its simplicity and readability, widely used in web development, data analysis, and AI.

    Example 2:
    Context: Lists in Python are ordered, mutable collections of items.
    Q: How do you create a list in Python?
    A: Let's think step-by-step:
       1. Lists in Python are created using square brackets `[]`.
       2. Items are separated by commas.
       3. For example, `my_list = [1, 2, 3]` creates a list with three integers.
       Therefore, you can create a list in Python using square brackets, like `my_list = [1, 2, 3]`.
    """

    # Define the prompt template with few-shot examples and CoT reasoning
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=f"""You are an AI assistant that answers Python programming questions using retrieved context.
        - Think step-by-step using logical reasoning (Chain-of-Thought).
        - Use relevant examples when needed.
        - Prioritize clarity and accuracy.

        Here are some examples of how to answer Python programming questions:
        {few_shot_examples}

        Now, answer the following Python programming question using the provided context:
        Context: {{context}}
        Question: {{query}}
        Answer: Let's think step-by-step:
        """
    )

    # Prepare the context
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # Chunk the context to fit within the model's token limit
    context_chunks = chunk_text(context, max_tokens=400)
    responses = []

    for chunk in context_chunks:
        # Format the prompt with the chunk and query
        prompt_text = prompt_template.format(context=chunk, query=query)
        
        # Generate the response
        response = text_generator(prompt_text, max_length=250)
        responses.append(response[0]['generated_text'])

    # Combine responses from all chunks
    combined_response = " ".join(responses)

    # Post-process the response to remove unwanted phrases
    cleaned_response = combined_response.replace("If unsure, say 'I don't know'.", "").strip()
    return cleaned_response

# Example usage
pdf_path = r"C:\Users\user\Downloads\health_care\Python_Durga.pdf"
output_folder = r"C:\Users\user\Downloads\health_care\output_image"

# Extract text, images, and tables
text = extract_text_from_pdf(pdf_path)
image_paths = extract_images_from_pdf(pdf_path, output_folder)
ocr_texts = perform_ocr_on_images(image_paths)
table_texts = extract_tables_from_pdf(pdf_path)

# Combine all text data
all_text = text + "\n".join(ocr_texts) + "\n".join(table_texts)

# Clean and chunk the text
cleaned_text = clean_text(all_text)
text_chunks = chunk_text(cleaned_text)

# Initialize the embedding model
embeddings_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Pinecone index and store embeddings
index_name = "durga-new"
if not check_index_exists(index_name):
    create_pinecone_index(index_name)
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")
docsearch = store_embeddings_in_pinecone(text_chunks, embeddings_model, index_name)

# Perform hybrid retrieval
query = "write a program to add two numbers with list comprehension?"
retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.7})
retrieved_docs = retriever.invoke(query)

# Initialize BM25
bm25 = BM25Okapi([word_tokenize(doc.page_content.lower()) for doc in retrieved_docs])

# Initialize the embedding model for hybrid retrieval
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Perform hybrid retrieval
top_docs = hybrid_retriever(query, bm25, embedder, retrieved_docs)

# Rerank documents
reranked_docs = rerank_documents(query, top_docs)

# Generate response
response = generate_response(query, reranked_docs)
print("🤖 AI Response:", response)