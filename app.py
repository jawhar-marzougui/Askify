import streamlit as st
import tempfile
import os
import warnings
import time
import re
import requests
from parsing import parse_pdf, get_text_documents
# Removed language detection
import requests
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.embeddings import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from neo4j import GraphDatabase
import spacy
from dotenv import load_dotenv
# Removed translation dependencies


warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ---- NEO4J SETUP ----
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# ---- ENVIRONMENT VARIABLES ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required environment variables
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable is required")
    st.stop()

if not neo4j_uri or not neo4j_password:
    st.error("NEO4J_URI and NEO4J_PASSWORD environment variables are required")
    st.stop()


# ---- PROMPT TEMPLATES ----
prompt_template = """You are an AI assistant that helps users understand and analyze the content of documents.
Use the following information to answer the user's question.
If you don't know the answer, simply say "No answer found."

Context: {context}
Graph Insights: {graph_insights}
Question: {question}

Instructions:

Read and understand the context carefully

Provide a comprehensive answer based on the available information

If the context contains relevant information, use it to answer the question

Be specific and refer to source material when possible

Format your answer clearly, with important parts in bold

Answer:"""
# Translation function removed - documents are processed directly

# Initialize components
@st.cache_resource(show_spinner=False)
def initialize_components():
    """Initialize embedding model, LLM, and spacy model"""
    # Use Arabic-compatible multilingual embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    
    # Set up LlamaIndex settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # Load English spacy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to multilingual model if English model not available
        nlp = spacy.load("xx_ent_wiki_sm")
        print("Warning: Using multilingual model instead of English-specific model")
    
    # Neo4j driver
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    return embed_model, llm, nlp, driver

def populate_graph(documents, driver, nlp):
    """Extract entities and relationships from documents and populate Neo4j graph - OPTIMIZED"""
    with driver.session() as session:
        # Batch process for better performance
        all_concepts = set()
        all_relationships = []
        
        # Process documents in batches to reduce memory usage
        batch_size = 5  # Process 5 documents at a time
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                # Limit text length for faster NLP processing
                doc_text = doc.text[:2000]  # Only process first 2000 chars per document
                nlp_doc = nlp(doc_text)
                concepts = [ent.text for ent in nlp_doc.ents if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE"]]
                
                # Limit concepts to avoid too many relationships
                concepts = concepts[:10]  # Only take first 10 concepts per document
                all_concepts.update(concepts)
                
                # Create relationships between adjacent concepts
                for j, concept in enumerate(concepts):
                    if j + 1 < len(concepts):
                        all_relationships.append((concept, concepts[j + 1]))
        
        # Batch insert concepts
        if all_concepts:
            concept_list = list(all_concepts)
            session.run(
                "UNWIND $concepts AS concept MERGE (:Concept {name: concept})",
                concepts=concept_list
            )
        
        # Batch insert relationships
        if all_relationships:
            session.run(
                """
                UNWIND $relationships AS rel
                MATCH (c1:Concept {name: rel[0]}), (c2:Concept {name: rel[1]})
                MERGE (c1)-[:RELATED_TO]->(c2)
                """,
                relationships=all_relationships
            )

def get_graph_insights(question, driver):
    """Get graph insights from Neo4j based on the question"""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($question)
            OPTIONAL MATCH (c)-[r:RELATED_TO]->(other:Concept)
            RETURN c.name AS concept, collect(other.name) AS related_concepts
            """,
            question=question
        )
        insights = []
        for record in result:
            insights.append(f"Concept: {record['concept']}, Related Concepts: {', '.join(record['related_concepts'])}")
        return "\n".join(insights) if insights else "No relevant graph insights found."

def create_vector_index(parsed_documents, file_names):
    """Create vector index from parsed documents with metadata"""
    # Convert parsed results to LlamaIndex documents with metadata
    documents = []
    for i, doc in enumerate(parsed_documents):
        # Extract page number from document metadata if available
        page_num = getattr(doc, 'metadata', {}).get('page_label', i + 1)
        if not page_num:
            page_num = i + 1
            
        # Create document with metadata
        document = Document(
            text=doc.text,
            metadata={
                'file_name': file_names[i] if i < len(file_names) else 'Unknown',
                'page_number': page_num,
                'source': f"{file_names[i] if i < len(file_names) else 'Unknown'}, Page {page_num}"
            }
        )
        documents.append(document)
    
    # Create vector store index with optimized settings
    vector_index = VectorStoreIndex.from_documents(
        documents, 
        show_progress=False,  # Disable progress bar for speed
        # Use smaller chunk size for faster processing
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50)
        ]
    )
    return vector_index, documents

def main():
    # Simple, clean header
    st.title("🤖 Askify")
    st.markdown("*Your AI-Powered Document Assistant*")
    st.markdown("""
    <style>
        body, .stChatInput, .stTextInput, .stMarkdown, .stAlert, .stButton>button {
            direction: ltr;
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)
    # New Chat button in sidebar
    with st.sidebar:
        st.markdown("### Controls")
        if st.button("🔄 New Chat", help="Start a new conversation", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Force file uploader to reset by incrementing a counter
            if "file_uploader_key" not in st.session_state:
                st.session_state.file_uploader_key = 0
            st.session_state.file_uploader_key += 1
            st.rerun()
        
        # Show loaded files in sidebar if any
        if "successful_files" in st.session_state:
            st.markdown("### 📄 Loaded Documents")
            for file in st.session_state.successful_files:
                st.markdown(f"• {file}")

    # Initialize components
    embed_model, llm, nlp, driver = initialize_components()

    # Initialize file uploader key if not exists
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    # Simple upload section
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or multiple PDF files",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    # Submit button
    submit_button = st.button("Process Documents", type="primary")

    if uploaded_files and submit_button:
        # Clear chat history when new documents are uploaded
        if "messages" in st.session_state:
            st.session_state.messages = []
            
        all_parsed_documents = []
        successful_files = []
        
        # Process all uploaded files
        all_file_names = []  # Track file names for each document
        
        for uploaded_file in uploaded_files:
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Parse the PDF
                with st.spinner("Loading..."):
                    parsed_result = parse_pdf(tmp_file_path)
                
                if parsed_result:
                    # Get text documents from parsed result
                    text_docs = get_text_documents(parsed_result, split_by_page=True)
                    all_parsed_documents.extend(text_docs)
                    successful_files.append(uploaded_file.name)
                    
                    # Track file name for each document page
                    for _ in text_docs:
                        all_file_names.append(uploaded_file.name)
                else:
                    st.error(f"❌ Failed to parse {uploaded_file.name}")
            
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass  # Ignore cleanup errors

        # If we have successfully parsed documents, create the RAG system
        if all_parsed_documents:
            with st.spinner("Loading..."):
                try:
                    # Create vector index with file names for metadata
                    vector_index, documents = create_vector_index(all_parsed_documents, all_file_names)
                    
                    
                    # Create query engine with similarity_top_k to get source nodes
                    query_engine = vector_index.as_query_engine(similarity_top_k=3, response_mode="compact")
                    st.session_state.query_engine = query_engine
                    st.session_state.successful_files = successful_files
                    st.session_state.driver = driver
                
                except Exception as e:
                    st.error(f"❌ Error creating knowledge base: {str(e)}")
        
        else:
            st.warning("No documents were successfully parsed. Please try uploading valid PDF files.")

    # Q&A Section - Show if query engine exists in session state
    if "query_engine" in st.session_state:
        st.markdown("### 💬 Chat with Your Documents")
        st.markdown("---")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # React to user input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get graph insights using the question directly
                    graph_insights = get_graph_insights(prompt, st.session_state.driver)
                    
                    # Use the English-only template
                    template = prompt_template
                    
                    # Create enhanced query
                    context = f"Documents from uploaded PDFs: {', '.join(st.session_state.successful_files)}"
                    query_prompt = template.format(
                        context=context, 
                        graph_insights=graph_insights, 
                        question=prompt
                    )
                    
                    # Query the engine
                    response = st.session_state.query_engine.query(query_prompt)
                    
                    # Use response directly without translation
                    formatted_response = response.response
                    
                    # Simple formatting: if response contains both answer and additional info, separate them
                    if "Additional Information:" in formatted_response or "Additional Helpful Information:" in formatted_response:
                        # Replace common patterns to ensure proper line breaks
                        formatted_response = formatted_response.replace("Additional Information:", "\n\n**Additional Information:**")
                        formatted_response = formatted_response.replace("Additional Helpful Information:", "\n\n**Additional Information:**")
                        
                        # If it doesn't start with Answer:, add it
                        if not formatted_response.startswith("**Answer:**"):
                            # Find where additional info starts
                            parts = formatted_response.split("\n\n**Additional Information:**")
                            if len(parts) == 2:
                                formatted_response = f"**Answer:** {parts[0].strip()}\n\n**Additional Information:** {parts[1].strip()}"
                    
                    # Create a placeholder for streaming text
                    response_placeholder = st.empty()
                    
                    # Stream the response word by word
                    words = formatted_response.split()
                    displayed_text = ""
                    
                    for i, word in enumerate(words):
                        displayed_text += word + " "
                        response_placeholder.markdown(displayed_text + "▌")  # Add cursor
                        time.sleep(0.1)  # Adjust speed as needed
                    
                    # Remove cursor and show final response
                    response_placeholder.markdown(displayed_text)
                    
                    # Display citations if available
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        st.markdown("---")
                        st.markdown("**📚 Sources:**")
                        
                        citations = []
                        for i, node in enumerate(response.source_nodes, 1):
                            if hasattr(node, 'metadata') and node.metadata:
                                source = node.metadata.get('source', 'Unknown source')
                                citations.append(f"{i}. {source}")
                            else:
                                citations.append(f"{i}. Unknown source")
                        
                        for citation in citations:
                            st.markdown(f"- {citation}")
            
            # Add assistant response to chat history (store the formatted response without streaming effect)
            full_response_with_citations = formatted_response
            if hasattr(response, 'source_nodes') and response.source_nodes:
                citations = []
                for i, node in enumerate(response.source_nodes, 1):
                    if hasattr(node, 'metadata') and node.metadata:
                        source = node.metadata.get('source', 'Unknown source')
                        citations.append(f"{i}. {source}")
                    else:
                        citations.append(f"{i}. Unknown source")
                
                citations_text = "\n\n---\n**📚 Sources:**\n" + "\n".join([f"- {citation}" for citation in citations])
                full_response_with_citations += citations_text
            
            st.session_state.messages.append({"role": "assistant", "content": full_response_with_citations})
    
    # Welcome message when no documents are loaded
    elif not uploaded_files:
        st.info("👋 Welcome to Askify! Upload your PDF documents above to start having intelligent conversations with your content.")

if __name__ == "__main__":
    main()
