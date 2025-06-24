"""
Streamlit application with integrated PII filtering for AWS RAG-based chatbot.
Supports both Bedrock and OpenAI models with real-time PII detection.
"""

import logging
import os
import uuid
from typing import Dict, List, Tuple
import tempfile
import time

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from aws_rag_quickstart.AgentLambda import main as agent_handler
from aws_rag_quickstart.IngestionLambda import main as ingest_handler
from aws_rag_quickstart.pii_detector import PIIDetector
from aws_rag_quickstart.bedrock_llm import BedrockLLM
from aws_rag_quickstart.constants import ALL_MODELS, BEDROCK_MODELS, OPENAI_MODELS
from aws_rag_quickstart.opensearch import get_opensearch_connection, create_index_opensearch
from aws_rag_quickstart.LLM import Embeddings

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS RAG Chatbot with PII Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* CSS Variables for theme support */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --border-color: #dee2e6;
        --accent-blue: #1f77b4;
        --accent-green: #28a745;
        --accent-yellow: #ffc107;
        --accent-red: #dc3545;
    }
    
    [data-theme="dark"] {
        --bg-primary: #1a1a1a;
        --bg-secondary: #2d2d2d;
        --bg-tertiary: #404040;
        --text-primary: #ffffff;
        --text-secondary: #cccccc;
        --border-color: #404040;
        --accent-blue: #4dabf7;
        --accent-green: #51cf66;
        --accent-yellow: #ffd43b;
        --accent-red: #ff6b6b;
    }
    
    /* Apply theme to Streamlit elements */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--accent-blue);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    
    .user-message {
        background-color: var(--bg-secondary);
        border-left: 4px solid var(--accent-blue);
        color: var(--text-primary);
    }
    
    .assistant-message {
        background-color: var(--bg-tertiary);
        border-left: 4px solid var(--accent-green);
        color: var(--text-primary);
    }
    
    .pii-warning {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid var(--accent-yellow);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid var(--accent-green);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .processing-message {
        background-color: rgba(31, 119, 180, 0.1);
        border: 1px solid var(--accent-blue);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .theme-selector {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: var(--bg-secondary);
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    /* Streamlit widget styling */
    .stSelectbox > div > div {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    .stButton > button {
        background-color: var(--accent-blue);
        color: white;
        border: none;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-green);
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid var(--accent-blue);
    }
    
    .stAlert > div {
        color: var(--text-primary) !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid var(--accent-yellow);
    }
    
    .stWarning > div {
        color: var(--text-primary) !important;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid var(--accent-red);
    }
    
    .stError > div {
        color: var(--text-primary) !important;
    }
    
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid var(--accent-green);
    }
    
    .stSuccess > div {
        color: var(--text-primary) !important;
    }
    
    /* Right panel specific styling */
    .element-container .stInfo p,
    .element-container .stInfo div,
    .element-container .stInfo span {
        color: var(--text-primary) !important;
    }
    
    /* Subheader styling */
    .stApp h3 {
        color: var(--text-primary) !important;
    }
    
    /* General text styling */
    .stMarkdown p,
    .stMarkdown div,
    .stMarkdown span {
        color: var(--text-primary) !important;
    }
</style>
<script>
function setTheme(theme) {
    const root = document.documentElement;
    if (theme === 'system') {
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        theme = systemPrefersDark ? 'dark' : 'light';
    }
    
    if (theme === 'dark') {
        root.setAttribute('data-theme', 'dark');
    } else {
        root.removeAttribute('data-theme');
    }
    
    // Store theme preference
    localStorage.setItem('theme-preference', theme);
}

// Initialize theme on load
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme-preference') || 'system';
    setTheme(savedTheme);
});

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
    const savedTheme = localStorage.getItem('theme-preference');
    if (savedTheme === 'system' || !savedTheme) {
        setTheme('system');
    }
});
</script>
""", unsafe_allow_html=True)

def ensure_opensearch_index():
    """Ensure OpenSearch index exists before starting the chat"""
    try:
        index_name = os.getenv("INDEX_NAME", "rag-index")
        
        # Get OpenSearch connection
        os_client = get_opensearch_connection()
        
        # Check if index exists
        if not os_client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} does not exist. Creating it...")
            
            # Create embeddings instance
            embeddings = Embeddings()
            
            # Create the index
            with st.spinner(f"Creating OpenSearch index '{index_name}'..."):
                create_index_opensearch(os_client, embeddings, index_name)
            
            st.success(f"✅ OpenSearch index '{index_name}' created successfully!")
            logger.info(f"Successfully created index: {index_name}")
        else:
            logger.info(f"Index {index_name} already exists")
            
    except Exception as e:
        logger.error(f"Error ensuring OpenSearch index: {e}")
        st.error(f"Failed to initialize OpenSearch index: {str(e)}")
        st.warning("The application may not work correctly without a proper OpenSearch index.")

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_ids" not in st.session_state:
        st.session_state.document_ids = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "pii_detector" not in st.session_state:
        with st.spinner("Loading PII detection model..."):
            st.session_state.pii_detector = PIIDetector()
    if "is_local" not in st.session_state:
        st.session_state.is_local = bool(int(os.getenv("LOCAL", "0")))
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = ALL_MODELS[12] if len(ALL_MODELS) > 12 else ALL_MODELS[0]
    if "opensearch_initialized" not in st.session_state:
        ensure_opensearch_index()
        st.session_state.opensearch_initialized = True
    if "theme" not in st.session_state:
        st.session_state.theme = "system"

def get_available_models() -> List[str]:
    """Get list of available Bedrock and OpenAI models"""
    # For local testing, allow both Bedrock and OpenAI models
    # Remove Ollama dependency for local testing
    return ALL_MODELS

def display_sidebar():
    """Display the sidebar with model selection and file upload"""
    with st.sidebar:
        st.header("Configuration")
        
        # Theme selection
        st.markdown('<div class="theme-selector">', unsafe_allow_html=True)
        theme_options = ["system", "light", "dark"]
        theme_labels = ["🔄 System", "☀️ Light", "🌙 Dark"]
        
        selected_theme = st.selectbox(
            "Theme",
            options=theme_options,
            format_func=lambda x: theme_labels[theme_options.index(x)],
            index=theme_options.index(st.session_state.theme),
            help="Choose your preferred theme"
        )
        
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            # JavaScript to update theme
            st.markdown(f"""
            <script>
                setTheme('{selected_theme}');
            </script>
            """, unsafe_allow_html=True)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
        
        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
            help="Choose the AI model for processing your queries"
        )
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            logger.info(f"Updated model selection: {selected_model}")
        
        st.divider()
        
        # File upload section
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to ask questions about"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Session information
        st.header("Session Info")
        st.info(f"User ID: {st.session_state.user_id[:8]}...")
        st.info(f"Documents: {len(st.session_state.document_ids)}")
        st.info(f"Messages: {len(st.session_state.messages)}")
        
        # Clear session button
        if st.button("Clear Session", type="secondary"):
            st.session_state.messages = []
            st.session_state.document_ids = []
            st.session_state.user_id = str(uuid.uuid4())
            st.rerun()

def process_uploaded_files(uploaded_files: List[UploadedFile]):
    """Process uploaded files and ingest them into the RAG system"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        local_storage_dir = tempfile.mkdtemp()
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save file temporarily
            file_path = os.path.join(local_storage_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Create ingestion event
            event = {
                "unique_id": st.session_state.user_id,
                "file_path": file_path,
                "use_local_storage": True,
                "model_id": st.session_state.selected_model
            }
            
            try:
                # Process the file
                response = ingest_handler(event)
                
                # Add to document IDs
                file_key = f"{st.session_state.user_id}_{uploaded_file.name}"
                st.session_state.document_ids.append(file_key)
                
                logger.info(f"Ingested document: {file_key}")
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        st.success(f"Successfully processed {len(uploaded_files)} files. You can now ask questions!")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(local_storage_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"Error in file processing: {e}")
        st.error(f"Error processing files: {str(e)}")

def display_chat_messages():
    """Display chat messages with proper styling"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            with st.container():
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        elif message["role"] == "pii_warning":
            with st.container():
                st.markdown(f"""
                <div class="pii-warning">
                    <strong>⚠️ PII Filter:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

def process_user_message(user_input: str) -> Tuple[bool, str]:
    """Process user message with PII filtering and RAG query"""
    
    # PII Detection
    is_safe, message_content, detected_entities = st.session_state.pii_detector.filter_text(user_input)
    logger.info(f"PII detection result: {detected_entities}, safe: {is_safe}")
    
    if not is_safe:
        # PII detected, reject the message
        st.session_state.messages.append({
            "role": "pii_warning",
            "content": message_content
        })
        return False, message_content
    
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    try:
        # Process the query through the RAG system
        event = {
            "unique_ids": st.session_state.document_ids,
            "question": user_input,
            "model_id": st.session_state.selected_model
        }
        
        with st.spinner("Thinking..."):
            response = agent_handler(event)
        
        # Add assistant response to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        return True, response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_message = "An error occurred while processing your query. Please try again."
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })
        return False, error_message

def display_welcome_message():
    """Display welcome message and instructions"""
    if not st.session_state.messages:
        st.markdown("""
        <div class="success-message">
            <h3>🛡️ Welcome to AWS RAG Chatbot with PII Protection</h3>
            <p>This chatbot helps you ask questions about your documents while protecting your privacy:</p>
            <ul>
                <li><strong>Upload Documents:</strong> Use the sidebar to upload PDF files</li>
                <li><strong>Ask Questions:</strong> Type your questions in the chat input below</li>
                <li><strong>PII Protection:</strong> All inputs are automatically screened for personal information</li>
                <li><strong>Model Selection:</strong> Choose from various AI models in the sidebar</li>
            </ul>
            <p><em>Note: Messages containing PII will be automatically rejected to protect your privacy.</em></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize theme on app load
    st.markdown(f"""
    <script>
        // Initialize theme immediately
        setTheme('{st.session_state.theme}');
    </script>
    """, unsafe_allow_html=True)
    
    # Display header
    st.markdown('<h1 class="main-header">🛡️ AWS RAG Chatbot with PII Protection</h1>', unsafe_allow_html=True)
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display welcome message
        display_welcome_message()
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            display_chat_messages()
        
        # Chat input
        if user_input := st.chat_input("Ask a question about your documents..."):
            with chat_container:
                # Process the message
                is_successful, response = process_user_message(user_input)
                
                # Rerun to display new messages
                st.rerun()
    
    with col2:
        # PII Detection Info
        st.subheader("🔒 PII Protection")
        st.info("""
        **Protected Information:**
        - Person names
        - Organizations
        - Locations
        - Email addresses
        - Phone numbers
        - Social security numbers
        - Credit card numbers
        - IP addresses
        """)
        
        # Model Information
        st.subheader("🤖 Current Model")
        st.info(f"**Selected:** {st.session_state.selected_model}")
        
        if st.session_state.selected_model in BEDROCK_MODELS:
            st.info("🌟 **AWS Bedrock Model**\n\nRecommended for production use")
        elif st.session_state.selected_model in OPENAI_MODELS:
            st.info("🧠 **OpenAI Model**\n\nRequires OPENAI_API_KEY in environment")

if __name__ == "__main__":
    main() 