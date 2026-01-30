import streamlit as st
import traceback
from model import GraphRAGPipeline  # your GraphRAGPipeline class

from PIL import Image


# ---------------------------
# Configuration
# ---------------------------
icon = Image.open("icon.png")

st.set_page_config(
    page_title="Carplexity", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=icon
)

# ---------------------------
# Load RAG Pipeline
# ---------------------------
@st.cache_resource
def load_rag_pipeline(path="GraphRagConfig.joblib"):
    """Load the RAG pipeline with error handling."""
    try:
        return GraphRAGPipeline.load(path)
    except FileNotFoundError:
        st.error(f"Configuration file '{path}' not found. Please ensure the configuration file exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {str(e)}")
        st.stop()

# Initialize RAG pipeline
with st.spinner("Loading GraphRAG Car Assistant..."):
    rag = load_rag_pipeline()

# ---------------------------
# Sidebar: Show last generated Cypher query
# ---------------------------
st.sidebar.image("image.jpg", caption="", use_container_width=True)
last_cypher = ""

# Safely check for the latest AI message with a cypher_query
if "conversation" in st.session_state:
    for msg in reversed(st.session_state.conversation):
        if (
            msg.get("role") == "ai" 
            and msg.get("metadata") 
            and msg["metadata"].get("cypher_query")
        ):
            last_cypher = msg["metadata"]["cypher_query"]
            break  # Stop at the latest AI message with a query

# Display the Cypher query (blank if none found)
st.sidebar.text_area(
    label="Cypher Query",
    value=last_cypher,
    height=200,
    disabled=True
)

# ---------------------------
# UI Components
# ---------------------------

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Advanced options in sidebar
with st.sidebar.expander("🔧 Advanced Options"):
    show_query_details = st.checkbox("Show Cypher queries", value=False, help="Display the generated Cypher queries")

# Initialize conversation if not present
if "conversation" not in st.session_state:
    st.session_state.conversation = []



# ---------------------------
# Main Chat Interface
# ---------------------------
st.subheader("Chat with your car expert")


# Chat container
chat_container = st.container()

# User input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:", 
            placeholder="Ask about cars, prices, models, comparisons...",
            label_visibility="collapsed"
        )
    
    with col2:
        submitted = st.form_submit_button("Send 📤", use_container_width=True)

# Handle example query selection
if hasattr(st.session_state, 'pending_query'):
    user_input = st.session_state.pending_query
    submitted = True
    delattr(st.session_state, 'pending_query')

# Process user input
if submitted and user_input and user_input.strip():
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "message": user_input})
    
    # Generate response
    with st.spinner("🔍 Typing..."):
        try:
            # Use the direct answer method for better control
            response_data = rag.answer(user_input)
            
            if isinstance(response_data, dict):
                response = response_data.get("answer", "Sorry, I couldn't generate a response.")
                intent = response_data.get("intent", "unknown")
                used_context = response_data.get("used_context", False)
                cypher_query = response_data.get("cypher_query", "")
                results_count = response_data.get("results_count", 0)
                
                # Create response metadata
                metadata = {
                    "intent": intent,
                    "used_context": used_context,
                    "cypher_query": cypher_query,
                    "results_count": results_count
                }
            else:
                # Fallback for string responses
                response = str(response_data)
                metadata = {"intent": "unknown", "used_context": False}
            
        except Exception as e:
            response = f"I apologize, but I encountered an error: {str(e)}"
            metadata = {"intent": "error", "used_context": False}
            
            # Log the error for debugging
            st.error("Error details:")
            st.code(traceback.format_exc())
    
    # Add AI response to conversation
    st.session_state.conversation.append({
        "role": "ai", 
        "message": response,
        "metadata": metadata
    })

# ---------------------------
# Chat Display
# ---------------------------
with chat_container:
    for i, chat in enumerate(st.session_state.conversation):
        if chat["role"] == "user":
            # User message
            st.markdown(
                f"""
                <div style='
                    text-align: right; 
                    background: linear-gradient(135deg, #34A853, #2E7D32);
                    color: white; 
                    padding: 15px 20px; 
                    border-radius: 20px 20px 5px 20px; 
                    margin: 10px 0 10px 20%; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    font-weight: 500;
                '>
                    {chat['message']}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        else:
            # AI response
            st.markdown(
                f"""
                <div style='
                    text-align: left; 
                    background: linear-gradient(135deg, #a8edea, #fed6e3);
                    color: #333; 
                    padding: 15px 20px; 
                    border-radius: 20px 20px 20px 5px; 
                    margin: 10px 20% 10px 0; 
                    box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
                    line-height: 1.6;
                    border: 1px solid rgba(168, 237, 234, 0.4);
                '>
                    {chat['message']}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Show metadata if requested
            if "metadata" in chat:
                metadata = chat["metadata"]
                
                # Create columns for metadata display
                if show_query_details:
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    if show_query_details and metadata.get("cypher_query"):
                        with col3:
                            with st.expander("<Cypher>"):
                                st.code(metadata["cypher_query"], language="cypher")
                
                st.markdown("---")

# ---------------------------
# Footer
# ---------------------------
if len(st.session_state.conversation) == 0:
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 50px;'>
        <h3>Carplexity</h3>
        <p>I can help you find information about cars, compare models, check prices, and answer questions about automotive specifications.</p>
        <p><em>Try asking me about specific car models, price ranges, or comparisons!</em></p>
    </div>
    """, unsafe_allow_html=True)

# Add some CSS for better styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: black;
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 10px 15px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1A73E8;
        box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
    }
    
    .stButton > button {
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)