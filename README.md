# Claude Document QA Chatbot

A powerful document question-answering system using Claude AI and Google Drive integration with advanced accuracy optimization features, real-time streaming responses, and intelligent document filtering.

## Features

### 🤖 Core Functionality
- **Google Drive Integration**: Automatically loads documents from your Google Drive
- **Claude AI Powered**: Uses Anthropic's Claude models for intelligent responses
- **Real-time Streaming**: Responses appear character-by-character like ChatGPT
- **Vector Search**: FAISS-based semantic search for relevant document retrieval
- **Response Evaluation**: Built-in quality assessment of AI responses
- **Smart Document Filtering**: Only includes documents that actually mention the person/query
- **Professional UI**: Beautiful loading screens and responsive design

### ⚙️ Accuracy Optimization Settings

The web app includes comprehensive settings to optimize accuracy:

#### **Model Selection**
- **Claude 3 Haiku**: Fast and efficient (default)
- **Claude 3 Sonnet**: Balanced performance and accuracy
- **Claude 3 Opus**: Highest accuracy but slower

#### **Document Processing**
- **Chunk Size**: Controls how documents are split (500-2000 tokens)
  - Larger chunks = more context but may reduce precision
  - Smaller chunks = more precise but may miss context
- **Chunk Overlap**: Prevents information loss at boundaries (50-500 tokens)
  - Higher overlap = better context preservation
  - Lower overlap = more efficient processing

#### **Retrieval Settings**
- **Retrieval Count**: Number of document chunks to retrieve (2-10)
  - More chunks = broader context
  - Fewer chunks = more focused responses
- **Search Strategy**:
  - `similarity`: Best semantic matches
  - `mmr`: Maximum Marginal Relevance (diverse results)
  - `similarity_score_threshold`: High confidence only
- **Confidence Threshold**: Minimum similarity score (0.0-1.0)
  - Higher threshold = more relevant documents
  - Lower threshold = more comprehensive search

#### **Response Generation**
- **Temperature**: Controls response creativity (0.0-1.0)
  - Lower values = more focused and consistent
  - Higher values = more creative and varied
- **Prompt Template**:
  - `default`: Standard QA format
  - `detailed`: Comprehensive explanations
  - `concise`: Brief, to-the-point answers
  - `academic`: Scholarly, well-referenced responses
- **Max Context Length**: Maximum tokens for context (2000-8000)
  - Larger context = more information available
  - Smaller context = faster processing

#### **Quality Assessment**
- **Response Evaluation**: Toggle quality scoring on/off
  - Relevance score (0-5)
  - Completeness score (0-5)
  - Overall quality score

### 🚀 Advanced Features

#### **Smart Document Filtering**
When you ask about a specific person (e.g., "tell me about John"), the system:
- **Only retrieves documents** that actually contain "John"
- **Excludes irrelevant documents** automatically
- **Shows debug information** about which documents are included/excluded
- **Prevents information from wrong people** being mixed in

#### **Real-time Streaming**
- **Immediate response start** - no waiting for complete generation
- **Character-by-character display** - natural typing effect
- **Professional experience** - like ChatGPT and modern AI assistants
- **Better user engagement** - users see progress in real-time

#### **Enhanced User Experience**
- **Loading screens** - professional initialization experience
- **Responsive design** - works on all devices
- **Clean chat interface** - modern, intuitive design
- **Stable indicators** - no flickering or changing text

## Usage

1. **Start the Application**:
```bash
python web_app.py
```

2. **Initial Loading**: Wait for the beautiful loading screen to complete initialization

3. **Login**: Enter your email ID to authenticate with Google Drive

4. **Ask Questions**: The system will:
   - Show a stable loading indicator while processing
   - Stream responses character-by-character in real-time
   - Only use documents that actually mention the person/query
   - Provide clean, deduplicated sources

## Enhanced Features

### **Smart Document Filtering**
When you ask about a specific person (e.g., "tell me about John"), the system:
- **Only retrieves documents** that actually contain "John"
- **Excludes irrelevant documents** automatically
- **Shows debug information** about which documents are included/excluded
- **Prevents information from wrong people** being mixed in

### **Real-time Streaming**
- **Immediate response start** - no waiting for complete generation
- **Character-by-character display** - natural typing effect
- **Professional experience** - like ChatGPT and modern AI assistants
- **Better user engagement** - users see progress in real-time

### **Improved UI/UX**
- **Loading screens** - professional initialization experience
- **Responsive design** - works on all devices
- **Clean chat interface** - modern, intuitive design
- **Stable indicators** - no flickering or changing text

## Accuracy Tips

### For Technical Documents
- Use **academic** prompt template
- Set **chunk size** to 1500-2000
- Use **similarity_score_threshold** with 0.8+ confidence
- Enable **evaluation** for quality feedback

### For General Documents
- Use **detailed** prompt template
- Set **chunk size** to 1000-1500
- Use **mmr** search strategy for diverse results
- Set **temperature** to 0.1-0.3 for consistency

### For Quick Queries
- Use **concise** prompt template
- Set **chunk size** to 500-1000
- Use **similarity** search strategy
- Set **retrieval count** to 2-4

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
ANTHROPIC_API_KEY=your_claude_api_key
```

3. Configure Google Drive credentials:
   - Place `credentials.json` in the project root
   - Follow Google Drive API setup instructions

## File Structure

```
PythonProject/
├── web_app.py          # Main web application with streaming UI
├── main.py             # Core QA functionality with enhanced filtering
├── evaluator.py        # Response quality evaluation
├── requirements.txt    # Python dependencies
├── credentials.json    # Google Drive API credentials
├── tokens/             # OAuth tokens (auto-generated)
├── user_data/          # User-specific data and indexes
└── vectorstores/       # FAISS vector databases
```

## Data Flow Diagram (DFD)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Gradio Web UI   │───▶│  Authentication │
│   (Question)    │    │   (web_app.py)   │    │   (Google OAuth)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streaming      │◀───│  Query Processing│◀───│  Google Drive   │
│  Response       │    │   (main.py)      │    │   Documents     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Response       │◀───│  Claude AI       │◀───│  Document       │
│  Evaluation     │    │  (Anthropic)     │    │  Retrieval      │
│  (evaluator.py) │    │                  │    │  (FAISS)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Output    │◀───│  Real-time       │◀───│  Quality        │
│  (Streaming)    │    │  Streaming       │    │  Assessment     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Data Flow Steps:**

1. **User Input** → User types question in Gradio interface
2. **Authentication** → Google OAuth validates user and loads documents
3. **Document Retrieval** → FAISS vector search finds relevant documents
4. **Smart Filtering** → Only documents mentioning the person/query are selected
5. **Query Processing** → Documents are processed and context is prepared
6. **Claude AI** → Anthropic's Claude generates response
7. **Real-time Streaming** → Response streams character-by-character
8. **Quality Evaluation** → Response is evaluated for relevance and completeness
9. **User Output** → Final response with sources and evaluation displayed

### **Key Components:**

- **Gradio UI**: Modern web interface with streaming capabilities
- **Google Drive API**: Secure document access and authentication
- **FAISS Vector DB**: Fast semantic search and document retrieval
- **Claude AI**: Intelligent response generation with streaming
- **Evaluation System**: Quality assessment and feedback
- **Smart Filtering**: Ensures only relevant documents are used

## Advanced Configuration

### Custom Prompt Templates
You can modify the prompt templates in `main.py` to suit your specific use case:

```python
prompt_templates = {
    "custom": """Your custom prompt template here.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
}
```

### Embedding Models
The system uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings. You can change this in `main.py`:

```python
embeddings = HuggingFaceEmbeddings(model_name="your-preferred-model")
```

## Troubleshooting

### Common Issues
1. **Low accuracy**: Try increasing chunk size, retrieval count, or switching to a more powerful model
2. **Slow responses**: Reduce chunk size, retrieval count, or use Claude 3 Haiku
3. **Irrelevant results**: The enhanced filtering should prevent this - check debug logs
4. **Missing context**: Increase chunk overlap or retrieval count
5. **Streaming not working**: Ensure you have the latest Gradio version

### Performance Optimization
- Use appropriate chunk sizes for your document types
- Balance retrieval count with response quality
- Consider document preprocessing for better chunking
- Monitor evaluation scores to tune settings
- The enhanced filtering reduces noise automatically

## Contributing

Feel free to submit issues and enhancement requests!