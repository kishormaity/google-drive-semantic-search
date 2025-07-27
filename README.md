# Claude Document QA Chatbot

A sophisticated document question-answering system powered by Claude AI, featuring Google Drive integration, real-time streaming responses, and intelligent document filtering.

## Features

### Core Functionality
- **Google Drive Integration**: Seamlessly load documents from personal and shared drives
- **Multi-format Support**: PDF, DOCX, XLSX, Google Docs, Google Sheets, and more
- **Intelligent Retrieval**: Advanced vector search with multiple strategies (MMR, similarity, threshold-based)
- **Claude AI Integration**: Powered by Anthropic's Claude models for accurate responses
- **Real-time Streaming**: Character-by-character response generation for natural interaction

### Accuracy Optimization Settings
- **Smart Document Filtering**: Person-specific queries automatically filter to relevant documents
- **Source Deduplication**: Prevents duplicate information from multiple sources
- **Content Validation**: Ensures responses are based on actual document content
- **Quality Evaluation**: Built-in response quality assessment
- **Relevance Thresholds**: Configurable confidence levels for document retrieval

### User Experience
- **Professional UI**: Clean, modern interface built with Gradio
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Scrollable Input**: Dynamic input box that adapts to content length
- **Loading Indicators**: Stable processing indicators and initial loading screen
- **Session Management**: Persistent user sessions with document caching

### Advanced Features
- **Multi-strategy Retrieval**: Combines MMR, similarity, and threshold-based search
- **Comprehensive Debugging**: Detailed logging and file analysis tools
- **Error Handling**: Robust error handling with helpful troubleshooting messages
- **Performance Optimization**: Efficient document chunking and vector indexing
- **Extensible Architecture**: Easy to add new document types and features

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PythonProject
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Drive API**:
   - Create a Google Cloud Project
   - Enable Google Drive API
   - Download `credentials.json` and place it in the project root
   - Share the credentials file with your team members

4. **Run the application**:
   ```bash
   python web_app.py
   ```

## Usage

1. **Start the application**: Run `python web_app.py` and open the provided URL
2. **Login**: Enter your email ID to authenticate with Google Drive
3. **Ask questions**: Type your questions about the documents in your Google Drive
4. **Get answers**: Receive real-time streaming responses with source citations

### Example Queries
- "What is John's work experience?"
- "Tell me about Sarah's education background"
- "What are the key skills mentioned in the resumes?"
- "Show me all the contact information for candidates"

## Configuration

The system supports various configuration options:

```python
default_config = {
    "model": "claude-3-haiku-20240307",
    "temperature": 0.01,
    "retrieval_count": 8,
    "search_strategy": "mmr",
    "confidence_threshold": 0.5,
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "evaluation_enabled": True,
    "max_context_length": 6000
}
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

### Key Components:
1. **User Input**: Questions entered through the web interface
2. **Authentication**: Google OAuth for secure document access
3. **Document Loading**: Comprehensive loading from all accessible Google Drive locations
4. **Vector Search**: Multi-strategy retrieval using FAISS
5. **AI Processing**: Claude AI generates contextual responses
6. **Quality Assessment**: Built-in evaluation of response quality
7. **Streaming Output**: Real-time character-by-character response delivery

## Troubleshooting

### Common Issues

**No documents found**:
- Check if files are in supported formats (PDF, DOCX, XLSX, etc.)
- Verify Google Drive permissions and sharing settings
- Use debug mode to see accessible files: `debug:your-email@domain.com`

**Login issues**:
- Ensure `credentials.json` is properly configured
- Check internet connection and Google API access
- Clear browser cache and try again

**Poor response quality**:
- Increase `retrieval_count` for more comprehensive search
- Adjust `confidence_threshold` for better relevance filtering
- Check if documents contain the information being asked about

**Performance issues**:
- Reduce `chunk_size` for faster processing
- Lower `max_context_length` for memory optimization
- Use `use_existing=True` to reuse cached vector indexes

### Debug Commands

- **List files**: `debug:your-email@domain.com`
- **Test loading**: `test:your-email@domain.com`
- **Comprehensive analysis**: Check console output for detailed file analysis

## Dependencies

- **Gradio**: Web interface framework
- **FAISS**: Vector similarity search
- **LangChain**: LLM application framework
- **Google APIs**: Drive integration and authentication
- **Anthropic**: Claude AI model access
- **Pydantic**: Data validation and parsing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.