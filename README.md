# Claude Document QA Chatbot with Google Drive & Evaluation

This project is an interactive, multi-user Claude chatbot built using **Gradio**, **LangChain**, **Anthropic Claude**, **FAISS**, and **Google Drive integration** with **automatic response evaluation**. Users can authenticate, load their Drive documents, and query the content via Claude in a responsive chat interface with real-time quality assessment.

---

## Features

- **Multi-user login system** (user ID/email-based)
- **Google Drive integration** – load PDFs and text files from user Drive
- **Claude Haiku** as backend LLM via `langchain_anthropic`
- **Document QA system** with `FAISS` + HuggingFace Embeddings
- **Responsive Gradio-based chatbot UI** with modern design
- **Automatic Response Evaluation** – real-time quality scoring for every response
- **Smart input handling** and minimal reloading for faster responses
- **Stores FAISS index per user** (no need to recreate on every run)
- **Mobile-responsive design** – works perfectly on all devices

---

## Response Evaluation System

The application includes an integrated evaluator that automatically assesses response quality:

### **Evaluation Metrics:**
- **Relevance Score (0-5)** - How well the answer matches the question
- **Completeness Score (0-5)** - How thorough the answer is
- **Overall Score (0-5)** - Average of relevance and completeness

### **Example Output:**
```
Response Quality:
• Relevance: 4/5
• Completeness: 3/5
• Overall Score: 3.5/5
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/kishormaity/google-drive-semantic-search.git
cd google-drive-semantic-search
```

### 2. Install dependencies

Make sure you are using Python 3.9+ and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Setup Environment Variables

#### a. Create a `.env` file:

```bash
cp .env.example .env
```

Add your **Anthropic API key** to `.env`:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
```

#### b. Add `credentials.json`

This is your **Google OAuth Client ID** JSON downloaded from [Google Cloud Console](https://console.cloud.google.com/apis/credentials).

### 4. Run the app

```bash
python web_app.py
```

Open the Gradio interface in your browser and start chatting with Claude!

---

## Responsive Design

The application features a fully responsive design that works on all devices:

- **Desktop** - Full layout with side-by-side input/button
- **Tablet** - Optimized spacing and sizing
- **Mobile** - Stacked layout with touch-friendly controls
- **Small Mobile** - Compact design for very small screens

### **Key Responsive Features:**
- Adaptive chatbot heights for different screen sizes
- Touch-friendly input and buttons
- Proper viewport handling
- iOS compatibility (prevents zoom on input)
- Clean, modern UI with minimal styling

---

## Project Structure

```
PythonProject/
├── web_app.py              # Responsive Gradio UI + session logic
├── main.py                 # Drive loader, vectorstore, query logic
├── evaluator.py            # Response quality evaluation system
├── requirements.txt        # Essential dependencies only
├── README.md              # This file
├── .env                   # Your secrets (never commit this)
├── credentials.json       # Google OAuth credentials
├── tokens/                # User authentication tokens
├── user_data/             # Per-user FAISS indexes
└── vectorstores/          # Vector database storage
```

---

## How It Works

1. **User Authentication**: Enter email ID to log in
2. **Document Loading**: Automatically loads documents from Google Drive
3. **Vector Indexing**: Creates FAISS index for semantic search
4. **Query Processing**: Uses Claude to answer questions about documents
5. **Quality Evaluation**: Automatically evaluates response quality
6. **Responsive Display**: Shows results in clean, mobile-friendly interface

---

## Evaluation System Details

The evaluator uses Claude Haiku to assess response quality:

### **Evaluation Process:**
1. **Query Analysis** - Understands the user's question
2. **Response Assessment** - Evaluates relevance and completeness
3. **Scoring** - Provides detailed quality metrics
4. **Display** - Shows scores in the chat interface

### **Benefits:**
- **Real-time feedback** on response quality
- **Continuous improvement** insights
- **Quality assurance** for document QA
- **Performance monitoring** over time

---

## Environment Variables (.env)

Required environment variables:

```env
ANTHROPIC_API_KEY=your_claude_api_key
```

Google OAuth credentials format:

```json
{
  "installed": {
    "client_id": "your_client_id_here",
    "project_id": "your_project_id_here",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "your_client_secret_here",
    "redirect_uris": ["http://localhost"]
  }
}
```

---

## Usage

1. **Start the application**: `python web_app.py`
2. **Access the UI**: Open http://127.0.0.1:7860
3. **Login**: Enter your email ID
4. **Ask questions**: Type questions about your documents
5. **View evaluation**: See quality scores for each response
6. **Enjoy**: Clean, responsive interface with automatic evaluation

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License. Free to use, modify, and distribute.

---

## Author

Built by [Kishor](https://github.com/kishormaity). Contributions welcome!