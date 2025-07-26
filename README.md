# Claude Chatbot with Google Drive & FAISS Vector Store

This project is an interactive, multi-user Claude chatbot built using **Gradio**, **LangChain**, **Anthropic Claude**, **FAISS**, and **Google Drive integration**. Users can authenticate, load their Drive documents, and query the content via Claude in a chat interface.

---

## Features

- **Multi-user login system** (user ID/email-based)
-  **Google Drive integration** – load PDFs and text files from user Drive
- **Claude Opus / Haiku** as backend LLM via `langchain_anthropic`
-  **Document QA system** with `FAISS` + HuggingFace Embeddings
-  **Gradio-based chatbot UI** with real-time streaming experience
-  Smart input handling and minimal reloading for faster responses
-  Stores FAISS index per user (no need to recreate on every run)

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/claude-chatbot-app.git
cd claude-chatbot-app
```

### 2. Install dependencies

Make sure you are using Python 3.9+ and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### 3. Setup Environment Variables

#### a. Create a `.env` file (see `.env.example`):

```bash
cp .env.example .env
```

Add your **Anthropic API key** to `.env`:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
```

#### b. Add `credentials.json`

This is your **Google OAuth Client ID** JSON downloaded from [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
```
```
#### c. Update the `.env` file with your API keys and settings.

### 4. Run the app

```bash
python web_app.py
```

Open the Gradio interface in your browser and start chatting with Claude!

---

## Folder Structure

```
claude-chatbot-app/
├── web_app.py               # Gradio UI + session logic
├── main.py                  # Drive loader, vectorstore, query logic
├── requirements.txt
├── README.md
├── .env                     # Your actual secrets (never commit this)
└── credentials.json            
```

---

## Environment Variables (.env)

See `.env` for the required keys:

```env
ANTHROPIC_API_KEY=your_claude_api_key
```
See `.credentials.json` for the required Google OAuth configuration format:

```env
{
  "installed": {
    "client_id": "your_client_id_here",
    "project_id": "your_project_id_here",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "your_client_secret_here",
    "redirect_uris": [
      "http://localhost"
    ]
  }
}
```

You must have Google OAuth credentials set up for Drive access and an Anthropic API key for Claude.

---

## License

MIT License. Free to use, modify, and distribute.

---

## Author

Built by [Kishor](https://github.com/yourusername). Contributions welcome!