import os
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Load environment variables
load_dotenv()

def authenticate_drive(user_id="default"):
    print(f"🔐 Authenticating Google Drive for user: {user_id}")
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    token_file = f"tokens/token_{user_id}.json"
    creds = None

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            print("✅ Found existing token.")
        except Exception:
            print("⚠️ Token corrupted. Reauthenticating...")
            os.remove(token_file)
            return authenticate_drive(user_id)

    if not creds or not creds.valid:
        print("🔄 No valid token. Starting authentication flow...")
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        os.makedirs("tokens", exist_ok=True)
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print("✅ New token stored.")

    return creds

def load_documents(user_id="default"):
    print(f"📁 Loading Google Drive documents for user: {user_id}")
    creds = authenticate_drive(user_id)
    loader = GoogleDriveLoader(
        credentials=creds,
        folder_id="root",
        recursive=True
    )
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} documents.")
    return docs

def load_or_create_vectorstore(documents=None, user_id="default", use_existing=False):
    index_path = f"user_data/{user_id}/faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if use_existing and os.path.exists(os.path.join(index_path, "index.faiss")):
        print("📂 Loading existing FAISS index...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if documents is None:
        raise ValueError("❌ No documents provided and use_existing=False")

    print("📦 Creating new FAISS index from documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore

def build_faiss_index(documents, embeddings, index_dir):
    print("🧩 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"🔢 Total chunks: {len(chunks)}")

    print("🧠 Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"✅ FAISS index saved at: {index_dir}")
    return vectorstore

def query_llm(vectorstore, query, evaluate_response=True):
    print(f"\n💬 New query received: {query}")
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    try:
        result = qa.invoke({"query": query})
        print("✅ Claude answered successfully.")
        sources = "\n".join(
            f"🔗 {doc.metadata.get('source') or doc.metadata.get('file_path') or doc.metadata.get('title') or 'Unknown'}"
            for doc in result['source_documents']
        )
        
        response = f"📌 **Answer**:\n{result['result']}\n\n📎 **Sources**:\n{sources}"
        
        # Add evaluation if requested
        if evaluate_response:
            try:
                from evaluator import evaluate_qa_response
                evaluation = evaluate_qa_response(query, result['result'])
                
                eval_info = f"\n\n📊 **Response Quality**:\n"
                eval_info += f"• Relevance: {evaluation.get('relevance_score', 'N/A')}/5\n"
                eval_info += f"• Completeness: {evaluation.get('completeness_score', 'N/A')}/5\n"
                eval_info += f"• Overall Score: {evaluation.get('overall_score', 'N/A')}/5"
                
                response += eval_info
                print(f"✅ Response evaluated - Score: {evaluation.get('overall_score', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                response += "\n\n❌ Evaluation failed"
        
        return response
    except Exception as e:
        print(f"❌ Claude query failed: {e}")
        return f"❌ Error: {str(e)}"
