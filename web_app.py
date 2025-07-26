import gradio as gr
from main import load_documents, load_or_create_vectorstore, query_llm
from dotenv import load_dotenv
import os

load_dotenv()

sessions = {}
user_state = {"logged_in": False, "user_id": None}

# === Step 1: Display user message and show "..." ===
def display_user_message(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    return chat_history, "", message  # Clear textbox, return last user message

# === Step 2: Generate assistant response ===
def respond_to_user(chat_history, last_user_message):
    if not user_state["logged_in"]:
        user_id = last_user_message.strip()

        try:
            index_path = f"user_data/{user_id}/faiss_index"

            if os.path.exists(os.path.join(index_path, "index.faiss")):
                print("⚡ FAISS exists. Loading index...")
                vectorstore = load_or_create_vectorstore(documents=None, user_id=user_id, use_existing=True)
            else:
                print(f"📁 First time login — authenticating and loading documents for: {user_id}")
                docs = load_documents(user_id)
                vectorstore = load_or_create_vectorstore(documents=docs, user_id=user_id)

            sessions[user_id] = {"vectorstore": vectorstore}
            user_state["user_id"] = user_id
            user_state["logged_in"] = True

            response = f"✅ Logged in as {user_id}. You can now ask questions about your documents."
        except Exception as e:
            response = f"❌ Login failed: {str(e)}"
    else:
        user_id = user_state["user_id"]
        vectorstore = sessions[user_id]["vectorstore"]
        response = query_llm(vectorstore, last_user_message, evaluate_response=True)

    # Replace the "..." with actual response
    if chat_history and chat_history[-1]["content"] == "...":
        chat_history.pop()
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, chat_history

# === Gradio UI ===
with gr.Blocks(
    title="Claude Document QA Chatbot",
    theme=gr.themes.Soft(),
    css="""
        /* Responsive container */
        .gradio-container {
            max-width: 100vw !important;
            width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: hidden !important;
        }
        
        /* Main content area */
        .main {
            max-width: 100vw !important;
            width: 100vw !important;
            margin: 0 !important;
            padding: 1rem !important;
            overflow-x: hidden !important;
        }
        
        /* Container responsive */
        .gradio-container {
            max-width: 100vw !important;
            width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: hidden !important;
        }
        
        /* Ensure proper viewport */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 0 !important;
            }
            
            .main {
                padding: 0.5rem !important;
            }
        }
        
        @media (max-width: 480px) {
            .main {
                padding: 0.25rem !important;
            }
        }
        
        /* Chatbot container - minimal styling */
        .chatbot {
            height: 450px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        
        /* Mobile chatbot */
        @media (max-width: 768px) {
            .chatbot {
                height: 400px !important;
            }
        }
        
        /* Blocks and containers */
        .gradio-block {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        .contain {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        /* Text input - responsive styling */
        .text-input {
            width: 100% !important;
        }
        
        @media (max-width: 768px) {
            .text-input {
                font-size: 16px !important; /* Prevents zoom on iOS */
            }
        }
        
        /* Button - minimal styling */
        .btn-primary {
            /* Default Gradio styling */
        }
        
        /* Title styling */
        .title {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #2c3e50 !important;
            margin-bottom: 0 !important;
            text-align: center !important;
        }
        
        @media (max-width: 768px) {
            .title {
                font-size: 1.5rem !important;
                margin-bottom: 0 !important;
            }
        }
        
        /* Evaluation info styling - removed since not used */
        
        /* Message styling - removed to prevent unwanted styling */
        
        /* Responsive grid */
        .responsive-grid {
            display: grid !important;
            grid-template-columns: 1fr auto !important;
            gap: 1rem !important;
            align-items: end !important;
        }
        
        /* Tablet responsive */
        @media (max-width: 1024px) {
            .responsive-grid {
                gap: 0.75rem !important;
            }
            
            .chatbot {
                height: 400px !important;
            }
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .responsive-grid {
                grid-template-columns: 1fr !important;
                gap: 0.5rem !important;
            }
            
            .chatbot {
                height: 350px !important;
            }
            
            .main {
                padding: 0.5rem !important;
            }
            
            .title {
                font-size: 1.5rem !important;
                margin-bottom: 0 !important;
            }
        }
        
        /* Small mobile responsive */
        @media (max-width: 480px) {
            .chatbot {
                height: 300px !important;
            }
            
            .main {
                padding: 0.25rem !important;
            }
            
            .title {
                font-size: 1.25rem !important;
            }
        }
        
        /* Loading animation */
        .loading {
            display: inline-block !important;
            width: 20px !important;
            height: 20px !important;
            border: 3px solid #f3f3f3 !important;
            border-top: 3px solid #3498db !important;
            border-radius: 50% !important;
            animation: spin 1s linear infinite !important;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Remove extra spacing from rows */
        .gradio-row {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Remove spacing from chatbot messages */
        .message-wrap {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Ensure first message has no top margin */
        .chatbot > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Use default Gradio message styling */
    """
) as demo:
    # Header
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🤖 Claude Document QA Chatbot", elem_classes=["title"])

    initial_message = [{"role": "assistant", "content": "Please enter your email ID to start:"}]
    chat_state = gr.State(initial_message.copy())
    last_user_msg = gr.State("")

    # Chat area
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="",
                type="messages",
                value=initial_message.copy(),
                height=500,
                elem_classes=["chatbot"]
            )

    # Input area with responsive grid
    with gr.Row(elem_classes=["responsive-grid"]):
        with gr.Column(scale=4):
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                container=False,
                elem_classes=["text-input"]
            )
        with gr.Column(scale=1, min_width=100):
            send_btn = gr.Button(
                "Send",
                variant="primary",
                size="lg",
                elem_classes=["btn-primary"]
            )

    # Chain: Display → Respond (evaluation always enabled)
    send_btn.click(display_user_message, inputs=[msg, chat_state], outputs=[chat_state, msg, last_user_msg])\
            .then(respond_to_user, inputs=[chat_state, last_user_msg], outputs=[chat_state, chatbot])

    msg.submit(display_user_message, inputs=[msg, chat_state], outputs=[chat_state, msg, last_user_msg])\
       .then(respond_to_user, inputs=[chat_state, last_user_msg], outputs=[chat_state, chatbot])

    chat_state.change(fn=lambda x: x, inputs=chat_state, outputs=chatbot)

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True,
    height=600,
    width="100%",
    favicon_path=None,
    inbrowser=True
)