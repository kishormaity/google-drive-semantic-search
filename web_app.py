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
        response = query_llm(vectorstore, last_user_message)

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
        .gradio-container {
            max-width: 100vw !important;
            width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: hidden !important;
        }
        .main {
            max-width: 100vw !important;
            width: 100vw !important;
            margin: 0 !important;
            padding: 1rem !important;
            overflow-x: hidden !important;
        }
        @media (max-width: 768px) {
            .main {
                padding: 0.5rem !important;
            }
        }
        .chatbot {
            min-height: 400px !important;
            max-height: calc(100vh - 200px) !important;
            overflow-y: auto !important;
        }
        @media (max-width: 768px) {
            .chatbot {
                min-height: 300px !important;
                max-height: calc(100vh - 150px) !important;
            }
        }
        .gradio-block {
            max-width: 100% !important;
            width: 100% !important;
        }
        .contain {
            max-width: 100% !important;
            width: 100% !important;
        }
    """
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🤖 Claude Document QA Chatbot", elem_classes=["title"])

    initial_message = [{"role": "assistant", "content": "Please enter your email ID to start:"}]
    chat_state = gr.State(initial_message.copy())
    last_user_msg = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="",
                type="messages",
                value=initial_message.copy(),
                height=500,
                elem_classes=["chatbot"]
            )

    with gr.Row():
        with gr.Column(scale=4):
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                container=False,
                scale=4
            )
        with gr.Column(scale=1, min_width=100):
            send_btn = gr.Button(
                "Send",
                variant="primary",
                size="lg",
                scale=1
            )

    # Chain: Display → Respond
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
    width="100%"
)