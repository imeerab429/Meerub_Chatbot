import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
import torch
import sqlite3
import threading
import re
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Qwen Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("Qwen 2.5 Chatbot 🤖")
st.caption("AI Assistant • Memory • MCP • Stop/Resume • Streaming • Database")


# ============================================================
# DATABASE SETUP
# ============================================================

DB_FILE = "chatbot.db"


def init_database():
    """Create the conversations table if it doesn't already exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_message(role, message):
    """Save a message to SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversations
        (role, message, timestamp)
        VALUES (?, ?, ?)
        """,
        (role, message, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_messages():
    """Load all previous conversation messages."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message
        FROM conversations
        ORDER BY id
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {"role": role, "content": message}
        for role, message in rows
    ]


def clear_database():
    """Delete all conversation messages."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()


# Initialize database
init_database()


# ============================================================
# LOAD QWEN MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model


with st.spinner("🤖 Qwen model is loading..."):
    tokenizer, model = load_model()


# ============================================================
# SYSTEM PROMPT & MCP CONFIGURATION
# ============================================================

BASE_SYSTEM_PROMPT = """
You are a helpful, friendly and intelligent AI assistant.

Rules:
1. Answer clearly and accurately.
2. Be polite and helpful.
3. Remember the conversation context provided to you.
4. If the user asks a calculation question, the application will use the calculator tool.
5. If the user asks for the current time in Pakistan, the application will use the current time tool.
6. If the user asks what you remember, the application will provide stored conversation memory.
7. Do not claim to have real-time access unless the application provides the information.
8. If you do not know something, say so honestly.
"""


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

if "partial_response" not in st.session_state:
    st.session_state.partial_response = ""

if "is_resuming" not in st.session_state:
    st.session_state.is_resuming = False


# ============================================================
# SIDEBAR (Settings, Stop/Resume & Clear)
# ============================================================

with st.sidebar:
    st.header("⚙️ Chatbot Settings")

    st.success("🟢 Qwen Model Loaded")
    st.success("🟢 Memory & MCP Enabled")
    st.success("🟢 Stop/Resume Enabled")
    st.success("🟢 Database Connected")
    st.success("🟢 Streaming Enabled")

    st.divider()

    st.write("### Generation Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🛑 Stop", use_container_width=True):
            st.session_state.stop_generation = True
            st.warning("Generation stopped!")

    with col2:
        if st.button("▶️ Resume", use_container_width=True):
            if st.session_state.partial_response:
                st.session_state.stop_generation = False
                st.session_state.is_resuming = True
                st.rerun()
            else:
                st.info("Nothing to resume.")

    st.divider()

    st.write("### Model")
    st.write("Qwen/Qwen2.5-0.5B-Instruct")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_database()
        st.session_state.messages = []
        st.session_state.stop_generation = False
        st.session_state.partial_response = ""
        st.session_state.is_resuming = False
        st.rerun()


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# TOOLS (Calculator, Time, Memory, MCP)
# ============================================================

def calculator_tool(expression):
    try:
        if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):
            return "Invalid mathematical expression."
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception:
        return "I could not calculate that."


def current_time_tool():
    pakistan_time = datetime.now(ZoneInfo("Asia/Karachi"))
    return pakistan_time.strftime("%I:%M:%S %p, %A, %d %B %Y")


def memory_tool():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM conversations ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    user_messages = [message for role, message in rows if role == "user"]
    return user_messages if user_messages else []


def mcp_context_tool():
    """Model Context Protocol (MCP) data simulation or retrieval."""
    return "[MCP Status: Active session connected, DB synchronized, User local preferences loaded]"


# ============================================================
# WORKFLOW DETECTION
# ============================================================

def detect_workflow(user_input):
    text = user_input.lower().strip()

    math_pattern = r"^[0-9+\-*/().%\s]+$"
    if re.fullmatch(math_pattern, user_input.strip()):
        return "calculator"

    if any(word in text for word in ["calculate", "calculator", "solve"]) and any(symbol in user_input for symbol in ["+", "-", "*", "/", "%"]):
        return "calculator"

    time_phrases = [
        "current time", "what time is it", "what is the time", "what's the time",
        "time in pakistan", "pakistan time", "current time in pakistan"
    ]
    if any(phrase in text for phrase in time_phrases):
        return "time"

    memory_phrases = [
        "what do you remember about me", "what do you remember", "remember about me",
        "my memory", "what do you know about me", "chat history"
    ]
    if any(phrase in text for phrase in memory_phrases):
        return "memory"

    return "chat"


# ============================================================
# STREAMING GENERATION
# ============================================================

def generate_stream(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }

    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()

    for new_text in streamer:
        if st.session_state.get("stop_generation", False):
            break
        yield new_text

    thread.join()


# ============================================================
# CHAT INPUT & EXECUTION
# ============================================================

user_input = st.chat_input("💬 Write your message here...")

if user_input or st.session_state.is_resuming:
    
    if user_input and not st.session_state.is_resuming:
        # Reset flags for new input
        st.session_state.stop_generation = False
        st.session_state.partial_response = ""
        
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)

    # Check Workflow if it's a new input and not resuming
    current_input = user_input if user_input else st.session_state.messages[-1]["content"]
    workflow = detect_workflow(current_input) if user_input else "chat"

    if user_input and workflow != "chat":
        with st.chat_message("assistant"):
            if workflow == "calculator":
                expression = re.sub(r"[^0-9+\-*/().%]", "", user_input)
                result = calculator_tool(expression)
                bot_reply = f"🧮 **Calculation Result:** `{result}`"
            elif workflow == "time":
                current_time = current_time_tool()
                bot_reply = f"🕐 **Current time in Pakistan:** {current_time}"
            elif workflow == "memory":
                memories = memory_tool()
                if memories:
                    bot_reply = "🧠 **What I remember from our conversation:**\n\n"
                    for memory in memories:
                        bot_reply += f"- {memory}\n"
                else:
                    bot_reply = "🧠 I don't have any previous conversation information yet."
            
            st.markdown(bot_reply)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        save_message("assistant", bot_reply)
    
    else:
        # Normal Qwen Chat / Streaming generation block
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            if st.session_state.is_resuming:
                full_response = st.session_state.partial_response
                st.session_state.is_resuming = False
            else:
                full_response = ""
                st.session_state.partial_response = ""

            MAX_CONTEXT_MESSAGES = 10
            recent_messages = st.session_state.messages[-MAX_CONTEXT_MESSAGES:]

            mcp_info = mcp_context_tool()
            dynamic_system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n[MCP Context Data]: {mcp_info}"

            formatted_messages = [
                {
                    "role": "system",
                    "content": dynamic_system_prompt
                }
            ]

            formatted_messages.extend(
                [
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                    for msg in recent_messages
                ]
            )

            if full_response:
                formatted_messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            for token in generate_stream(formatted_messages):
                if st.session_state.stop_generation:
                    full_response += "\n\n*[Stopped by user]*"
                    response_placeholder.markdown(full_response)
                    break

                full_response += token
                st.session_state.partial_response = full_response
                response_placeholder.markdown(full_response + "▌")

            if not st.session_state.stop_generation:
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message("assistant", full_response)
                st.session_state.partial_response = ""