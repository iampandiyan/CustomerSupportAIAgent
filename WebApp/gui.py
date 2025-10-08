import streamlit as st
import httpx
import io
import base64
from pydantic import BaseModel
from typing import Optional

# Pydantic models (copy from FastAPI if separate file)
class TranscriptRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

# Streamlit app
st.title("Black Bull Fitness Studio Voice AI Monitor")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "test_session"

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat"):
        st.session_state.messages = []
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format="audio/wav")

# User input
if prompt := st.chat_input("Enter transcript (simulate user speech):"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process via FastAPI (replace with your local URL)
    async def call_api():
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/generate_tts",
                json={"text": prompt, "session_id": st.session_state.session_id},
                timeout=30
            )
            return response
    
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                api_response = asyncio.run(call_api())
                if api_response.status_code == 200:
                    data = api_response.json()
                    llm_text = data["response"] if "response" in data else "Error in LLM"
                    st.markdown(llm_text)
                    
                    # Play audio
                    audio_bytes = io.BytesIO(data["audio_content"])
                    st.audio(audio_bytes, format="audio/wav")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": llm_text,
                        "audio": audio_bytes
                    })
                else:
                    st.error(f"API Error: {api_response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

# Footer
st.markdown("---")
st.info("This GUI simulates voice interactions. For live mic input, run your original script alongside.")
