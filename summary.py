import json
import os
from typing import AsyncGenerator, Tuple

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import AzureChatOpenAI

# --- Pydantic Models ---
class SummarizeRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str

# --- ElevenLabs Configuration ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# The voice ID for 'Clara' from ElevenLabs
CLARA_VOICE_ID = "ZIlrSGI4jZqobxRKprJz"
TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{CLARA_VOICE_ID}/stream"

# --- Helper Functions ---
def _sse(event: str, data: dict) -> str:
    """Formats a dictionary into a server-sent event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def _extract_output_text(out: any) -> str:
    """A robust utility to extract the final string output from various LangChain return types."""
    if isinstance(out, str):
        return out
    if hasattr(out, "return_values") and isinstance(out.return_values, dict):
        return (
            out.return_values.get("output")
            or out.return_values.get("final_output")
            or next((v for v in out.return_values.values() if isinstance(v, str)), "")
        ) or ""
    if isinstance(out, dict):
        for k in ("output", "final_output", "answer", "content"):
            v = out.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                for kk in ("output", "content", "text"):
                    vv = v.get(kk)
                    if isinstance(vv, str):
                        return vv
        for v in out.values():
            if isinstance(v, str):
                return v
    return str(out) if out is not None else ""


def create_summary_agent_and_router(llm: AzureChatOpenAI) -> Tuple[AgentExecutor, APIRouter]:
    """
    Creates the summary agent, TTS endpoint, and the FastAPI router.
    Returns the agent executor and the router for use in the main application.
    """
    router = APIRouter()

    # --- Summary Agent Definition ---
    summary_system = """You are Summary Agent, an executive editor.

            Write ONE short paragraph (2-3 sentences max) that captures the essence of the Ask-AI answer.
            Requirements:
            - Analyze any tables qualitatively: direction of change, who/what leads or lags, stability vs. volatility, and whether things look ahead/behind target.
            - Use qualitative wording only (e.g., “rising”, “flat”, “slightly behind target”, “leading”).
            - Do NOT include any numbers (no digits, no % signs, no currency).
            - No lists, headings, no bullet points or preamble—just the paragraph.
            - Keep it very concise and information-dense."""

    summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summary_system),
        ("human",
         "Summarize the following Ask-AI answer in ONE tight paragraph (2-3 sentences). "
         "Be qualitative and trend-focused, and do not include numbers.\n\n"
         "```markdown\n{answer_md}\n```"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

    summary_agent = create_openai_tools_agent(llm=llm, tools=[], prompt=summary_prompt)
    summary_agent_executor = AgentExecutor(
        agent=summary_agent, tools=[], verbose=True, handle_parsing_errors=True, max_iterations=4
    )

    # --- /summarize Endpoint ---
    @router.post("/summarize")
    async def summarize(req: SummarizeRequest, request: Request):
        # ... (your existing /summarize endpoint code remains unchanged)
        wanted = (request.headers.get("accept") or "").lower()
        if "application/json" in wanted:
            try:
                result = await summary_agent_executor.ainvoke({"input": "", "answer_md": req.text})
                md = _extract_output_text(result if not isinstance(result, dict) else result.get("output", result))
                return JSONResponse({"summary": md})
            except Exception as e:
                return JSONResponse({"summary": f"**Failed to summarize:** {str(e)}"}, status_code=500)
        async def event_gen():
            async for chunk in _stream_summary(req.text):
                yield chunk
        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    async def _stream_summary(full_markdown: str) -> AsyncGenerator[bytes, None]:
        """Runs the summary agent and yields SSE events."""
        try:
            async for ev in summary_agent_executor.astream_events(
                {"input": "", "answer_md": full_markdown}, version="v1"
            ):
                if ev["event"] == "on_chain_end":
                    out = ev["data"].get("output", "")
                    final_md = _extract_output_text(out)
                    if not final_md:
                        final_md = "**No summary produced.**"
                    yield _sse("data", {"summary": final_md}).encode("utf-8")
        except Exception as e:
            yield _sse("data", {"summary": f"**Failed to summarize:** {e}"}).encode("utf-8")


    # --- NEW: /tts Endpoint ---
    @router.post("/tts")
    async def text_to_speech(req: TTSRequest):
        """
        Endpoint to convert text to speech using ElevenLabs and stream the audio back.
        """
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=503, detail="ElevenLabs API key not configured.")

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        payload = {
            "text": req.text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        async def audio_stream():
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", TTS_URL, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            error_detail = await response.aread()
                            print(f"ElevenLabs Error: {response.status_code} - {error_detail.decode()}")
                            # Don't yield anything on error to prevent client-side audio issues
                            return

                        async for chunk in response.aiter_bytes():
                            yield chunk
            except httpx.RequestError as e:
                print(f"Error calling ElevenLabs API: {e}")
                # Don't yield anything on error

        return StreamingResponse(audio_stream(), media_type="audio/mpeg")

    return summary_agent_executor, router