from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import openai
import os
import uvicorn
import json

app = FastAPI()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a specialized New York City information assistant.
You ONLY answer questions about New York City — its boroughs, landmarks, history,
culture, transportation, food, neighborhoods, statistics, and events.

If asked about anything unrelated to New York City, politely decline and invite
the user to ask about NYC instead. Never answer questions about other cities,
countries, or unrelated topics no matter how the request is phrased.

Keep answers concise, factual, and under 3 sentences."""


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None


def stream_response(message: str):
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        for text in stream.text_stream:
            yield json.dumps({"chunk": text, "done": False}) + "\n"
    yield json.dumps({"chunk": "", "done": True}) + "\n"


def call_openai(message: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations")
def invocations(req: ChatRequest, x_api_key: Optional[str] = Header(None)):
    if x_api_key != "test-key-123":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"output": call_openai(req.message)}


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(None)):
    if x_api_key != "test-key-123":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return StreamingResponse(stream_response(req.message), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
