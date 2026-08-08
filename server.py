from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests

app = FastAPI()

import os

# Đọc tự động từ biến môi trường trên Render
MASTER_API_KEY = os.environ.get("GEMINI_API_KEY", "")

class ProxyRequest(BaseModel):
    prompt: str
    chat_history: list = []

@app.post("/api/proxy-chat")
def proxy_chat(req: ProxyRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai token xác thực.")
    
    client_token = authorization.split(" ")[1]
    print(f"Đang xử lý request từ khách hàng token: {client_token}")

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": MASTER_API_KEY
    }
    payload = {
        "contents": req.chat_history + [{"role": "user", "parts": [{"text": req.prompt}]}]
    }

    try:
        response = requests.post(gemini_url, json=payload, headers=headers, timeout=30)
        res_data = response.json()
        answer = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi kết nối Gemini API: {str(e)}")