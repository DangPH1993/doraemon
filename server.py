import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests

app = FastAPI()

# Đọc API Key từ biến môi trường trên Render
MASTER_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

class ProxyRequest(BaseModel):
    prompt: str
    chat_history: list = []

@app.post("/api/proxy-chat")
def proxy_chat(req: ProxyRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai token xác thực.")
    
    client_token = authorization.split(" ")[1]

    if not MASTER_API_KEY:
        raise HTTPException(status_code=500, detail="Server chưa nhận được GEMINI_API_KEY từ biến môi trường.")

    # Danh sách ưu tiên lựa chọn giữa phiên bản 3.6 và 2.5
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-2.5-flash"
    ]

    # Định dạng lại lịch sử chat
    contents = []
    for msg in req.chat_history:
        contents.append(msg)
    contents.append({"role": "user", "parts": [{"text": req.prompt}]})

    payload = {
        "contents": contents
    }

    response_data = None
    last_error = ""

    # Tự động quét và chọn phiên bản khả dụng trong 2 bản 3.6 hoặc 2.5
    for model in models_to_try:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={MASTER_API_KEY}"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(gemini_url, json=payload, headers=headers, timeout=20)
            res_data = response.json()
            
            if "error" not in res_data and "candidates" in res_data and len(res_data["candidates"]) > 0:
                response_data = res_data
                break
            else:
                if "error" in res_data:
                    last_error = res_data["error"].get("message", "Unknown error")
        except Exception as e:
            last_error = str(e)
            continue

    if not response_data:
        raise HTTPException(status_code=400, detail=f"Gemini từ chối cả hai phiên bản 3.6 và 2.5. Lỗi: {last_error}")

    try:
        answer = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích dữ liệu trả về: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)