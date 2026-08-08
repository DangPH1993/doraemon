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
    print(f"Đang xử lý request từ khách hàng token: {client_token}")

    if not MASTER_API_KEY:
        raise HTTPException(status_code=500, detail="Server chưa nhận được GEMINI_API_KEY từ biến môi trường.")

    # Đã đổi từ v1beta sang v1 để tương thích chuẩn với model gemini-1.5-flash
    gemini_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={MASTER_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Định dạng lại lịch sử chat
    contents = []
    for msg in req.chat_history:
        contents.append(msg)
    contents.append({"role": "user", "parts": [{"text": req.prompt}]})

    payload = {
        "contents": contents
    }

    try:
        response = requests.post(gemini_url, json=payload, headers=headers, timeout=30)
        res_data = response.json()
        
        # Kiểm tra lỗi trả về từ Google
        if "error" in res_data:
            err_msg = res_data["error"].get("message", "Lỗi không xác định từ Google")
            raise HTTPException(status_code=400, detail=f"Gemini từ chối: {err_msg}")
        
        if "candidates" not in res_data or len(res_data["candidates"]) == 0:
            raise HTTPException(status_code=500, detail="Google không trả về kết quả.")
            
        answer = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"status": "success", "response": answer}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống proxy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)