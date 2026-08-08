import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests

app = FastAPI()

# Đọc API Key từ biến môi trường trên Render
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
    
    # Định dạng lại lịch sử chat cho đúng chuẩn API của Gemini
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
        
        # 1. Kiểm tra nếu Google API trả về lỗi
        if "error" in res_data:
            err_msg = res_data["error"].get("message", "Lỗi không xác định từ Google")
            raise HTTPException(status_code=400, detail=f"Gemini từ chối: {err_msg}")
        
        # 2. Kiểm tra an toàn xem có tồn tại candidates không
        if "candidates" not in res_data or len(res_data["candidates"]) == 0:
            raise HTTPException(status_code=500, detail="Google không trả về kết quả (có thể bị chặn nội dung).")
            
        answer = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"status": "success", "response": answer}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống proxy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)