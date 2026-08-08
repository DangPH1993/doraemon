from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI()

VALID_TOKENS = {
    "DORA_VIP_001": {"client_name": "Nguyễn Văn A", "status": "active"},
    "DORA_VIP_002": {"client_name": "Trần Thị B", "status": "active"}
}

def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Thiếu Client Token xác thực!")
    
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Định dạng Token không hợp lệ!")
    
    token = parts[1]
    if token not in VALID_TOKENS or VALID_TOKENS[token]["status"] != "active":
        raise HTTPException(status_code=403, detail="Client Token không tồn tại hoặc đã bị khóa!")
    
    return VALID_TOKENS[token]

class ChatRequest(BaseModel):
    prompt: str
    chat_history: list = []

@app.post("/api/proxy-chat")
async def proxy_chat(data: ChatRequest, client_info: dict = Depends(verify_token)):
    try:
        print(f"Khách hàng đang tương tác: {client_info['client_name']}")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Server chưa cấu hình GEMINI_API_KEY!")

        genai.configure(api_key=api_key)
        
        # Sử dụng đúng tên model theo giao diện của bạn (có thể đổi thành "gemini-3.6-flash" nếu muốn)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        chat = model.start_chat(history=data.chat_history)
        response = chat.send_message(data.prompt)

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi từ server: {str(e)}")

@app.get("/")
def home():
    return {"status": "Doraemon Proxy Server is running!"}