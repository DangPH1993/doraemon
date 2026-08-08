from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI()

# Danh sách các Client Token được cấp cho khách hàng
VALID_TOKENS = {
    "DORA_VIP_001": {"client_name": "Nguyễn Văn A", "status": "active"},
    "DORA_VIP_002": {"client_name": "Trần Thị B", "status": "active"}
}

# Hàm xác thực Token
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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Server chưa cấu hình GEMINI_API_KEY!")

        genai.configure(api_key=api_key)
        
        # Sử dụng model gemini-3.6-flash theo yêu cầu
        model = genai.GenerativeModel("gemini-3.6-flash")
        
        chat = model.start_chat(history=data.chat_history)
        response = chat.send_message(data.prompt)

        return {"response": response.text}

    except Exception as e:
        # Nếu gặp lỗi Rate Limit (429), trả về thông báo để phía client hiển thị cho người dùng
        raise HTTPException(status_code=500, detail=f"Lỗi từ server: {str(e)}")

@app.get("/")
def home():
    return {"status": "Doraemon Proxy Server is running with gemini-3.6-flash!"}