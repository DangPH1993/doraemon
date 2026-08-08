from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI()

# 1. Danh sách các Client Token được cấp cho khách hàng của bạn
# Bạn có thể tự thêm hoặc xóa token của khách tại đây
VALID_TOKENS = {
    "DORA_VIP_001": {"client_name": "Nguyễn Văn A", "status": "active"},
    "DORA_VIP_002": {"client_name": "Trần Thị B", "status": "active"}
}

# 2. Hàm xác thực Token từ phía client gửi lên
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

# 3. Endpoint nhận tin nhắn từ ứng dụng của khách hàng
@app.post("/api/proxy-chat")
async def proxy_chat(data: ChatRequest, client_info: dict = Depends(verify_token)):
    try:
        # Ghi log nhẹ để biết khách hàng nào đang gọi
        print(f"Khách hàng đang tương tác: {client_info['client_name']}")

        # Lấy API Key của Google Gemini từ biến môi trường trên Render
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Server chưa cấu hình GEMINI_API_KEY!")

        genai.configure(api_key=api_key)
        
        # Khởi tạo mô hình Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Gửi kèm lịch sử trò chuyện để AI nhớ ngữ cảnh
        chat = model.start_chat(history=data.chat_history)
        response = chat.send_message(data.prompt)

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi từ server: {str(e)}")

@app.get("/")
def home():
    return {"status": "Doraemon Proxy Server is running!"}