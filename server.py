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

    # Danh sách các mô hình và phiên bản API ưu tiên để tự động quét phòng trường hợp Google cập nhật
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro"
    ]
    api_versions = ["v1", "v1beta"]

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

    # Vòng lặp thông minh tự động tìm phiên bản và model hoạt động tốt nhất
    for version in api_versions:
        for model in models_to_try:
            gemini_url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={MASTER_API_KEY}"
            headers = {"Content-Type": "application/json"}
            
            try:
                response = requests.post(gemini_url, json=payload, headers=headers, timeout=20)
                res_data = response.json()
                
                # Nếu gọi thành công và có kết quả trả về hợp lệ thì dừng vòng lặp
                if "error" not in res_data and "candidates" in res_data and len(res_data["candidates"]) > 0:
                    response_data = res_data
                    break
                else:
                    if "error" in res_data:
                        last_error = res_data["error"].get("message", "Unknown error")
            except Exception as e:
                last_error = str(e)
                continue
        if response_data:
            break

    if not response_data:
        raise HTTPException(status_code=400, detail=f"Gemini từ chối tất cả cấu hình. Lỗi cuối: {last_error}")

    try:
        answer = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích dữ liệu trả về: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)