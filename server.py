import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone

# Khởi tạo FastAPI
app = FastAPI(title="Pinecone Search Server")

# Cấu hình Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2JeqCP_CopwHVAcXJcSqQWDT1wNACUwJnrVYMJQA3sMH5f1GXadJh5JdgzDKSw4yympFeC")
INDEX_NAME = "doraemon"

# Khởi tạo kết nối 1 lần khi chạy server
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"❌ Lỗi khởi tạo Pinecone: {e}")

# Định nghĩa dữ liệu đầu vào (Payload)
class SearchRequest(BaseModel):
    vector: list[float]
    top_k: int = 3

@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    API nhận vào một vector và trả về các kết quả gần nhất từ Pinecone.
    """
    try:
        # Truy vấn Pinecone
        result = index.query(
            vector=request.vector,
            top_k=request.top_k,
            include_metadata=True
        )
        # Chuyển đổi object trả về thành dạng dictionary chuẩn JSON
        return {"status": "success", "data": result.to_dict()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "Server đang hoạt động bình thường!"}

if __name__ == "__main__":
    import uvicorn
    # Chạy server ở port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)