import os
import base64
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from pinecone import Pinecone
from google import genai
from google.genai import types

# ============================================================
# DORAEMON RAG SERVER
# PDF -> Gemini Embedding -> Pinecone -> Gemini Answer
# ============================================================

app = FastAPI(title="Doraemon RAG Server")

# -----------------------------
# Environment variables
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX", "doraemon")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "japanese_n5")

# Nếu đặt CLIENT_TOKEN trên Render thì server sẽ kiểm tra token.
# Nếu để trống, server không bắt buộc token.
CLIENT_TOKEN = os.getenv("CLIENT_TOKEN", "")

if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY chưa được cấu hình.")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY chưa được cấu hình.")

# -----------------------------
# Initialize clients
# -----------------------------
pc = None
index = None
gemini = None

try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        print(f"Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"WARNING: Không khởi tạo được Pinecone: {e}")

try:
    if GEMINI_API_KEY:
        gemini = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"WARNING: Không khởi tạo được Gemini: {e}")


# ============================================================
# Request models
# ============================================================

class SearchRequest(BaseModel):
    vector: list[float]
    top_k: int = 3


class ChatRequest(BaseModel):
    prompt: str
    chat_history: list = []
    image_base64: str | None = None

    # Client mới gửi 2 field này để bật RAG.
    use_knowledge_base: bool = True
    knowledge_namespace: str = NAMESPACE

    top_k: int = 5


# ============================================================
# Authentication
# ============================================================

def check_token(authorization: str | None):
    if not CLIENT_TOKEN:
        return

    expected = f"Bearer {CLIENT_TOKEN}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Client Token không hợp lệ."
        )


# ============================================================
# Pinecone search
# ============================================================

def search_pinecone(query: str, top_k: int = 5, namespace: str = NAMESPACE):
    if gemini is None:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình trên Render.")

    if index is None:
        raise RuntimeError("Pinecone chưa được khởi tạo.")

    # Gemini embedding phải khớp dimension 768 của index doraemon.
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    query_vector = response.embeddings[0].values

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    matches = []

    for match in result.matches:
        metadata = match.metadata or {}

        text = metadata.get("text", "")

        if text:
            matches.append({
                "score": float(match.score),
                "text": str(text),
                "course": metadata.get("course", "")
            })

    return matches


# ============================================================
# Existing Pinecone search endpoint
# ============================================================

@app.post("/search")
async def search_documents(
    request: SearchRequest,
    authorization: str | None = Header(default=None)
):
    check_token(authorization)

    try:
        if index is None:
            raise RuntimeError("Pinecone chưa được khởi tạo.")

        result = index.query(
            vector=request.vector,
            top_k=request.top_k,
            include_metadata=True
        )

        return {
            "status": "success",
            "data": result.to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main Doraemon chat endpoint
# ============================================================

@app.post("/api/proxy-chat")
async def proxy_chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None)
):
    check_token(authorization)

    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt không được để trống.")

    if gemini is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY chưa được cấu hình trên Render."
        )

    try:
        # ----------------------------------------------------
        # 1. Search knowledge base
        # ----------------------------------------------------
        knowledge = []

        if request.use_knowledge_base:
            knowledge = search_pinecone(
                query=request.prompt,
                top_k=request.top_k,
                namespace=request.knowledge_namespace
            )

        # ----------------------------------------------------
        # 2. Build RAG context
        # ----------------------------------------------------
        if knowledge:
            context_parts = []

            for i, item in enumerate(knowledge, start=1):
                context_parts.append(
                    f"[Tài liệu {i} | score={item['score']:.4f}]\n"
                    f"{item['text']}"
                )

            knowledge_context = "\n\n".join(context_parts)
        else:
            knowledge_context = "Không tìm thấy đoạn tài liệu phù hợp trong Knowledge Base."

        # ----------------------------------------------------
        # 3. Build conversation history
        # ----------------------------------------------------
        history_text = ""

        if request.chat_history:
            recent_history = request.chat_history[-10:]

            history_parts = []

            for item in recent_history:
                role = item.get("role", "")
                parts = item.get("parts", [])

                texts = []

                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(str(part["text"]))

                if texts:
                    history_parts.append(
                        f"{role}: {' '.join(texts)}"
                    )

            history_text = "\n".join(history_parts)

        # ----------------------------------------------------
        # 4. System/RAG instruction
        # ----------------------------------------------------
        system_instruction = """Bạn là Doraemon, trợ lý AI trên máy tính.

Nhiệm vụ:
- Trả lời người dùng bằng tiếng Việt, trừ khi người dùng yêu cầu ngôn ngữ khác.
- Khi câu hỏi liên quan đến tài liệu trong Knowledge Base, hãy ưu tiên thông tin trong phần TÀI LIỆU THAM KHẢO.
- Không được bịa nội dung tài liệu.
- Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói rõ rằng tài liệu hiện tại không cung cấp đủ thông tin.
- Có thể dùng kiến thức chung để giải thích thêm, nhưng phải phân biệt với thông tin lấy từ tài liệu.
- Trả lời tự nhiên, ngắn gọn và hữu ích.
"""

        final_prompt = f"""{system_instruction}

===== TÀI LIỆU THAM KHẢO TỪ PINECONE =====
{knowledge_context}

===== LỊCH SỬ HỘI THOẠI =====
{history_text if history_text else "(Chưa có lịch sử hội thoại)"}

===== CÂU HỎI HIỆN TẠI =====
{request.prompt}
"""

        # ----------------------------------------------------
        # 5. Prepare Gemini contents
        # ----------------------------------------------------
        contents = [final_prompt]

        if request.image_base64:
            try:
                image_bytes = base64.b64decode(request.image_base64)

                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                )
            except Exception as e:
                print(f"WARNING: Không đọc được image_base64: {e}")

        # ----------------------------------------------------
        # 6. Generate final answer
        # ----------------------------------------------------
        response = gemini.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents
        )

        answer = response.text or "Doraemon không tạo được câu trả lời."

        return {
            "response": answer,
            "rag_used": bool(knowledge),
            "sources": [
                {
                    "score": item["score"],
                    "course": item["course"]
                }
                for item in knowledge
            ]
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"ERROR /api/proxy-chat: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Health check
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "Server đang hoạt động bình thường!",
        "pinecone": index is not None,
        "gemini": gemini is not None,
        "index": INDEX_NAME,
        "namespace": NAMESPACE
    }


# ============================================================
# Local development
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_rag:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
