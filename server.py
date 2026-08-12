
import os
import re
import json
import base64
import hashlib
import io
from datetime import datetime, timezone, timedelta
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from pinecone import Pinecone
from google import genai
from google.genai import types
from pypdf import PdfReader

# ============================================================
# Doraemon SaaS Server - COMPLETE
# - User registration/login/subscription
# - Admin user management
# - Admin <-> client realtime chat
# - PDF upload directly from Admin
# - Multi lesson/topic/question/answer metadata per PDF
# - RAG with Gemini Embedding -> Pinecone
# - Learning progress database
# ============================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "doraemon")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_RENDER")
ADMIN_WS_TOKEN = os.getenv("ADMIN_WS_TOKEN", "")
ADMIN_PANEL_PASSWORD = os.getenv("ADMIN_PANEL_PASSWORD") or ADMIN_WS_TOKEN

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
UPLOAD_BATCH_SIZE = 50

app = FastAPI(title="Doraemon SaaS Server")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

pc = None
index = None
gemini = None
connected_users = {}
admin_connections = set()


# ============================================================
# Database
# ============================================================

def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL chưa được cấu hình trên Render.")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    phone VARCHAR(30) UNIQUE NOT NULL,
                    nickname VARCHAR(100) NOT NULL,
                    password_hash TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    plan VARCHAR(100) NOT NULL DEFAULT 'N5',
                    started_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ,
                    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_messages (
                    id BIGSERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    sender VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    is_read BOOLEAN NOT NULL DEFAULT FALSE
                );
            """)

            # New learning-progress table.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS learning_progress (
                    id BIGSERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    subject VARCHAR(200),
                    course VARCHAR(200),
                    lesson VARCHAR(300),
                    topic VARCHAR(300),
                    last_studied_at TIMESTAMPTZ,
                    times_studied INTEGER NOT NULL DEFAULT 0,
                    mastery DOUBLE PRECISION NOT NULL DEFAULT 0,
                    next_review_at TIMESTAMPTZ,
                    notes TEXT
                );
            """)

            # IMPORTANT: existing installations may already have learning_progress
            # without last_studied_at. CREATE TABLE IF NOT EXISTS does NOT migrate
            # an old table, so add missing columns explicitly.
            for sql in [
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS subject VARCHAR(200)",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS course VARCHAR(200)",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS lesson VARCHAR(300)",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS topic VARCHAR(300)",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS last_studied_at TIMESTAMPTZ",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS times_studied INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS mastery DOUBLE PRECISION NOT NULL DEFAULT 0",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS next_review_at TIMESTAMPTZ",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS notes TEXT",
            ]:
                cur.execute(sql)

            # Catalog of uploaded PDF files and their structured learning sections.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id BIGSERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    subject VARCHAR(200) NOT NULL,
                    namespace VARCHAR(200) NOT NULL DEFAULT '__default__',
                    chunk_size INTEGER NOT NULL DEFAULT 1200,
                    overlap INTEGER NOT NULL DEFAULT 200,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_sections (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    lesson VARCHAR(300),
                    lesson_pages VARCHAR(100),
                    topic VARCHAR(300),
                    topic_pages VARCHAR(100),
                    question TEXT,
                    question_pages VARCHAR(100),
                    answer TEXT,
                    answer_pages VARCHAR(100)
                );
            """)

            # The old deployment failed here because the table existed without
            # last_studied_at. Keep index creation AFTER the migration above.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_progress_user
                ON learning_progress(user_id, last_studied_at DESC);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_progress_review
                ON learning_progress(user_id, next_review_at);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_sections_document
                ON knowledge_sections(document_id);
            """)

        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    global pc, index, gemini

    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX)
            print("Pinecone:", PINECONE_INDEX)
        except Exception as e:
            print("WARNING: Pinecone init failed:", e)

    if GEMINI_API_KEY:
        try:
            gemini = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print("WARNING: Gemini init failed:", e)

    if DATABASE_URL:
        init_db()
        print("PostgreSQL: OK")
    else:
        print("WARNING: DATABASE_URL chưa được cấu hình.")

    print("Gemini model:", GEMINI_MODEL)
    print("Embedding:", EMBEDDING_MODEL, EMBEDDING_DIMENSION)


# ============================================================
# Models / auth
# ============================================================

class RegisterRequest(BaseModel):
    phone: str
    nickname: str
    password: str


class LoginRequest(BaseModel):
    phone: str
    password: str


class ChatRequest(BaseModel):
    # "message" is the current client format; "prompt" is kept for compatibility.
    message: str | None = None
    prompt: str | None = None
    chat_history: list = []
    image_base64: str | None = None
    use_knowledge_base: bool = True
    knowledge_namespace: str = PINECONE_NAMESPACE
    top_k: int = 8

    @property
    def text(self):
        return (self.message if self.message is not None else self.prompt or "").strip()


def hash_password(p):
    return pwd_context.hash(p)


def verify_password(p, h):
    return pwd_context.verify(p, h)


def create_token(user_id):
    exp = datetime.now(timezone.utc) + timedelta(days=30)
    return jwt.encode(
        {"sub": str(user_id), "exp": exp, "type": "user"},
        JWT_SECRET,
        algorithm="HS256"
    )


def bearer(authorization):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Thiếu hoặc sai Authorization header.")
    return authorization.split(" ", 1)[1].strip()


def current_user(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        uid = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(401, "Access token không hợp lệ hoặc đã hết hạn.")

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id,phone,nickname,status,created_at FROM users WHERE id=%s",
                (uid,)
            )
            user = cur.fetchone()
    finally:
        conn.close()

    if not user:
        raise HTTPException(401, "Tài khoản không tồn tại.")
    return dict(user)


def subscription_status(user_id):
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id,plan,started_at,expires_at,status
                FROM subscriptions
                WHERE user_id=%s
                ORDER BY id DESC LIMIT 1
            """, (user_id,))
            sub = cur.fetchone()
    finally:
        conn.close()

    if not sub:
        return None, "Tài khoản chưa được Admin kích hoạt."
    if sub["status"] != "ACTIVE":
        return dict(sub), "Tài khoản chưa được Admin kích hoạt."
    if not sub["expires_at"] or sub["expires_at"] <= datetime.now(timezone.utc):
        return dict(sub), "Gói sử dụng đã hết hạn."
    return dict(sub), None


def require_active_user(authorization):
    user = current_user(bearer(authorization))
    sub, msg = subscription_status(user["id"])
    if user["status"] == "LOCKED":
        raise HTTPException(403, "Tài khoản đã bị khóa.")
    if msg:
        raise HTTPException(403, msg)
    return user


# ============================================================
# User auth
# ============================================================

@app.post("/auth/register")
def register(data: RegisterRequest):
    phone = data.phone.strip()
    nickname = data.nickname.strip()
    password = data.password

    if not phone or not nickname or not password:
        raise HTTPException(400, "Vui lòng nhập đầy đủ SĐT, nickname và mật khẩu.")
    if len(password) < 6:
        raise HTTPException(400, "Mật khẩu phải có ít nhất 6 ký tự.")

    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE phone=%s", (phone,))
            if cur.fetchone():
                raise HTTPException(409, "Số điện thoại đã được đăng ký.")

            cur.execute("""
                INSERT INTO users(phone,nickname,password_hash,status)
                VALUES(%s,%s,%s,'PENDING') RETURNING id
            """, (phone, nickname, hash_password(password)))
            uid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    return {
        "success": True,
        "user_id": uid,
        "status": "PENDING",
        "message": "Đăng ký thành công. Tài khoản đang chờ Admin kích hoạt."
    }


@app.post("/auth/login")
def login(data: LoginRequest):
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id,phone,nickname,password_hash,status
                FROM users WHERE phone=%s
            """, (data.phone.strip(),))
            user = cur.fetchone()
    finally:
        conn.close()

    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "SĐT hoặc mật khẩu không đúng.")

    token = create_token(user["id"])
    sub, msg = subscription_status(user["id"])

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {k: user[k] for k in ("id", "phone", "nickname", "status")},
        "subscription": sub,
        "subscription_message": msg
    }


@app.get("/auth/me")
def me(authorization: Optional[str] = Header(default=None)):
    user = current_user(bearer(authorization))
    sub, msg = subscription_status(user["id"])
    return {
        "user": user,
        "subscription": sub,
        "subscription_message": msg
    }


# ============================================================
# Admin <-> client chat
# ============================================================

@app.get("/admin-chat/history")
def history(limit: int = 100, authorization: Optional[str] = Header(default=None)):
    user = current_user(bearer(authorization))
    limit = max(1, min(limit, 500))

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id,sender,message,created_at,is_read
                FROM admin_messages
                WHERE user_id=%s
                ORDER BY id DESC LIMIT %s
            """, (user["id"], limit))
            rows = list(reversed(cur.fetchall()))
    finally:
        conn.close()

    return {"messages": rows}


@app.post("/admin-chat/send")
def user_send_admin(data: dict, authorization: Optional[str] = Header(default=None)):
    user = current_user(bearer(authorization))
    msg = str(data.get("message", "")).strip()

    if not msg:
        raise HTTPException(400, "Tin nhắn trống.")

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO admin_messages(user_id,sender,message)
                VALUES(%s,'user',%s)
                RETURNING id,user_id,sender,message,created_at,is_read
            """, (user["id"], msg))
            row = dict(cur.fetchone())
        conn.commit()
    finally:
        conn.close()

    return {"message": row}


async def send_json(ws, data):
    try:
        await ws.send_json(data)
        return True
    except Exception:
        return False


async def notify_user(uid, data):
    for ws in list(connected_users.get(uid, set())):
        if not await send_json(ws, data):
            connected_users.get(uid, set()).discard(ws)


async def notify_admin(data):
    for ws in list(admin_connections):
        if not await send_json(ws, data):
            admin_connections.discard(ws)


@app.websocket("/ws/user")
async def ws_user(websocket: WebSocket):
    await websocket.accept()

    try:
        user = current_user(websocket.query_params.get("token", ""))
    except HTTPException as e:
        await websocket.send_json({"type": "error", "message": e.detail})
        await websocket.close(code=1008)
        return

    uid = user["id"]
    connected_users.setdefault(uid, set()).add(websocket)

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Đã kết nối chat Admin.",
            "user_id": uid
        })

        while True:
            data = await websocket.receive_json()
            msg = str(data.get("message", "")).strip()
            if not msg:
                continue

            conn = db()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO admin_messages(user_id,sender,message)
                        VALUES(%s,'user',%s)
                        RETURNING id,user_id,sender,message,created_at,is_read
                    """, (uid, msg))
                    row = dict(cur.fetchone())
                conn.commit()
            finally:
                conn.close()

            payload = {"type": "message", "data": row}
            await websocket.send_json(payload)
            await notify_admin(payload)

    except WebSocketDisconnect:
        pass
    finally:
        connected_users.get(uid, set()).discard(websocket)
        if not connected_users.get(uid):
            connected_users.pop(uid, None)


@app.websocket("/ws/admin")
async def ws_admin(websocket: WebSocket):
    await websocket.accept()

    if not ADMIN_WS_TOKEN or websocket.query_params.get("token") != ADMIN_WS_TOKEN:
        await websocket.send_json({
            "type": "error",
            "message": "Admin token không hợp lệ."
        })
        await websocket.close(code=1008)
        return

    admin_connections.add(websocket)

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Admin WebSocket connected."
        })

        while True:
            data = await websocket.receive_json()
            uid = int(data.get("user_id"))
            msg = str(data.get("message", "")).strip()

            if not msg:
                continue

            conn = db()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO admin_messages(user_id,sender,message)
                        VALUES(%s,'admin',%s)
                        RETURNING id,user_id,sender,message,created_at,is_read
                    """, (uid, msg))
                    row = dict(cur.fetchone())
                conn.commit()
            finally:
                conn.close()

            payload = {"type": "message", "data": row}
            await notify_user(uid, payload)
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        pass
    finally:
        admin_connections.discard(websocket)


# ============================================================
# Pinecone / Gemini RAG
# ============================================================

def embed_text(text):
    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")

    r = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )
    return r.embeddings[0].values


def search_pinecone(query, top_k=8, namespace=PINECONE_NAMESPACE):
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")

    result = index.query(
        vector=embed_text(query),
        top_k=max(1, min(int(top_k), 30)),
        namespace=namespace,
        include_metadata=True
    )

    matches = []
    for m in result.matches:
        md = m.metadata or {}
        text = md.get("text", md.get("content", ""))
        if text:
            matches.append({
                "score": float(m.score),
                "text": str(text),
                "metadata": dict(md)
            })
    return matches


@app.post("/search")
def search(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization)
    if not data.text:
        raise HTTPException(400, "Tin nhắn không được để trống.")

    return {
        "matches": search_pinecone(
            data.text,
            data.top_k,
            data.knowledge_namespace
        )
    }


@app.post("/api/proxy-chat")
def proxy_chat(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)

    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    current_text = data.text
    if not current_text:
        raise HTTPException(400, "Tin nhắn không được để trống.")

    # Normalize client history so both the old Gemini-style format and simple
    # {role, text/content/message} formats work without crashing the server.
    history_parts = []
    for item in (data.chat_history or [])[-16:]:
        if isinstance(item, str):
            txt = item.strip()
            if txt:
                history_parts.append(txt)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        texts = []
        if isinstance(item.get("parts"), list):
            for part in item.get("parts"):
                if isinstance(part, dict) and part.get("text"):
                    texts.append(str(part["text"]))
                elif isinstance(part, str) and part.strip():
                    texts.append(part.strip())
        for key in ("text", "content", "message"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        # De-duplicate fields that may contain the same text.
        seen = set()
        texts = [x for x in texts if not (x in seen or seen.add(x))]
        if texts:
            history_parts.append(f"{role}: {' '.join(texts)}")

    # For short follow-up commands such as "hãy kể cho mình", retrieve using
    # the recent conversation as well. This prevents a generic follow-up from
    # losing the story/lesson/topic selected in the previous turn.
    retrieval_parts = history_parts[-6:] + [f"user: {current_text}"]
    retrieval_query = "\n".join(retrieval_parts)
    if len(retrieval_query) > 6000:
        retrieval_query = retrieval_query[-6000:]

    knowledge = []
    rag_error = None
    if data.use_knowledge_base:
        try:
            knowledge = search_pinecone(
                retrieval_query,
                data.top_k,
                data.knowledge_namespace
            )
        except Exception as e:
            # Do not turn a temporary Pinecone/embedding problem into a generic
            # Internal Server Error. Gemini can still answer from the dialogue.
            rag_error = repr(e)
            print("WARNING /api/proxy-chat RAG:", rag_error)

    contexts = []
    for item in knowledge:
        md = item["metadata"]
        contexts.append(
            "[TÀI LIỆU | "
            f"môn={md.get('subject', md.get('course', ''))} | "
            f"bài={md.get('lesson', '')} | "
            f"chủ đề={md.get('topic', '')} | "
            f"câu hỏi={md.get('question', '')} | "
            f"đáp án={md.get('answer', '')} | "
            f"trang={md.get('page', '')}]\n"
            + item["text"]
        )

    learning_context = get_learning_context(user["id"])

    prompt = f"""
Bạn là Doraemon, gia sư tiếng Nhật cá nhân của người học.

NGUYÊN TẮC QUAN TRỌNG NHẤT:
1. Hãy hiểu câu hiện tại trong NGỮ CẢNH của các lượt nói trước.
2. Nếu người học đã yêu cầu một hành động cụ thể thì THỰC HIỆN NGAY, không hỏi lại một câu hỏi lựa chọn đã được trả lời.
3. Ví dụ: nếu trước đó Doraemon nói có truyện "Cô bé quàng khăn đỏ" và người học nói "hãy kể cho mình", phải bắt đầu kể truyện ngay.
4. Nếu người học nói "kể tiếp đi", "tiếp tục", "đọc tiếp", hãy tiếp tục đúng nội dung đang học.
5. Chỉ hỏi "Bạn muốn học gì?" hoặc đưa menu lựa chọn khi người học thực sự chưa có mục tiêu học tập rõ ràng.
6. Không được hỏi lại thông tin mà người học vừa cung cấp.
7. Khi có tài liệu Knowledge Base liên quan, ưu tiên và bám sát tài liệu; không bịa nội dung của tài liệu.
8. Nếu người học yêu cầu kể/đọc một truyện có trong tài liệu, hãy kể/đọc nội dung ngay. Có thể chia thành từng đoạn nếu truyện dài.
9. Nếu tài liệu không chứa đủ nội dung để thực hiện yêu cầu, hãy nói rõ phần nào có thể làm được thay vì quay lại menu học.

MỤC TIÊU:
- Dạy tiếng Nhật giao tiếp hiệu quả, tự nhiên và theo lộ trình.
- Ưu tiên giáo trình trong Knowledge Base.
- Khi có metadata bài học/chủ đề/câu hỏi/đáp án, sử dụng chúng để tổ chức bài học.
- Có thể giải thích bằng tiếng Việt.
- Từ vựng: nghĩa, cách dùng, ví dụ.
- Ngữ pháp: cấu trúc, sắc thái, ví dụ.
- Giao tiếp: hội thoại thực tế và luyện phản xạ.
- Truyện: đọc hiểu, từ vựng và mẫu câu.
- Chia nội dung thành các bước nhỏ, tránh dạy quá nhiều cùng lúc.
- Sau khi dạy một phần, có thể đặt 1-3 câu kiểm tra nếu phù hợp.
- Nếu người học đang ôn lại phần đã học, ưu tiên ôn tập trước khi mở rộng.

TIẾN ĐỘ NGƯỜI HỌC:
{learning_context}

TÀI LIỆU THAM KHẢO:
{chr(10).join(contexts) if contexts else "Không tìm thấy tài liệu phù hợp cho truy vấn hiện tại."}

LỊCH SỬ HỘI THOẠI GẦN NHẤT:
{chr(10).join(history_parts) if history_parts else "(Chưa có)"}

CÂU HỎI HIỆN TẠI:
{current_text}
"""

    contents = [prompt]

    if data.image_base64:
        try:
            image_bytes = base64.b64decode(data.image_base64)
            contents.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ))
        except Exception as e:
            print("WARNING: image_base64:", e)

    try:
        response = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.3)
        )
    except Exception as e:
        print("ERROR /api/proxy-chat:", repr(e))
        raise HTTPException(500, f"Lỗi Gemini: {e}")

    answer = response.text or "Doraemon không tạo được câu trả lời."

    return {
        "reply": answer,
        "response": answer,
        "model": GEMINI_MODEL,
        "rag_used": bool(knowledge),
        "rag_warning": rag_error,
        "sources": [
            {
                "score": x["score"],
                "subject": x["metadata"].get("subject", x["metadata"].get("course", "")),
                "lesson": x["metadata"].get("lesson", ""),
                "topic": x["metadata"].get("topic", ""),
                "question": x["metadata"].get("question", ""),
                "answer": x["metadata"].get("answer", ""),
                "page": x["metadata"].get("page", "")
            }
            for x in knowledge
        ]
    }


# ============================================================
# Learning progress / catalog
# ============================================================

def get_learning_context(user_id):
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT subject,course,lesson,topic,last_studied_at,
                       times_studied,mastery,next_review_at,notes
                FROM learning_progress
                WHERE user_id=%s
                ORDER BY COALESCE(last_studied_at, '1970-01-01'::timestamptz) DESC
                LIMIT 20
            """, (user_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return "Chưa có lịch sử học tập được lưu."

    lines = []
    for r in rows:
        lines.append(
            f"- {r.get('subject') or r.get('course') or ''} | "
            f"Bài: {r.get('lesson') or '-'} | "
            f"Chủ đề: {r.get('topic') or '-'} | "
            f"Đã học: {r.get('times_studied') or 0} lần | "
            f"Mức độ: {r.get('mastery') or 0}"
        )
    return "\n".join(lines)


@app.get("/api/learning/catalog")
def learning_catalog(authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization)

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT d.id,d.file_name,d.subject,d.namespace,
                       s.id section_id,s.lesson,s.lesson_pages,
                       s.topic,s.topic_pages,s.question,s.question_pages,
                       s.answer,s.answer_pages
                FROM knowledge_documents d
                LEFT JOIN knowledge_sections s ON s.document_id=d.id
                ORDER BY d.id DESC,s.id ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    documents = {}
    for r in rows:
        did = r["id"]
        if did not in documents:
            documents[did] = {
                "id": did,
                "file_name": r["file_name"],
                "subject": r["subject"],
                "namespace": r["namespace"],
                "sections": []
            }

        if r["section_id"] is not None:
            documents[did]["sections"].append({
                "id": r["section_id"],
                "lesson": r["lesson"],
                "lesson_pages": r["lesson_pages"],
                "topic": r["topic"],
                "topic_pages": r["topic_pages"],
                "question": r["question"],
                "question_pages": r["question_pages"],
                "answer": r["answer"],
                "answer_pages": r["answer_pages"]
            })

    return {"documents": list(documents.values())}


@app.post("/api/learning/progress")
def save_learning_progress(data: dict, authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)

    subject = str(data.get("subject", "")).strip()
    course = str(data.get("course", "")).strip()
    lesson = str(data.get("lesson", "")).strip()
    topic = str(data.get("topic", "")).strip()
    notes = str(data.get("notes", "")).strip()
    mastery = float(data.get("mastery", 0) or 0)
    mastery = max(0, min(100, mastery))

    if not subject and not course:
        raise HTTPException(400, "Thiếu môn học.")

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Reuse an existing progress row when subject/course + lesson/topic match.
            cur.execute("""
                SELECT id,times_studied
                FROM learning_progress
                WHERE user_id=%s
                  AND COALESCE(subject,course,'')=%s
                  AND COALESCE(lesson,'')=%s
                  AND COALESCE(topic,'')=%s
                ORDER BY id DESC LIMIT 1
            """, (
                user["id"],
                subject or course,
                lesson,
                topic
            ))
            old = cur.fetchone()

            now = datetime.now(timezone.utc)
            if old:
                cur.execute("""
                    UPDATE learning_progress
                    SET subject=%s,course=%s,lesson=%s,topic=%s,
                        last_studied_at=%s,times_studied=%s,
                        mastery=%s,notes=%s
                    WHERE id=%s
                    RETURNING id
                """, (
                    subject or course, course or subject, lesson, topic,
                    now, int(old["times_studied"] or 0) + 1,
                    mastery, notes, old["id"]
                ))
            else:
                cur.execute("""
                    INSERT INTO learning_progress(
                        user_id,subject,course,lesson,topic,
                        last_studied_at,times_studied,mastery,notes
                    )
                    VALUES(%s,%s,%s,%s,%s,%s,1,%s,%s)
                    RETURNING id
                """, (
                    user["id"], subject or course, course or subject,
                    lesson, topic, now, mastery, notes
                ))
                cur.fetchone()

        conn.commit()
    finally:
        conn.close()

    return {"success": True}


@app.get("/api/learning/progress")
def read_learning_progress(authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id,subject,course,lesson,topic,last_studied_at,
                       times_studied,mastery,next_review_at,notes
                FROM learning_progress
                WHERE user_id=%s
                ORDER BY COALESCE(last_studied_at, '1970-01-01'::timestamptz) DESC
            """, (user["id"],))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    return {"progress": rows}


# ============================================================
# Admin authentication
# ============================================================

def check_admin(password: str):
    expected = ADMIN_PANEL_PASSWORD
    if not expected or password != expected:
        raise HTTPException(401, "Admin password không đúng.")


def parse_pages(value):
    """
    Accept:
      1
      1-10
      1,2,5-8
      1;2;5-8
    Returns a set of page numbers.
    """
    if value is None:
        return set()

    s = str(value).strip()
    if not s:
        return set()

    result = set()
    for part in re.split(r"[,;]+", s):
        part = part.strip()
        if not part:
            continue

        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            result.update(range(a, b + 1))
            continue

        if part.isdigit():
            result.add(int(part))

    return result


def clean_text(text):
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text, chunk_size, overlap):
    text = clean_text(text)
    if not text:
        return []

    chunk_size = max(100, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    step = chunk_size - overlap

    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def make_vector_id(source_name, chunk_index):
    stem = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        os.path.splitext(os.path.basename(source_name))[0]
    )[:60].strip("_")
    digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    return f"doraemon_{stem}_{digest}_{chunk_index}"


def parse_metadata_rows(metadata_json):
    try:
        rows = json.loads(metadata_json or "[]")
    except Exception:
        raise HTTPException(400, "metadata_json không phải JSON hợp lệ.")

    if not isinstance(rows, list):
        raise HTTPException(400, "metadata_json phải là một mảng.")

    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        item = {
            "lesson": str(row.get("lesson", "") or "").strip(),
            "lesson_pages": str(row.get("lesson_pages", "") or "").strip(),
            "topic": str(row.get("topic", "") or "").strip(),
            "topic_pages": str(row.get("topic_pages", "") or "").strip(),
            "question": str(row.get("question", "") or "").strip(),
            "question_pages": str(row.get("question_pages", "") or "").strip(),
            "answer": str(row.get("answer", "") or "").strip(),
            "answer_pages": str(row.get("answer_pages", "") or "").strip(),
        }

        # Keep rows that have at least one meaningful field.
        if any(item.values()):
            clean_rows.append(item)

    return clean_rows


def metadata_for_page(page_no, rows):
    """
    A chunk can belong to multiple configured rows when ranges overlap.
    Store matching values as ' | '-separated metadata.
    """
    fields = {
        "lesson": [],
        "topic": [],
        "question": [],
        "answer": [],
        "lesson_pages": [],
        "topic_pages": [],
        "question_pages": [],
        "answer_pages": [],
    }

    for row in rows:
        # A row is considered matching if ANY configured page range contains this page.
        matches = False
        for key in ("lesson_pages", "topic_pages", "question_pages", "answer_pages"):
            if page_no in parse_pages(row.get(key, "")):
                matches = True
                break

        # If the row has no page range at all, do not attach it to every chunk.
        if not matches:
            continue

        for key in fields:
            val = row.get(key, "")
            if val and val not in fields[key]:
                fields[key].append(val)

    return {k: " | ".join(v) for k, v in fields.items()}


@app.post("/admin/api/upload-pdf")
async def admin_upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(...),
    subject: str = Form(...),
    metadata_json: str = Form("[]"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    overlap: int = Form(DEFAULT_OVERLAP),
    namespace: str = Form("")
):
    """
    Upload PDF directly from Admin.

    One PDF has exactly ONE mandatory subject.
    metadata_json can contain many lesson/topic/question/answer rows.
    """
    check_admin(password)

    subject = subject.strip()
    if not subject:
        raise HTTPException(400, "Môn học là bắt buộc.")

    if not file.filename:
        raise HTTPException(400, "Chưa chọn file PDF.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Chỉ hỗ trợ file PDF.")

    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")

    namespace = (namespace or PINECONE_NAMESPACE).strip() or PINECONE_NAMESPACE
    chunk_size = max(100, min(int(chunk_size), 10000))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    rows = parse_metadata_rows(metadata_json)

    try:
        content = await file.read()
        if not content:
            raise ValueError("File PDF rỗng.")
        pdf_stream = io.BytesIO(content)
        pdf_stream.seek(0)
        reader = PdfReader(pdf_stream, strict=False)
    except Exception as e:
        raise HTTPException(400, f"Không đọc được PDF: {e}")

    all_chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        chunks = split_text(text, chunk_size, overlap)
        page_meta = metadata_for_page(page_no, rows)

        for local_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page": page_no,
                "local_chunk": local_idx,
                "meta": page_meta
            })

    if not all_chunks:
        raise HTTPException(
            400,
            "PDF không có text để embedding. Nếu PDF là bản scan, cần OCR trước."
        )

    # Save catalog/document metadata BEFORE Pinecone upsert so the admin
    # can later use the catalog for lesson selection.
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO knowledge_documents(
                    file_name,subject,namespace,chunk_size,overlap,
                    total_pages,total_chunks
                )
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                file.filename, subject, namespace, chunk_size, overlap,
                len(reader.pages), len(all_chunks)
            ))
            document_id = cur.fetchone()["id"]

            for row in rows:
                cur.execute("""
                    INSERT INTO knowledge_sections(
                        document_id,lesson,lesson_pages,topic,topic_pages,
                        question,question_pages,answer,answer_pages
                    )
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    document_id,
                    row["lesson"], row["lesson_pages"],
                    row["topic"], row["topic_pages"],
                    row["question"], row["question_pages"],
                    row["answer"], row["answer_pages"]
                ))

        conn.commit()
    finally:
        conn.close()

    vectors = []
    uploaded = 0

    try:
        for idx_no, item in enumerate(all_chunks):
            response = gemini.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=item["text"],
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )
            values = response.embeddings[0].values

            md = {
                "text": item["text"],
                "subject": subject,
                "course": subject,  # compatibility with old vectors
                "source_file": file.filename,
                "page": item["page"],
                "chunk_index": idx_no,
                "local_chunk": item["local_chunk"],
            }
            md.update(item["meta"])

            vectors.append({
                "id": make_vector_id(file.filename, idx_no),
                "values": values,
                "metadata": md
            })

            if len(vectors) >= UPLOAD_BATCH_SIZE or idx_no == len(all_chunks) - 1:
                index.upsert(vectors=vectors, namespace=namespace)
                uploaded += len(vectors)
                vectors.clear()

    except Exception as e:
        # The catalog is retained for audit/debugging. The response tells the
        # admin that Pinecone upload failed.
        raise HTTPException(500, f"Lỗi embedding/Pinecone: {e}")

    return {
        "success": True,
        "message": "Upload PDF thành công.",
        "file_name": file.filename,
        "document_id": document_id,
        "subject": subject,
        "namespace": namespace,
        "pages": len(reader.pages),
        "chunks": len(all_chunks),
        "vectors_uploaded": uploaded,
        "metadata_rows": len(rows)
    }


# ============================================================
# Admin API
# ============================================================

@app.get("/admin/api/users")
def admin_users(password: str):
    check_admin(password)

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT u.id,u.phone,u.nickname,u.status,u.created_at,
                       s.id subscription_id,s.plan,s.started_at,
                       s.expires_at,s.status subscription_status
                FROM users u
                LEFT JOIN LATERAL (
                    SELECT * FROM subscriptions
                    WHERE user_id=u.id
                    ORDER BY id DESC LIMIT 1
                ) s ON TRUE
                ORDER BY u.id DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "users": [
            {
                "id": r["id"],
                "phone": r["phone"],
                "nickname": r["nickname"],
                "status": r["status"],
                "created_at": r["created_at"],
                "subscription": None if r["subscription_id"] is None else {
                    "id": r["subscription_id"],
                    "plan": r["plan"],
                    "started_at": r["started_at"],
                    "expires_at": r["expires_at"],
                    "status": r["subscription_status"]
                }
            }
            for r in rows
        ]
    }


@app.post("/admin/api/users/{user_id}/activate")
def admin_activate(user_id: int, data: dict):
    check_admin(str(data.get("password", "")))
    months = int(data.get("months", 1))

    if months not in (1, 3, 6, 12):
        raise HTTPException(400, "Thời hạn phải 1, 3, 6 hoặc 12 tháng.")

    plan = str(data.get("plan", "N5")).strip() or "N5"

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE id=%s", (user_id,))
            if not cur.fetchone():
                raise HTTPException(404, "Không tìm thấy user.")

            cur.execute("""
                SELECT id,expires_at
                FROM subscriptions
                WHERE user_id=%s
                ORDER BY id DESC LIMIT 1
            """, (user_id,))
            old = cur.fetchone()

            now = datetime.now(timezone.utc)
            start = (
                old["expires_at"]
                if old and old["expires_at"] and old["expires_at"] > now
                else now
            )
            exp = start + timedelta(days=30 * months)

            if old:
                cur.execute("""
                    UPDATE subscriptions
                    SET plan=%s,
                        started_at=COALESCE(started_at,%s),
                        expires_at=%s,
                        status='ACTIVE'
                    WHERE id=%s
                """, (plan, now, exp, old["id"]))
            else:
                cur.execute("""
                    INSERT INTO subscriptions(
                        user_id,plan,started_at,expires_at,status
                    )
                    VALUES(%s,%s,%s,%s,'ACTIVE')
                """, (user_id, plan, now, exp))

            cur.execute(
                "UPDATE users SET status='ACTIVE' WHERE id=%s",
                (user_id,)
            )

        conn.commit()
    finally:
        conn.close()

    return {"success": True, "expires_at": exp}


@app.post("/admin/api/users/{user_id}/status")
def admin_status(user_id: int, data: dict):
    check_admin(str(data.get("password", "")))
    status = str(data.get("status", "")).upper()

    if status not in ("ACTIVE", "LOCKED", "PENDING"):
        raise HTTPException(400, "Trạng thái không hợp lệ.")

    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET status=%s WHERE id=%s",
                (status, user_id)
            )
            if cur.rowcount == 0:
                raise HTTPException(404, "Không tìm thấy user.")
        conn.commit()
    finally:
        conn.close()

    return {"success": True, "status": status}


@app.post("/admin/api/chat/send")
async def admin_send_chat(data: dict):
    check_admin(str(data.get("password", "")))

    user_id = int(data.get("user_id"))
    msg = str(data.get("message", "")).strip()

    if not msg:
        raise HTTPException(400, "Tin nhắn trống.")

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE id=%s", (user_id,))
            if not cur.fetchone():
                raise HTTPException(404, "Không tìm thấy user.")

            cur.execute("""
                INSERT INTO admin_messages(user_id,sender,message)
                VALUES(%s,'admin',%s)
                RETURNING id,user_id,sender,message,created_at,is_read
            """, (user_id, msg))
            row = dict(cur.fetchone())
        conn.commit()
    finally:
        conn.close()

    payload = {"type": "message", "data": row}
    await notify_user(user_id, payload)

    return {"message": row}


@app.get("/admin/api/chat/history")
def admin_chat_history(
    user_id: int,
    password: str,
    limit: int = 200,
    after_id: int = 0
):
    check_admin(password)

    limit = max(1, min(limit, 500))
    after_id = max(0, int(after_id or 0))

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if after_id > 0:
                cur.execute("""
                    SELECT id,user_id,sender,message,created_at,is_read
                    FROM admin_messages
                    WHERE user_id=%s AND id>%s
                    ORDER BY id ASC LIMIT %s
                """, (user_id, after_id, limit))
            else:
                cur.execute("""
                    SELECT id,user_id,sender,message,created_at,is_read
                    FROM admin_messages
                    WHERE user_id=%s
                    ORDER BY id ASC LIMIT %s
                """, (user_id, limit))

            rows = [dict(r) for r in cur.fetchall()]

            if rows:
                cur.execute("""
                    UPDATE admin_messages
                    SET is_read=TRUE
                    WHERE user_id=%s AND sender='user' AND id<=%s
                """, (user_id, rows[-1]["id"]))

        conn.commit()
    finally:
        conn.close()

    return {
        "messages": rows,
        "last_id": rows[-1]["id"] if rows else after_id
    }


@app.get("/admin/api/ws-token")
def admin_ws_token(password: str):
    check_admin(password)

    if not ADMIN_WS_TOKEN:
        raise HTTPException(
            500,
            "ADMIN_WS_TOKEN chưa được cấu hình trên Render."
        )

    return {"token": ADMIN_WS_TOKEN}


# ============================================================
# Admin HTML
# ============================================================

@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    return HTMLResponse(r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Doraemon Admin</title>
<style>
*{box-sizing:border-box}
body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#222}
header{background:#1677ff;color:#fff;padding:18px 24px;font-size:22px;font-weight:700}
main{max-width:1280px;margin:20px auto;padding:0 15px}
.card{background:#fff;padding:18px;border-radius:12px;margin-bottom:18px;box-shadow:0 2px 10px #0001}
input,button,textarea{padding:9px;border-radius:7px;border:1px solid #ccc}
button{background:#1677ff;color:#fff;border:0;cursor:pointer}
button.gray{background:#666}
button.red{background:#d93025}
button.green{background:#16803c}
button:disabled{opacity:.5;cursor:not-allowed}
#login{max-width:420px;margin:60px auto}
.layout{display:grid;grid-template-columns:44% 56%;gap:18px}
.user{padding:10px;border-bottom:1px solid #eee;cursor:pointer}
.user:hover{background:#f5f8ff}
.user.sel{background:#e8f1ff}
.status-ACTIVE{color:#16803c}
.status-PENDING{color:#b76b00}
.status-LOCKED{color:#c00}
#users{max-height:610px;overflow:auto}
.chat{display:flex;flex-direction:column;height:610px}
#messages{flex:1;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}
.msg{margin:7px 0;padding:8px 10px;border-radius:10px;max-width:82%;white-space:pre-wrap}
.msg.user{background:#dff0ff;margin-right:auto}
.msg.admin{background:#dff7df;margin-left:auto}
.meta{font-size:11px;color:#777;margin-top:3px}
.chatbar{display:flex;gap:7px;margin-top:10px}
.chatbar input{flex:1}
.small{font-size:13px;color:#666}
.upload-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}
.upload-grid input{width:100%}
.section-row{display:grid;grid-template-columns:1.1fr .7fr 1.1fr .7fr 1.2fr .7fr 1.2fr .7fr;gap:6px;margin-bottom:7px}
.section-row input{min-width:0;width:100%}
.help{font-size:12px;color:#666;margin:7px 0 12px}
#uploadStatus{margin-top:10px;white-space:pre-wrap}
.catalog-row{padding:8px 0;border-bottom:1px solid #eee}
@media(max-width:1000px){
 .layout{grid-template-columns:1fr}
 .section-row{grid-template-columns:1fr 1fr}
 .upload-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<header>🤖 Doraemon Admin</header>
<main>

<div class="card" id="login">
  <h3>Đăng nhập Admin</h3>
  <input id="pw" type="password" placeholder="Mật khẩu Admin" style="width:70%"
         onkeydown="if(event.key==='Enter')login()">
  <button onclick="login()">Đăng nhập</button>
  <div id="err" style="color:#c00;margin-top:8px"></div>
</div>

<div id="panel" style="display:none">

  <div class="card">
    <h3>📚 Knowledge Base</h3>
    <div class="small">
      Upload PDF trực tiếp lên Pinecone · Gemini Embedding 768 ·
      Namespace: <b id="namespaceLabel"></b>
    </div>
    <br>

    <input id="pdfFile" type="file" accept=".pdf,application/pdf"
           style="width:100%;padding:10px">

    <div class="upload-grid" style="margin-top:8px">
      <input id="subject" placeholder="Môn học * (bắt buộc)">
      <input id="namespace" placeholder="Namespace (để trống = mặc định)">
      <input id="chunkSize" type="number" value="1200" min="100"
             placeholder="Chunk size">
      <input id="overlap" type="number" value="200" min="0"
             placeholder="Overlap">
    </div>

    <h4>📚 Cấu hình nội dung trong PDF</h4>
    <div class="help">
      Một PDF chỉ chọn <b>1 Môn học</b>. Bên dưới có thể tạo nhiều dòng
      để mô tả nhiều Bài học / Chủ đề / Câu hỏi / Đáp án.
      Trang có thể nhập: <b>1</b>, <b>1-10</b>, hoặc <b>1,3,5-8</b>.
      Tất cả các trường ngoại trừ Môn học đều optional.
    </div>

    <div style="overflow:auto">
      <div id="sectionRows"></div>
    </div>

    <button class="gray" onclick="addSectionRow()">＋ Thêm bài/chủ đề</button>
    <button onclick="uploadPdf()">⬆ Upload PDF</button>
    <div id="uploadStatus"></div>
  </div>

  <div class="card">
    <button onclick="loadUsers()">🔄 Làm mới</button>
    <span id="count" class="small"></span>
    <span id="wsState" class="small"
          style="float:right;color:green">● Chat đang đồng bộ tự động</span>
  </div>

  <div class="layout">
    <div class="card">
      <h3>👥 Tài khoản</h3>
      <div id="users"></div>
    </div>

    <div class="card chat">
      <h3 id="chatTitle">💬 Chọn một khách hàng để chat</h3>
      <div id="messages"></div>
      <div class="chatbar">
        <input id="chatInput" placeholder="Nhập tin nhắn..." disabled
          onkeydown="if(event.key==='Enter')sendAdminMessage()">
        <button id="sendBtn" onclick="sendAdminMessage()" disabled>Gửi</button>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>📖 Nội dung đã upload</h3>
    <div class="small">
      Danh mục này giúp Doraemon biết có những môn, bài và chủ đề nào để
      xây dựng lựa chọn học tập.
    </div>
    <button style="margin-top:8px" onclick="loadCatalog()">🔄 Làm mới danh mục</button>
    <div id="catalog" style="margin-top:10px"></div>
  </div>

</div>
</main>

<script>
let pw="";
let ws=null;
let wsToken="";
let selectedUser=null;
let seenMessageIds=new Set();
let pollTimer=null;
let pollBusy=false;
let lastChatId=0;

function esc(x){
  return String(x??"").replace(/[&<>"']/g,m=>({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[m]));
}

async function api(u,o={}){
  o.headers={"Content-Type":"application/json",...(o.headers||{})};
  const r=await fetch(u,o);
  const t=await r.text();
  let d={};
  try{d=JSON.parse(t)}catch{d={detail:t}}
  if(!r.ok)throw Error(
    typeof d.detail==="string"?d.detail:JSON.stringify(d.detail||d)
  );
  return d;
}

async function login(){
  pw=document.getElementById("pw").value;
  document.getElementById("err").textContent="";
  try{
    await api("/admin/api/users?password="+encodeURIComponent(pw));
    document.getElementById("login").style.display="none";
    document.getElementById("panel").style.display="block";
    document.getElementById("namespaceLabel").textContent =
      "__default__";
    addSectionRow();
    await loadUsers();
    await loadCatalog();
    startChatPolling();
    getWsToken();
  }catch(e){
    document.getElementById("err").textContent=e.message;
  }
}

function addSectionRow(v={}){
  const row=document.createElement("div");
  row.className="section-row";
  row.innerHTML=`
    <input class="lesson" placeholder="Bài học">
    <input class="lesson_pages" placeholder="Trang bài học">
    <input class="topic" placeholder="Chủ đề">
    <input class="topic_pages" placeholder="Trang chủ đề">
    <input class="question" placeholder="Câu hỏi">
    <input class="question_pages" placeholder="Trang câu hỏi">
    <input class="answer" placeholder="Đáp án">
    <input class="answer_pages" placeholder="Trang đáp án">
  `;
  row.querySelector(".lesson").value=v.lesson||"";
  row.querySelector(".lesson_pages").value=v.lesson_pages||"";
  row.querySelector(".topic").value=v.topic||"";
  row.querySelector(".topic_pages").value=v.topic_pages||"";
  row.querySelector(".question").value=v.question||"";
  row.querySelector(".question_pages").value=v.question_pages||"";
  row.querySelector(".answer").value=v.answer||"";
  row.querySelector(".answer_pages").value=v.answer_pages||"";
  document.getElementById("sectionRows").appendChild(row);
}

function collectSectionRows(){
  return [...document.querySelectorAll(".section-row")].map(row=>({
    lesson:row.querySelector(".lesson").value.trim(),
    lesson_pages:row.querySelector(".lesson_pages").value.trim(),
    topic:row.querySelector(".topic").value.trim(),
    topic_pages:row.querySelector(".topic_pages").value.trim(),
    question:row.querySelector(".question").value.trim(),
    question_pages:row.querySelector(".question_pages").value.trim(),
    answer:row.querySelector(".answer").value.trim(),
    answer_pages:row.querySelector(".answer_pages").value.trim()
  })).filter(x=>Object.values(x).some(Boolean));
}

async function uploadPdf(){
  const f=document.getElementById("pdfFile").files[0];
  const subject=document.getElementById("subject").value.trim();

  if(!f){alert("Hãy chọn file PDF.");return;}
  if(!subject){alert("Môn học là bắt buộc.");return;}

  const fd=new FormData();
  fd.append("file",f);
  fd.append("password",pw);
  fd.append("subject",subject);
  fd.append("namespace",
    document.getElementById("namespace").value.trim());
  fd.append("chunk_size",
    document.getElementById("chunkSize").value||"1200");
  fd.append("overlap",
    document.getElementById("overlap").value||"200");
  fd.append("metadata_json",JSON.stringify(collectSectionRows()));

  const st=document.getElementById("uploadStatus");
  st.style.color="#555";
  st.textContent="⏳ Đang đọc PDF, embedding và upload lên Pinecone...";

  try{
    const r=await fetch("/admin/api/upload-pdf",{method:"POST",body:fd});
    const t=await r.text();
    let d={};
    try{d=JSON.parse(t)}catch{d={detail:t}}
    if(!r.ok)throw Error(
      typeof d.detail==="string"?d.detail:JSON.stringify(d.detail||d)
    );

    st.style.color="green";
    st.textContent=
      "✓ Upload thành công\n"+
      "File: "+d.file_name+
      "\nMôn học: "+d.subject+
      "\nSố trang: "+d.pages+
      "\nChunks: "+d.chunks+
      "\nMetadata: "+d.metadata_rows+" dòng";

    await loadCatalog();
  }catch(e){
    st.style.color="red";
    st.textContent="✕ Upload lỗi: "+e.message;
  }
}

async function loadUsers(){
  const d=await api("/admin/api/users?password="+encodeURIComponent(pw));
  document.getElementById("count").textContent="  Tổng: "+d.users.length;

  document.getElementById("users").innerHTML=d.users.map(u=>{
    const s=u.subscription||{};
    const st=u.status||"PENDING";
    const ex=s.expires_at?
      new Date(s.expires_at).toLocaleString("vi-VN"):"-";

    return `<div class="user ${selectedUser===u.id?'sel':''}"
      onclick="selectUser(${u.id},'${esc(u.nickname)}')">
      <b>#${u.id} ${esc(u.nickname)}</b> — ${esc(u.phone)}
      <div>
        <span class="status-${st}"><b>${st}</b></span>
        · ${esc(s.plan||"-")} · hết hạn: ${ex}
      </div>
      <div class="small">Bấm để xem lịch sử và chat</div>
      <div style="margin-top:7px">
        <button onclick="event.stopPropagation();act(${u.id},1)">1 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},3)">3 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},12)">12 tháng</button>
        <button class="red"
          onclick="event.stopPropagation();lock(${u.id})">Khóa</button>
      </div>
    </div>`;
  }).join("");
}

async function selectUser(id,nickname){
  selectedUser=id;
  lastChatId=0;
  seenMessageIds=new Set();

  document.getElementById("chatTitle").textContent=
    "💬 Chat với "+nickname+" (#"+id+")";
  document.getElementById("chatInput").disabled=false;
  document.getElementById("sendBtn").disabled=false;
  document.getElementById("messages").innerHTML="";

  await loadUsers();
  await pollSelectedChat(true);
}

function addMessage(m){
  if(m && m.id!=null){
    const id=String(m.id);
    if(seenMessageIds.has(id))return;
    seenMessageIds.add(id);
    lastChatId=Math.max(lastChatId,Number(m.id)||0);
  }

  const box=document.getElementById("messages");
  const div=document.createElement("div");
  div.className="msg "+(m.sender==="admin"?"admin":"user");
  const who=m.sender==="admin"?"Admin":
             m.sender==="user"?"Khách":"System";
  const when=m.created_at?
    new Date(m.created_at).toLocaleString("vi-VN"):"";

  div.innerHTML="<b>"+who+"</b><br>"+esc(m.message)+
    "<div class='meta'>"+when+"</div>";
  box.appendChild(div);
  box.scrollTop=box.scrollHeight;
}

async function pollSelectedChat(initial=false){
  if(!selectedUser||!pw||pollBusy)return;
  pollBusy=true;

  try{
    const d=await api(
      "/admin/api/chat/history?user_id="+selectedUser+
      "&password="+encodeURIComponent(pw)+
      "&limit=200&after_id="+(initial?0:lastChatId)
    );

    d.messages.forEach(addMessage);
    if(d.last_id!=null)
      lastChatId=Math.max(lastChatId,Number(d.last_id)||0);

    document.getElementById("wsState").textContent=
      "● Chat đang đồng bộ tự động";
  }catch(e){
    console.error("Admin polling error:",e);
    document.getElementById("wsState").textContent=
      "● Đang kết nối lại chat...";
  }finally{
    pollBusy=false;
  }
}

function startChatPolling(){
  if(pollTimer)clearInterval(pollTimer);
  pollTimer=setInterval(()=>pollSelectedChat(false),1200);
}

async function getWsToken(){
  try{
    const d=await api(
      "/admin/api/ws-token?password="+encodeURIComponent(pw)
    );
    wsToken=d.token;
    connectWS();
  }catch(e){
    console.log("WebSocket token unavailable; polling remains active.");
  }
}

function connectWS(){
  if(!wsToken)return;
  if(ws && ws.readyState===WebSocket.OPEN)return;

  const proto=location.protocol==="https:"?"wss":"ws";
  ws=new WebSocket(
    proto+"://"+location.host+
    "/ws/admin?token="+encodeURIComponent(wsToken)
  );

  ws.onopen=()=>{
    document.getElementById("wsState").textContent=
      "● Admin realtime: Đã kết nối";
  };

  ws.onmessage=e=>{
    try{
      const d=JSON.parse(e.data);

      if(d.type==="connected"){
        document.getElementById("wsState").textContent=
          "● Admin realtime: Đã kết nối";
        return;
      }

      if(d.type==="message" && d.data){
        const uid=Number(d.data.user_id);

        if(selectedUser && uid===Number(selectedUser)){
          addMessage(d.data);
        }
        loadUsers();
      }

      if(d.type==="error"){
        document.getElementById("wsState").textContent=
          "● Lỗi: "+(d.message||"WebSocket");
      }
    }catch(err){
      console.error("Admin WS error",err);
    }
  };

  ws.onerror=()=>{};
  ws.onclose=()=>{
    setTimeout(connectWS,2000);
  };
}

async function sendAdminMessage(){
  const inp=document.getElementById("chatInput");
  const msg=inp.value.trim();

  if(!msg||!selectedUser)return;

  try{
    const d=await api("/admin/api/chat/send",{
      method:"POST",
      body:JSON.stringify({
        password:pw,
        user_id:selectedUser,
        message:msg
      })
    });

    inp.value="";
    if(d.message)addMessage(d.message);
  }catch(e){
    alert("Không gửi được tin nhắn: "+e.message);
  }
}

async function act(id,m){
  if(!confirm("Kích hoạt/gia hạn "+m+" tháng?"))return;
  await api("/admin/api/users/"+id+"/activate",{
    method:"POST",
    body:JSON.stringify({
      password:pw,months:m,plan:"N5"
    })
  });
  loadUsers();
}

async function lock(id){
  if(!confirm("Khóa tài khoản?"))return;
  await api("/admin/api/users/"+id+"/status",{
    method:"POST",
    body:JSON.stringify({
      password:pw,status:"LOCKED"
    })
  });
  loadUsers();
}

async function loadCatalog(){
  try{
    const token="";
    // Catalog is currently protected by user authentication, so Admin
    // uses the dedicated lightweight admin catalog endpoint below.
    const d=await api(
      "/admin/api/knowledge/catalog?password="+encodeURIComponent(pw)
    );

    document.getElementById("catalog").innerHTML=
      d.documents.length?
      d.documents.map(doc=>`
        <div class="catalog-row">
          <b>${esc(doc.file_name)}</b> — ${esc(doc.subject)}
          ${doc.sections.map(s=>`
            <div class="small" style="margin:5px 0 0 15px">
              ${s.lesson?`Bài: <b>${esc(s.lesson)}</b>
              (trang ${esc(s.lesson_pages||"-")}) · `:""}
              ${s.topic?`Chủ đề: <b>${esc(s.topic)}</b>
              (trang ${esc(s.topic_pages||"-")}) · `:""}
              ${s.question?`Câu hỏi: ${esc(s.question)}
              (trang ${esc(s.question_pages||"-")}) · `:""}
              ${s.answer?`Đáp án: ${esc(s.answer)}
              (trang ${esc(s.answer_pages||"-")})`:""}
            </div>
          `).join("")}
        </div>
      `).join("")
      :"Chưa có tài liệu.";
  }catch(e){
    document.getElementById("catalog").textContent=
      "Không tải được danh mục: "+e.message;
  }
}
</script>
</body>
</html>""")


@app.get("/admin/api/knowledge/catalog")
def admin_knowledge_catalog(password: str):
    check_admin(password)

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT d.id,d.file_name,d.subject,d.namespace,
                       s.id section_id,s.lesson,s.lesson_pages,
                       s.topic,s.topic_pages,s.question,s.question_pages,
                       s.answer,s.answer_pages
                FROM knowledge_documents d
                LEFT JOIN knowledge_sections s ON s.document_id=d.id
                ORDER BY d.id DESC,s.id ASC
            """)
            rows=cur.fetchall()
    finally:
        conn.close()

    documents={}
    for r in rows:
        did=r["id"]
        if did not in documents:
            documents[did]={
                "id":did,
                "file_name":r["file_name"],
                "subject":r["subject"],
                "namespace":r["namespace"],
                "sections":[]
            }
        if r["section_id"] is not None:
            documents[did]["sections"].append({
                "id":r["section_id"],
                "lesson":r["lesson"],
                "lesson_pages":r["lesson_pages"],
                "topic":r["topic"],
                "topic_pages":r["topic_pages"],
                "question":r["question"],
                "question_pages":r["question_pages"],
                "answer":r["answer"],
                "answer_pages":r["answer_pages"]
            })

    return {"documents":list(documents.values())}


@app.get("/health")
def health():
    return {
        "status":"ok",
        "pinecone":index is not None,
        "gemini":gemini is not None,
        "database":bool(DATABASE_URL),
        "index":PINECONE_INDEX,
        "namespace":PINECONE_NAMESPACE,
        "gemini_model":GEMINI_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_multi_metadata:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
