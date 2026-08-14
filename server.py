import os
import io
import uuid
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
import json

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

try:
    import fitz  # PyMuPDF - render scanned PDF pages
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import boto3
    from botocore.client import Config as BotoConfig
except Exception:
    boto3 = None
    BotoConfig = None

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "doraemon")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_RENDER")
ADMIN_WS_TOKEN = os.getenv("ADMIN_WS_TOKEN")
ADMIN_PANEL_PASSWORD = os.getenv("ADMIN_PANEL_PASSWORD", ADMIN_WS_TOKEN)
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Optional Backblaze B2 object storage for original PDFs and extracted images.
# Set these on Render for production image/PDF storage.
B2_ENDPOINT = os.getenv("B2_ENDPOINT", "").strip().rstrip("/")
B2_BUCKET = os.getenv("B2_BUCKET", "").strip()
B2_KEY_ID = os.getenv("B2_KEY_ID", "").strip()
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY", "").strip()
B2_PUBLIC_BASE_URL = os.getenv("B2_PUBLIC_BASE_URL", "").strip().rstrip("/")
B2_PRESIGN_SECONDS = int(os.getenv("B2_PRESIGN_SECONDS", "86400"))
b2 = None

app = FastAPI(title="Doraemon SaaS Server")
SERVER_VERSION = "2026-08-14-gemini-ocr-b2-v1"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pc = None
index = None
gemini = None
connected_users = {}
admin_connections = set()

def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL chưa được cấu hình trên Render.")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY, phone VARCHAR(30) UNIQUE NOT NULL,
                nickname VARCHAR(100) NOT NULL, password_hash TEXT NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                plan VARCHAR(100) NOT NULL DEFAULT 'N5', started_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ, status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS admin_messages (
                id BIGSERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                sender VARCHAR(20) NOT NULL, message TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                is_read BOOLEAN NOT NULL DEFAULT FALSE);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS learning_progress (
                id BIGSERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                subject VARCHAR(255) NOT NULL DEFAULT '',
                content_type VARCHAR(30) NOT NULL DEFAULT 'Từ vựng',
                content_id VARCHAR(255),
                lesson VARCHAR(255), topic VARCHAR(255), item_key VARCHAR(500),
                score INTEGER,
                status VARCHAR(50) NOT NULL DEFAULT 'in_progress',
                current_position INTEGER DEFAULT 0,
                current_page INTEGER,
                attempt_count INTEGER DEFAULT 0,
                correct_count INTEGER DEFAULT 0,
                wrong_count INTEGER DEFAULT 0,
                last_studied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                next_review_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_documents (
                id BIGSERIAL PRIMARY KEY, source_file VARCHAR(500) NOT NULL,
                subject VARCHAR(255) NOT NULL,
                content_type VARCHAR(30) NOT NULL DEFAULT 'Từ vựng',
                lesson VARCHAR(255), lesson_pages VARCHAR(255), topic VARCHAR(255), topic_pages VARCHAR(255),
                question_pages VARCHAR(255), answer_pages VARCHAR(255),
                namespace VARCHAR(255) NOT NULL DEFAULT '__default__',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_images (
                id BIGSERIAL PRIMARY KEY, source_file VARCHAR(500) NOT NULL,
                subject VARCHAR(255) NOT NULL DEFAULT '', content_type VARCHAR(30) NOT NULL DEFAULT 'Từ vựng',
                lesson VARCHAR(255), topic VARCHAR(255), page INTEGER NOT NULL,
                image_key TEXT NOT NULL, image_url TEXT, description TEXT,
                width INTEGER, height INTEGER, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")

            # Safe migrations for databases created by previous versions.
            for sql in [
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS subject VARCHAR(255);",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS content_type VARCHAR(30) DEFAULT 'Từ vựng';",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS content_id VARCHAR(255);",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS lesson VARCHAR(255);",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS topic VARCHAR(255);",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS item_key VARCHAR(500);",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS score INTEGER;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'in_progress';",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS current_position INTEGER DEFAULT 0;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS current_page INTEGER;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS correct_count INTEGER DEFAULT 0;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS wrong_count INTEGER DEFAULT 0;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS last_studied_at TIMESTAMPTZ DEFAULT NOW();",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS next_review_at TIMESTAMPTZ;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;",
                "ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();",
                "ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS content_type VARCHAR(30) DEFAULT 'Từ vựng';",
            ]:
                cur.execute(sql)
            cur.execute("UPDATE learning_progress SET last_studied_at=NOW() WHERE last_studied_at IS NULL;")
            cur.execute("UPDATE learning_progress SET subject='' WHERE subject IS NULL;")
            cur.execute("UPDATE learning_progress SET content_type='Từ vựng' WHERE content_type IS NULL OR TRIM(content_type)='';")
            cur.execute("UPDATE knowledge_documents SET content_type='Từ vựng' WHERE content_type IS NULL OR TRIM(content_type)='';")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_learning_progress_user
                           ON learning_progress(user_id,last_studied_at DESC);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_learning_progress_content
                           ON learning_progress(user_id,content_type,content_id,last_studied_at DESC);""")
        conn.commit()
    finally:
        conn.close()

@app.on_event("startup")
def startup():
    global pc, index, gemini, b2
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
    if GEMINI_API_KEY:
        gemini = genai.Client(api_key=GEMINI_API_KEY)
    if B2_ENDPOINT and B2_KEY_ID and B2_APPLICATION_KEY and B2_BUCKET and boto3:
        b2 = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT if B2_ENDPOINT.startswith("http") else f"https://{B2_ENDPOINT}",
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APPLICATION_KEY,
            region_name="us-east-005",
            config=BotoConfig(signature_version="s3v4") if BotoConfig else None,
        )
        print("Backblaze B2: OK", B2_BUCKET)
    else:
        print("WARNING: Backblaze B2 chưa được cấu hình; ảnh/PDF sẽ không được lưu cloud.")
    if DATABASE_URL:
        init_db()
        print("PostgreSQL: OK")
    else:
        print("WARNING: DATABASE_URL chưa được cấu hình.")
    print("Gemini model:", GEMINI_MODEL)

class RegisterRequest(BaseModel):
    phone: str
    nickname: str
    password: str

class LoginRequest(BaseModel):
    phone: str
    password: str

class ChatRequest(BaseModel):
    # API mới dùng "message". Giữ "prompt" để tương thích với client cũ.
    message: str | None = None
    prompt: str | None = None
    chat_history: list = []
    image_base64: str | None = None
    use_knowledge_base: bool = True
    knowledge_namespace: str = "default"
    top_k: int = 8

    @property
    def text(self) -> str:
        value = self.message if self.message is not None else self.prompt
        return (value or "").strip()

def hash_password(p): return pwd_context.hash(p)
def verify_password(p, h): return pwd_context.verify(p, h)

def create_token(user_id):
    exp = datetime.now(timezone.utc) + timedelta(days=30)
    return jwt.encode({"sub": str(user_id), "exp": exp, "type": "user"},
                      JWT_SECRET, algorithm="HS256")

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
            cur.execute("SELECT id,phone,nickname,status,created_at FROM users WHERE id=%s", (uid,))
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
            cur.execute("""SELECT id,plan,started_at,expires_at,status FROM subscriptions
                           WHERE user_id=%s ORDER BY id DESC LIMIT 1""", (user_id,))
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

@app.post("/auth/register")
def register(data: RegisterRequest):
    phone, nickname, password = data.phone.strip(), data.nickname.strip(), data.password
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
            cur.execute("""INSERT INTO users(phone,nickname,password_hash,status)
                           VALUES(%s,%s,%s,'PENDING') RETURNING id""",
                        (phone, nickname, hash_password(password)))
            uid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "user_id": uid, "status": "PENDING",
            "message": "Đăng ký thành công. Tài khoản đang chờ Admin kích hoạt."}

@app.post("/auth/login")
def login(data: LoginRequest):
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id,phone,nickname,password_hash,status FROM users WHERE phone=%s""",
                        (data.phone.strip(),))
            user = cur.fetchone()
    finally:
        conn.close()
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "SĐT hoặc mật khẩu không đúng.")
    token = create_token(user["id"])
    sub, msg = subscription_status(user["id"])
    return {"success": True, "access_token": token, "token_type": "bearer",
            "user": {k: user[k] for k in ("id","phone","nickname","status")},
            "subscription": sub, "subscription_message": msg}

@app.get("/auth/me")
def me(authorization: Optional[str] = Header(default=None)):
    user = current_user(bearer(authorization))
    sub, msg = subscription_status(user["id"])
    return {"user": user, "subscription": sub, "subscription_message": msg}

@app.get("/admin-chat/history")
def history(limit: int = 100, authorization: Optional[str] = Header(default=None)):
    user = current_user(bearer(authorization))
    limit = max(1, min(limit, 500))
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id,sender,message,created_at,is_read FROM admin_messages
                           WHERE user_id=%s ORDER BY id DESC LIMIT %s""", (user["id"], limit))
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
            cur.execute("""INSERT INTO admin_messages(user_id,sender,message)
                           VALUES(%s,'user',%s)
                           RETURNING id,user_id,sender,message,created_at,is_read""",
                        (user["id"], msg))
            row = dict(cur.fetchone())
        conn.commit()
    finally:
        conn.close()
    return {"message": row}

async def send_json(ws, data):
    try:
        await ws.send_json(data); return True
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
        await websocket.send_json({"type":"error","message":e.detail})
        await websocket.close(code=1008); return
    uid = user["id"]
    connected_users.setdefault(uid, set()).add(websocket)
    try:
        await websocket.send_json({"type":"connected","message":"Đã kết nối chat Admin.","user_id":uid})
        while True:
            data = await websocket.receive_json()
            msg = str(data.get("message","")).strip()
            if not msg: continue
            conn = db()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""INSERT INTO admin_messages(user_id,sender,message)
                                   VALUES(%s,'user',%s)
                                   RETURNING id,user_id,sender,message,created_at,is_read""", (uid,msg))
                    row = dict(cur.fetchone())
                conn.commit()
            finally:
                conn.close()
            await websocket.send_json({"type":"message","data":row})
            await notify_admin({"type":"message","data":row})
    except WebSocketDisconnect:
        pass
    finally:
        connected_users.get(uid, set()).discard(websocket)
        if not connected_users.get(uid): connected_users.pop(uid, None)

@app.websocket("/ws/admin")
async def ws_admin(websocket: WebSocket):
    await websocket.accept()
    if not ADMIN_WS_TOKEN or websocket.query_params.get("token") != ADMIN_WS_TOKEN:
        await websocket.send_json({"type":"error","message":"Admin token không hợp lệ."})
        await websocket.close(code=1008); return
    admin_connections.add(websocket)
    try:
        await websocket.send_json({"type":"connected","message":"Admin WebSocket connected."})
        while True:
            data = await websocket.receive_json()
            uid, msg = int(data.get("user_id")), str(data.get("message","")).strip()
            if not msg: continue
            conn = db()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""INSERT INTO admin_messages(user_id,sender,message)
                                   VALUES(%s,'admin',%s)
                                   RETURNING id,user_id,sender,message,created_at,is_read""", (uid,msg))
                    row = dict(cur.fetchone())
                conn.commit()
            finally:
                conn.close()
            await notify_user(uid, {"type":"message","data":row})
            await websocket.send_json({"type":"message","data":row})
    except WebSocketDisconnect:
        pass
    finally:
        admin_connections.discard(websocket)

def require_active_user(authorization):
    user = current_user(bearer(authorization))
    sub, msg = subscription_status(user["id"])
    if msg: raise HTTPException(403, msg)
    return user

def embed_text(text):
    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    r = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    return r.embeddings[0].values

@app.post("/search")
def search(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization)
    if not index: raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    if not data.text:
        raise HTTPException(400, "Tin nhắn không được để trống.")
    result = index.query(vector=embed_text(data.text), top_k=8, include_metadata=True)
    matches = []
    for m in result.matches:
        md = m.metadata or {}
        text = md.get("text", md.get("content", ""))
        if text: matches.append({"score":float(m.score),"text":text,"metadata":md})
    return {"matches":matches}


CONTENT_TYPES = {"Từ vựng", "Ngữ pháp", "Bài tập", "Truyện đọc"}


def _normalize_content_type(value):
    value = str(value or "").strip()
    return value if value in CONTENT_TYPES else "Từ vựng"


def _review_days(content_type, score=None, status="in_progress"):
    """Review schedule: exercises use score; non-scored learning uses a gentle revisit schedule."""
    if content_type == "Truyện đọc":
        return None
    if content_type == "Bài tập" and score is not None:
        try:
            sc = float(score)
        except Exception:
            sc = None
        if sc is not None:
            if sc < 60: return 1
            if sc < 80: return 3
            if sc < 90: return 7
            return 14
    if status == "completed":
        return 7 if content_type in {"Từ vựng", "Ngữ pháp"} else None
    return 3 if content_type in {"Từ vựng", "Ngữ pháp"} else None


def record_learning_event(user_id, event):
    event = dict(event or {})
    content_type = _normalize_content_type(event.get("content_type"))
    subject = str(event.get("subject") or "Tiếng Nhật").strip()
    lesson = str(event.get("lesson") or "").strip() or None
    topic = str(event.get("topic") or "").strip() or None
    item_key = str(event.get("item_key") or lesson or topic or "").strip() or None
    content_id = str(event.get("content_id") or f"{content_type}|{subject}|{lesson or ''}|{topic or ''}|{item_key or ''}")[:255]
    status = str(event.get("status") or "in_progress").strip()
    if status not in {"in_progress", "completed", "review", "needs_review"}:
        status = "in_progress"

    score = event.get("score")
    try:
        score = int(score) if score is not None and str(score).strip() != "" else None
    except Exception:
        score = None
    if score is not None:
        score = max(0, min(100, score))

    current_position = int(event.get("current_position") or 0)
    current_page = event.get("current_page")
    try:
        current_page = int(current_page) if current_page is not None and str(current_page).strip() != "" else None
    except Exception:
        current_page = None
    attempt_count = max(0, int(event.get("attempt_count") or 0))
    correct_count = max(0, int(event.get("correct_count") or 0))
    wrong_count = max(0, int(event.get("wrong_count") or 0))

    # Exercise scoring: correct+wrong is the source of truth when supplied.
    total = correct_count + wrong_count
    if content_type == "Bài tập" and total > 0 and score is None:
        score = round(correct_count * 100 / total)
    if content_type == "Bài tập" and total > 0 and status == "in_progress":
        status = "completed" if wrong_count == 0 else "needs_review"
    if content_type != "Bài tập" and event.get("completed") is True:
        status = "completed"

    days = _review_days(content_type, score, status)
    next_review = None if days is None else datetime.now(timezone.utc) + timedelta(days=days)
    completed_at = datetime.now(timezone.utc) if status == "completed" else None

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id,attempt_count,correct_count,wrong_count
                           FROM learning_progress
                           WHERE user_id=%s AND content_type=%s AND content_id=%s
                           ORDER BY id DESC LIMIT 1""",
                        (user_id, content_type, content_id))
            old = cur.fetchone()
            if old:
                if content_type == "Bài tập":
                    attempts = max(int(old.get("attempt_count") or 0), attempt_count) + 1
                else:
                    attempts = max(int(old.get("attempt_count") or 0), attempt_count)
                correct = max(int(old.get("correct_count") or 0), correct_count) if content_type == "Bài tập" else 0
                wrong = max(int(old.get("wrong_count") or 0), wrong_count) if content_type == "Bài tập" else 0
                cur.execute("""UPDATE learning_progress SET
                    subject=%s,lesson=%s,topic=%s,item_key=%s,score=%s,status=%s,
                    current_position=%s,current_page=%s,attempt_count=%s,correct_count=%s,wrong_count=%s,
                    last_studied_at=NOW(),next_review_at=%s,completed_at=%s
                    WHERE id=%s
                    RETURNING *""",
                    (subject,lesson,topic,item_key,score,status,current_position,current_page,
                     attempts,correct,wrong,next_review,completed_at,old["id"]))
            else:
                attempts = max(1, attempt_count) if content_type == "Bài tập" else 0
                cur.execute("""INSERT INTO learning_progress
                    (user_id,subject,content_type,content_id,lesson,topic,item_key,score,status,
                     current_position,current_page,attempt_count,correct_count,wrong_count,
                     last_studied_at,next_review_at,completed_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),%s,%s)
                    RETURNING *""",
                    (user_id,subject,content_type,content_id,lesson,topic,item_key,score,status,
                     current_position,current_page,attempts,correct_count,wrong_count,next_review,completed_at))
            row = dict(cur.fetchone())
        conn.commit()
        return row
    finally:
        conn.close()


def infer_learning_event(user_id, user_text, reply, catalog, learning, source_meta=None):
    """Infer only learning progress, never a score. Exercises are scored via /learning/progress."""
    text = (user_text or "").strip()
    low = text.lower()
    source_meta = source_meta or []
    chosen = None
    for md in source_meta:
        if md.get("content_type") in CONTENT_TYPES:
            chosen = md
            break

    # Prefer explicit catalog matches in the user's request.
    if not chosen:
        for item in catalog:
            hay = " ".join(str(item.get(k) or "") for k in ("lesson", "topic", "subject")).strip()
            if hay and any(part.lower() in low for part in [str(item.get("lesson") or ""), str(item.get("topic") or "")] if part):
                chosen = item
                break

    if not chosen:
        # Detect the content type from explicit words in the conversation.
        for typ, keys in {
            "Truyện đọc": ["truyện", "câu chuyện", "đọc truyện"],
            "Từ vựng": ["từ vựng", "từ mới", "vocabulary"],
            "Ngữ pháp": ["ngữ pháp", "grammar"],
            "Bài tập": ["bài tập", "làm bài", "câu hỏi", "quiz"]
        }.items():
            if any(k in low for k in keys):
                chosen = {"content_type": typ, "subject": "Tiếng Nhật"}
                break

    if not chosen:
        return None

    content_type = _normalize_content_type(chosen.get("content_type"))
    subject = chosen.get("subject") or chosen.get("course") or "Tiếng Nhật"
    lesson = chosen.get("lesson") or None
    topic = chosen.get("topic") or None
    page = chosen.get("page")
    # User can explicitly state a page: "trang 7", "đến trang 7".
    m = re.search(r"(?:trang|page)\s*(\d+)", low)
    if m:
        page = int(m.group(1))

    completed = any(x in low for x in ["đã học xong", "học xong", "đọc xong", "xong bài", "hoàn thành"])
    status = "completed" if completed else "in_progress"
    item_key = lesson or topic or str(chosen.get("source_file") or "") or None
    return {
        "content_type": content_type,
        "subject": subject,
        "content_id": f"{content_type}|{subject}|{lesson or ''}|{topic or ''}|{item_key or ''}",
        "lesson": lesson,
        "topic": topic,
        "item_key": item_key,
        "current_page": page,
        "current_position": 0,
        "status": status,
        "completed": completed,
    }

@app.post("/api/proxy-chat")
def proxy_chat(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)
    if not gemini: raise HTTPException(500, "Gemini chưa được khởi tạo.")
    if not index: raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    if not data.text: raise HTTPException(400, "Tin nhắn không được để trống.")

    result = index.query(
        vector=embed_text(data.text),
        top_k=data.top_k,
        include_metadata=True,
        namespace=data.knowledge_namespace or "__default__"
    )
    contexts=[]; source_meta=[]
    for m in result.matches:
        md=m.metadata or {}
        txt=md.get("text",md.get("content",""))
        if txt:
            label=(
                f"[Loại: {md.get('content_type','') or 'Không rõ'} | "
                f"Môn: {md.get('subject',md.get('course',''))} | "
                f"Bài: {md.get('lesson','')} | Chủ đề: {md.get('topic','')} | Trang: {md.get('page','')}]"
            )
            contexts.append(label+"\n"+txt)
            source_meta.append(md)

    learning=[]; catalog=[]
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT subject,content_type,content_id,lesson,topic,item_key,score,status,
                       current_position,current_page,attempt_count,correct_count,wrong_count,
                       last_studied_at,next_review_at,completed_at
                FROM learning_progress
                WHERE user_id=%s
                ORDER BY last_studied_at DESC LIMIT 100
            """,(user["id"],))
            learning=[dict(x) for x in cur.fetchall()]
            cur.execute("""
                SELECT subject,content_type,lesson,lesson_pages,topic,topic_pages,
                       question_pages,answer_pages,source_file,namespace
                FROM knowledge_documents
                ORDER BY subject,content_type,lesson,topic,id
            """)
            catalog=[dict(x) for x in cur.fetchall()]
    finally:
        conn.close()

    prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật cá nhân.

QUY TẮC QUAN TRỌNG:
1. Nếu user đã chọn một nội dung cụ thể và yêu cầu thực hiện nó, HÃY THỰC HIỆN NGAY.
   Ví dụ user nói "hãy kể cho mình truyện Cô bé quàng khăn đỏ" thì phải bắt đầu
   kể truyện, không hỏi lại "có muốn đọc truyện không?".
2. Không liên tục quay lại câu hỏi "Hôm nay bạn muốn học gì?" khi user đã xác định
   bài/chủ đề/nội dung. Chỉ hỏi lại lựa chọn học khi user thực sự chưa chọn nội dung.
3. Khi user muốn học mới, có thể hỏi lần lượt môn học -> loại nội dung -> bài học/chủ đề.
4. Chỉ dùng tên môn/bài/chủ đề có trong DANH MỤC nếu đang đề xuất nội dung.
5. Nếu user muốn ôn, ưu tiên nội dung có status review, điểm thấp hoặc lâu chưa học.
6. Loại nội dung gồm: Từ vựng, Ngữ pháp, Bài tập, Truyện đọc.
7. Với Bài tập, hỏi user làm trước rồi mới đưa đáp án. Không tiết lộ đáp án trước.
8. Với Truyện đọc, hãy kể/đọc trực tiếp theo tài liệu khi user yêu cầu; có thể chia
   thành từng đoạn và hỏi tiếp khi cần, nhưng không hỏi lại việc có muốn đọc hay không.
9. Với Từ vựng, dạy từ mới và có thể kiểm tra nhẹ nhưng không bắt buộc chấm điểm.
10. Với Ngữ pháp, giải thích, ví dụ và luyện tập.
11. Dựa vào LỊCH SỬ HỌC để tiếp tục đúng phần user đang học dở.
12. Không bịa trang tài liệu.
13. Không tự chấm điểm Từ vựng, Ngữ pháp hoặc Truyện đọc. Chỉ ghi nhận tiến độ.
14. Với Bài tập, chỉ chấm khi người học thực sự nộp/được xác định kết quả; không tự suy đoán điểm từ lời giải thích của Doraemon.
15. Nếu người học đang học dở, tiếp tục từ tiến độ đã lưu thay vì hỏi lại từ đầu.

DANH MỤC GIÁO TRÌNH:
{json.dumps(catalog,ensure_ascii=False,default=str)}

LỊCH SỬ HỌC CỦA USER:
{json.dumps(learning,ensure_ascii=False,default=str)}

NỘI DUNG TÌM ĐƯỢC:
{chr(10).join(contexts)}

TIN NHẮN:
{data.text}"""

    response=gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    reply=response.text or ""
    image_keys=[]
    for md in source_meta:
        try:
            keys=json.loads(md.get("image_keys") or "[]")
            if isinstance(keys,list):
                image_keys.extend([str(k) for k in keys if k])
        except Exception:
            pass
    image_keys=list(dict.fromkeys(image_keys))[:6]
    images=[{"key":k,"url":b2_url(k)} for k in image_keys if b2_url(k)]

    # Tự động ghi nhận tiến độ. Nếu bước này lỗi thì KHÔNG làm hỏng câu trả lời
    # của người học.
    tracked_event=None
    try:
        event=infer_learning_event(user["id"],data.text,reply,catalog,learning,source_meta)
        if event:
            tracked_event=record_learning_event(user["id"],event)
    except Exception as e:
        print("Learning progress save skipped:", type(e).__name__, str(e))

    return {
        "reply":reply,
        "model":GEMINI_MODEL,
        "sources":source_meta[:10],
        "images":images,
        "learning_history_count":len(learning),
        "learning_progress":tracked_event
    }


@app.post("/learning/progress")
def save_learning_progress(data: dict, authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization)
    payload=dict(data or {})
    payload["content_type"]=_normalize_content_type(payload.get("content_type"))
    # Only Bài tập should normally carry score/correct/wrong.
    if payload["content_type"] != "Bài tập":
        payload.pop("score",None)
        payload.pop("correct_count",None)
        payload.pop("wrong_count",None)
    row=record_learning_event(user["id"],payload)
    return {"success":True,"progress":row}

@app.get("/learning/summary")
def learning_summary(authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization)
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT subject,content_type,content_id,lesson,topic,item_key,score,status,
                       current_position,current_page,attempt_count,correct_count,wrong_count,
                       last_studied_at,next_review_at,completed_at
                FROM learning_progress
                WHERE user_id=%s
                ORDER BY last_studied_at DESC LIMIT 200
            """,(user["id"],))
            rows=[dict(x) for x in cur.fetchall()]
        return {"success":True,"user_id":user["id"],"learning_history":rows}
    finally:
        conn.close()


@app.get("/learning/catalog")
def learning_catalog(authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization); conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT subject,content_type,lesson,lesson_pages,topic,topic_pages,question_pages,answer_pages,source_file,namespace FROM knowledge_documents ORDER BY subject,lesson,topic,id")
            return {"success":True,"documents":[dict(x) for x in cur.fetchall()]}
    finally: conn.close()

def check_admin(password: str):
    expected = os.getenv("ADMIN_PANEL_PASSWORD", os.getenv("ADMIN_WS_TOKEN", ""))
    if not expected:
        raise HTTPException(500, "ADMIN_PANEL_PASSWORD chưa được cấu hình trên Render.")
    if password != expected:
        raise HTTPException(401, "Admin password không đúng.")


@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    return HTMLResponse("""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Doraemon Admin</title>
<style>
*{box-sizing:border-box}body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#222}
header{background:#1677ff;color:#fff;padding:18px 24px;font-size:22px;font-weight:700}
main{max-width:1250px;margin:20px auto;padding:0 15px}
.card{background:#fff;padding:18px;border-radius:12px;margin-bottom:18px;box-shadow:0 2px 10px #0001}
input,button{padding:9px;border-radius:7px;border:1px solid #ccc}
button{background:#1677ff;color:#fff;border:0;cursor:pointer}
button.gray{background:#666}button.red{background:#d93025}
#login{max-width:420px;margin:60px auto}.layout{display:grid;grid-template-columns:52% 48%;gap:18px}
.user{padding:10px;border-bottom:1px solid #eee;cursor:pointer}.user:hover{background:#f5f8ff}
.user.sel{background:#e8f1ff}.status-ACTIVE{color:#16803c}.status-PENDING{color:#b76b00}.status-LOCKED{color:#c00}
#users{max-height:610px;overflow:auto}.chat{display:flex;flex-direction:column;height:610px}
#messages{flex:1;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}
.msg{margin:7px 0;padding:8px 10px;border-radius:10px;max-width:82%;white-space:pre-wrap}
.msg.user{background:#dff0ff;margin-right:auto}.msg.admin{background:#dff7df;margin-left:auto}
.meta{font-size:11px;color:#777;margin-top:3px}
.chatbar{display:flex;gap:7px;margin-top:10px}.chatbar input{flex:1}
.small{font-size:13px;color:#666}\n.meta-row input{min-width:0}@media(max-width:1000px){.meta-row{grid-template-columns:1fr 1fr 1fr!important}}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>🤖 Doraemon Admin</header>
<main>
<div class="card" id="login">
<h3>Đăng nhập Admin</h3>
<input id="pw" type="password" placeholder="Mật khẩu Admin" style="width:70%">
<button onclick="login()">Đăng nhập</button>
<div id="err" style="color:#c00;margin-top:8px"></div>
</div>

<div id="panel" style="display:none">
<div class="card">
<h3>📚 Knowledge Base</h3>
<div class="small" style="margin-bottom:10px">
Upload PDF trực tiếp lên Pinecone · Gemini Embedding 768 · Namespace: __default__
</div>
<form id="uploadForm" onsubmit="uploadKnowledge(event)">
<input id="pdfFile" type="file" accept=".pdf,application/pdf" required style="width:100%;margin-bottom:8px">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
<input id="subject" value="Tiếng Nhật" placeholder="Môn học *" required style="flex:1;min-width:180px">
<input id="chunkSize" type="number" value="1200" min="300" max="5000" title="Kích thước chunk" style="width:120px">
<input id="overlap" type="number" value="200" min="0" max="4900" title="Độ chồng lấn" style="width:110px">
</div>
<div style="margin-top:12px">
  <div style="font-weight:700;margin-bottom:7px">📚 Cấu hình nội dung trong PDF</div>
  <div class="small" style="margin-bottom:8px">
    Một file PDF chỉ chọn <b>1 Môn học</b>. Bạn có thể tạo nhiều dòng để mô tả nhiều bài học/chủ đề/câu hỏi/đáp án trong cùng file.
  </div>
  <div id="metaRows"></div>
  <button type="button" class="gray" onclick="addMetaRow()" style="margin-top:8px">＋ Thêm bài/chủ đề</button>
</div>
<div style="margin-top:12px">
  <button id="uploadBtn" type="submit">⬆️ Upload PDF</button>
</div>
</form>
<div id="uploadStatus" class="small" style="margin-top:10px"></div>
</div>

<div class="card">
<button onclick="loadUsers()">🔄 Làm mới</button>
<span id="count" class="small"></span>
<span id="wsState" class="small" style="float:right;color:green">● Đồng bộ realtime: 1 giây</span>
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
</div>
</main>

<script>
let pw="", ws=null, wsToken="", selectedUser=null, seenMessageIds=new Set(), pollTimer=null, pollBusy=false, lastChatId=0;

function esc(x){return String(x??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}
async function api(u,o={}) {
  o.headers={"Content-Type":"application/json",...(o.headers||{})};
  const r=await fetch(u,o); const t=await r.text(); let d={};
  try{d=JSON.parse(t)}catch{d={detail:t}}
  if(!r.ok) throw Error(d.detail||("HTTP "+r.status));
  return d;
}
async function login(){
  pw=document.getElementById("pw").value;
  try{
    await api("/admin/api/users?password="+encodeURIComponent(pw));
    document.getElementById("login").style.display="none";
    document.getElementById("panel").style.display="block";
    document.getElementById("wsState").textContent="● Đồng bộ tin nhắn tự động";
    await loadUsers();
    startChatPolling();
  }catch(e){document.getElementById("err").textContent=e.message}
}
function addMetaRow(values={}){
  const wrap=document.getElementById("metaRows");
  const row=document.createElement("div");
  row.className="meta-row";
  row.style.cssText="display:grid;grid-template-columns:1fr 1.2fr .9fr 1.2fr .9fr .9fr .9fr auto;gap:6px;margin-bottom:7px;align-items:center";
  row.innerHTML=`
    <select class="m-content-type" title="Loại nội dung">
      <option value="Từ vựng" ${values.content_type==="Từ vựng"?"selected":""}>Từ vựng</option>
      <option value="Ngữ pháp" ${values.content_type==="Ngữ pháp"?"selected":""}>Ngữ pháp</option>
      <option value="Bài tập" ${values.content_type==="Bài tập"?"selected":""}>Bài tập</option>
      <option value="Truyện đọc" ${values.content_type==="Truyện đọc"?"selected":""}>Truyện đọc</option>
    </select>
    <input class="m-lesson" placeholder="Bài học" value="${esc(values.lesson||"")}">
    <input class="m-lesson-pages" placeholder="Trang bài: 1-10" value="${esc(values.lesson_pages||"")}">
    <input class="m-topic" placeholder="Chủ đề" value="${esc(values.topic||"")}">
    <input class="m-topic-pages" placeholder="Trang chủ đề: 3-5" value="${esc(values.topic_pages||"")}">
    <input class="m-question-pages" placeholder="Trang câu hỏi: 8-10" value="${esc(values.question_pages||"")}">
    <input class="m-answer-pages" placeholder="Trang đáp án: 20-21" value="${esc(values.answer_pages||"")}">
    <button type="button" class="red" onclick="this.parentElement.remove()">✕</button>`;
  wrap.appendChild(row);
}
function getMetaRows(){
  return [...document.querySelectorAll(".meta-row")].map(row=>({
    content_type:row.querySelector(".m-content-type").value,
    lesson:row.querySelector(".m-lesson").value.trim(),
    lesson_pages:row.querySelector(".m-lesson-pages").value.trim(),
    topic:row.querySelector(".m-topic").value.trim(),
    topic_pages:row.querySelector(".m-topic-pages").value.trim(),
    question_pages:row.querySelector(".m-question-pages").value.trim(),
    answer_pages:row.querySelector(".m-answer-pages").value.trim()
  })).filter(x=>x.lesson||x.lesson_pages||x.topic||x.topic_pages||x.question_pages||x.answer_pages);
}
addMetaRow();

async function uploadKnowledge(event){
  event.preventDefault();
  const file=document.getElementById("pdfFile").files[0];
  if(!file)return;
  const status=document.getElementById("uploadStatus"), btn=document.getElementById("uploadBtn");
  const rows=getMetaRows();
  btn.disabled=true;
  status.textContent="⏳ Đang phân tích PDF... PDF scan sẽ được OCR bằng Gemini và ảnh sẽ được lưu vào Backblaze B2 nếu đã cấu hình...";
  try{
    const fd=new FormData();
    fd.append("file",file);
    fd.append("password",pw);
    fd.append("subject",document.getElementById("subject").value.trim());
    fd.append("metadata_json",JSON.stringify(rows));
    fd.append("chunk_size",document.getElementById("chunkSize").value||1200);
    fd.append("overlap",document.getElementById("overlap").value||200);
    const r=await fetch("/admin/api/knowledge/upload",{method:"POST",body:fd});
    const t=await r.text(); let d={}; try{d=JSON.parse(t)}catch{d={detail:t}}
    if(!r.ok)throw Error(d.detail||("HTTP "+r.status));
    status.textContent=`✅ ${d.filename}: ${d.pages} trang · OCR scan: ${d.scanned_pages_ocr||0} trang · ${d.chunks} chunks · ${d.records} cấu hình · ${d.images||0} ảnh · ${d.dimension} dimensions`;
    document.getElementById("pdfFile").value="";
  }catch(e){status.textContent="❌ Upload lỗi: "+e.message}
  finally{btn.disabled=false}
}

async function loadUsers(){
  const d=await api("/admin/api/users?password="+encodeURIComponent(pw));
  document.getElementById("count").textContent="  Tổng: "+d.users.length;
  document.getElementById("users").innerHTML=d.users.map(u=>{
    const s=u.subscription||{}, st=u.status||"PENDING";
    const ex=s.expires_at?new Date(s.expires_at).toLocaleString("vi-VN"):"-";
    return `<div class="user ${selectedUser===u.id?'sel':''}" onclick="selectUser(${u.id},'${esc(u.nickname)}')">
      <b>#${u.id} ${esc(u.nickname)}</b> — ${esc(u.phone)}
      <div><span class="status-${st}"><b>${st}</b></span> · ${esc(s.plan||"-")} · hết hạn: ${ex}</div>
      <div class="small">Bấm để xem lịch sử và chat</div>
      <div style="margin-top:7px">
        <button onclick="event.stopPropagation();act(${u.id},1)">1 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},3)">3 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},12)">12 tháng</button>
        <button class="red" onclick="event.stopPropagation();lock(${u.id})">Khóa</button>
      </div>
    </div>`;
  }).join("");
}
async function selectUser(id,nickname){
  selectedUser=id; lastChatId=0; seenMessageIds=new Set();
  document.getElementById("chatTitle").textContent="💬 Chat với "+nickname+" (#"+id+")";
  document.getElementById("chatInput").disabled=false; document.getElementById("sendBtn").disabled=false;
  document.getElementById("messages").innerHTML="";
  await loadUsers();
  await pollSelectedChat(true);
}
function addMessage(m){
  if(m && m.id!=null){const id=String(m.id); if(seenMessageIds.has(id))return; seenMessageIds.add(id); lastChatId=Math.max(lastChatId,Number(m.id)||0);}
  const box=document.getElementById("messages"), div=document.createElement("div");
  div.className="msg "+(m.sender==="admin"?"admin":"user");
  const who=m.sender==="admin"?"Admin":m.sender==="user"?"Khách":"System";
  const when=m.created_at?new Date(m.created_at).toLocaleString("vi-VN"):"";
  div.innerHTML="<b>"+who+"</b><br>"+esc(m.message)+"<div class='meta'>"+when+"</div>";
  box.appendChild(div); box.scrollTop=box.scrollHeight;
}
async function pollSelectedChat(initial=false){
  if(!selectedUser||!pw||pollBusy)return;
  pollBusy=true;
  try{
    const d=await api("/admin/api/chat/history?user_id="+selectedUser+"&password="+encodeURIComponent(pw)+"&limit=200&after_id="+(initial?0:lastChatId));
    if(selectedUser) d.messages.forEach(addMessage);
    if(d.last_id!=null) lastChatId=Math.max(lastChatId,Number(d.last_id)||0);
    document.getElementById("wsState").textContent="● Chat đang đồng bộ tự động";
  }catch(e){
    console.error("Admin polling error:",e);
    document.getElementById("wsState").textContent="● Đang kết nối lại chat...";
  }finally{pollBusy=false;}
}
function startChatPolling(){
  if(pollTimer)clearInterval(pollTimer);
  pollTimer=setInterval(()=>pollSelectedChat(false),1500);
}
function connectWS(){
  if(ws && ws.readyState===WebSocket.OPEN)return;
  const proto=location.protocol==="https:"?"wss":"ws";
  ws=new WebSocket(proto+"://"+location.host+"/ws/admin?token="+encodeURIComponent(wsToken));
  ws.onopen=()=>{document.getElementById("wsState").textContent="● Admin realtime: Đã kết nối";};
  ws.onmessage=e=>{
    try{
      const d=JSON.parse(e.data);
      if(d.type==="connected"){
        document.getElementById("wsState").textContent="● Admin realtime: Đã kết nối";
        return;
      }
      if(d.type==="message" && d.data){
        const uid=Number(d.data.user_id);
        if(selectedUser && uid===Number(selectedUser)){
          addMessage(d.data);
        } else {
          // Có tin nhắn mới từ user khác: vẫn cập nhật danh sách.
          // Khi chọn user đó, lịch sử sẽ được tải đầy đủ.
        }
        loadUsers();
      }
      if(d.type==="error"){
        document.getElementById("wsState").textContent="● Lỗi: "+(d.message||"WebSocket");
      }
    }catch(err){ console.error("Admin WS message error",err); }
  };
  ws.onerror=()=>{ console.log("Optional admin WebSocket unavailable; polling remains active."); };
  ws.onclose=()=>{ console.log("Optional admin WebSocket closed; polling remains active."); };
}
async function sendAdminMessage(){
  const inp=document.getElementById("chatInput"), msg=inp.value.trim();
  if(!msg||!selectedUser)return;
  try{
    await api("/admin/api/chat/send", {
      method:"POST",
      body:JSON.stringify({password:pw,user_id:selectedUser,message:msg})
    });
    inp.value="";
    await pollSelectedChat();
  }catch(e){
    alert("Không gửi được tin nhắn: "+e.message);
  }
}
async function act(id,m){
  if(!confirm("Kích hoạt/gia hạn "+m+" tháng?"))return;
  await api("/admin/api/users/"+id+"/activate",{method:"POST",body:JSON.stringify({password:pw,months:m,plan:"N5"})});
  loadUsers();
}
async function lock(id){
  if(!confirm("Khóa tài khoản?"))return;
  await api("/admin/api/users/"+id+"/status",{method:"POST",body:JSON.stringify({password:pw,status:"LOCKED"})});
  loadUsers();
}
</script>
</body></html>""")


# ============================================================
# Knowledge Base upload from Admin
# ============================================================
def kb_chunk_text(text, chunk_size=1200, overlap=200):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    out=[]; start=0
    while start < len(text):
        end=min(len(text), start+chunk_size)
        chunk=text[start:end].strip()
        if chunk: out.append(chunk)
        if end>=len(text): break
        start=max(start+1, end-overlap)
    return out

def parse_page_ranges(value: str):
    """Parse '1-10,12,15-18' into a set of 1-based PDF page numbers."""
    pages=set()
    value=(value or "").strip()
    if not value:
        return pages
    for part in value.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a,b=part.split("-",1)
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Khoảng trang không hợp lệ: {part}")
            a,b=int(a),int(b)
            if a<1 or b<a:
                raise ValueError(f"Khoảng trang không hợp lệ: {part}")
            pages.update(range(a,b+1))
        else:
            if not part.isdigit() or int(part)<1:
                raise ValueError(f"Trang không hợp lệ: {part}")
            pages.add(int(part))
    return pages

def normalize_kb_records(metadata_json: str, total_pages: int):
    try:
        raw=json.loads(metadata_json or "[]")
    except Exception as e:
        raise ValueError(f"metadata_json không hợp lệ: {e}")
    if not isinstance(raw,list):
        raise ValueError("metadata_json phải là một danh sách các cấu hình.")

    out=[]
    for i,item in enumerate(raw,1):
        if not isinstance(item,dict):
            raise ValueError(f"Cấu hình dòng {i} không hợp lệ.")
        rec={
            "content_type":str(item.get("content_type","Từ vựng")).strip() or "Từ vựng",
            "lesson":str(item.get("lesson","")).strip(),
            "lesson_pages":str(item.get("lesson_pages","")).strip(),
            "topic":str(item.get("topic","")).strip(),
            "topic_pages":str(item.get("topic_pages","")).strip(),
            "question_pages":str(item.get("question_pages","")).strip(),
            "answer_pages":str(item.get("answer_pages","")).strip()
        }
        rec["content_type"]=_normalize_content_type(rec["content_type"])
        for key in ("lesson_pages","topic_pages","question_pages","answer_pages"):
            pages=parse_page_ranges(rec[key])
            if pages and max(pages)>total_pages:
                raise ValueError(f"Dòng {i}: {key} có trang {max(pages)} vượt quá PDF ({total_pages} trang).")
        if not any(rec.values()):
            continue
        if not (rec["lesson"] or rec["topic"]):
            if rec["question_pages"] or rec["answer_pages"]:
                rec["lesson"]=f"Nội dung câu hỏi {i}"
            else:
                raise ValueError(f"Dòng {i}: cần ít nhất Bài học hoặc Chủ đề.")
        rec["_lesson_set"]=parse_page_ranges(rec["lesson_pages"])
        rec["_topic_set"]=parse_page_ranges(rec["topic_pages"])
        rec["_question_set"]=parse_page_ranges(rec["question_pages"])
        rec["_answer_set"]=parse_page_ranges(rec["answer_pages"])
        out.append(rec)
    return out

def metadata_for_page(records, page_no):
    matched=[]
    for r in records:
        ranges=[r["_lesson_set"],r["_topic_set"],r["_question_set"],r["_answer_set"]]
        if any(ranges):
            if any(page_no in x for x in ranges if x):
                matched.append(r)
        else:
            matched.append(r)
    return matched


def b2_ready():
    return b2 is not None and bool(B2_BUCKET)

def b2_put_bytes(key: str, data: bytes, content_type: str):
    if not b2_ready():
        raise RuntimeError("Backblaze B2 chưa được cấu hình. Cần B2_ENDPOINT, B2_KEY_ID, B2_APPLICATION_KEY và B2_BUCKET.")
    b2.put_object(Bucket=B2_BUCKET, Key=key, Body=data, ContentType=content_type)
    if B2_PUBLIC_BASE_URL:
        return f"{B2_PUBLIC_BASE_URL}/{key}"
    return None

def b2_url(key: str):
    if not key:
        return None
    if B2_PUBLIC_BASE_URL:
        return f"{B2_PUBLIC_BASE_URL}/{key}"
    if not b2_ready():
        return None
    return b2.generate_presigned_url(
        "get_object", Params={"Bucket": B2_BUCKET, "Key": key},
        ExpiresIn=max(60, min(B2_PRESIGN_SECONDS, 604800))
    )

def render_pdf_page(raw_pdf: bytes, page_no: int, dpi: int = 150) -> bytes:
    if fitz is None:
        raise RuntimeError("PyMuPDF chưa được cài đặt. Thêm PyMuPDF vào requirements.")
    doc = fitz.open(stream=raw_pdf, filetype="pdf")
    try:
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def _parse_gemini_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Gemini OCR không trả về JSON hợp lệ.")

def gemini_ocr_page(page_png: bytes, page_no: int):
    if not gemini:
        raise RuntimeError("Gemini chưa được khởi tạo.")
    prompt = f"""Đây là trang {page_no} của một sách học tập. Hãy OCR toàn bộ chữ nhìn thấy trên trang, giữ nguyên ngôn ngữ gốc, thứ tự đọc hợp lý và xuống dòng ở tiêu đề/ví dụ. Đồng thời phát hiện các hình minh họa, ảnh, biểu đồ hoặc sơ đồ có ý nghĩa giáo dục. Không coi các vùng chữ thuần túy là hình. Không coi toàn bộ trang là một hình. Với mỗi hình, trả về bounding box chuẩn hóa theo thang 0-1000 theo thứ tự [ymin, xmin, ymax, xmax], kèm mô tả ngắn.

Chỉ trả JSON đúng schema:
{{"text":"...","images":[{{"box":[0,0,1000,1000],"description":"..."}}]}}"""
    part = types.Part.from_bytes(data=page_png, mime_type="image/png")
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[part, prompt],
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    )
    data = _parse_gemini_json(response.text or "{}")
    text = str(data.get("text") or "").strip()
    images = data.get("images") if isinstance(data.get("images"), list) else []
    return text, images

def crop_image_from_page(page_png: bytes, box):
    if Image is None:
        raise RuntimeError("Pillow chưa được cài đặt. Thêm Pillow vào requirements.")
    im = Image.open(io.BytesIO(page_png)).convert("RGB")
    w, h = im.size
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [max(0, min(1000, int(float(x)))) for x in box]
    except Exception:
        return None
    left = int(w * xmin / 1000); top = int(h * ymin / 1000)
    right = int(w * xmax / 1000); bottom = int(h * ymax / 1000)
    if right <= left or bottom <= top:
        return None
    # Avoid tiny false-positive boxes.
    if (right-left) < 80 or (bottom-top) < 80:
        return None
    crop = im.crop((left, top, right, bottom))
    out = io.BytesIO(); crop.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue(), crop.size

def extract_embedded_images(raw_pdf: bytes, page_no: int, source_file: str, subject: str, page_meta):
    """Extract native images from non-scan PDFs with PyMuPDF."""
    if fitz is None:
        return []
    if not b2_ready():
        return []
    doc = fitz.open(stream=raw_pdf, filetype="pdf")
    stored=[]
    try:
        page=doc.load_page(page_no-1)
        seen=set()
        for idx, img in enumerate(page.get_images(full=True), 1):
            xref=img[0]
            if xref in seen:
                continue
            seen.add(xref)
            info=doc.extract_image(xref)
            data=info.get("image")
            ext=info.get("ext","png")
            if not data or len(data)<1000:
                continue
            mime={"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext.lower(), "image/png")
            key=f"images/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}/page_{page_no:04d}/embedded_{idx:02d}.{ext}"
            b2_put_bytes(key,data,mime)
            primary=page_meta[0] if page_meta else {}
            conn=db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO knowledge_images
                        (source_file,subject,content_type,lesson,topic,page,image_key,image_url,description,width,height)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (source_file,subject,primary.get("content_type","Từ vựng"),primary.get("lesson"),primary.get("topic"),
                         page_no,key,b2_url(key),"Embedded PDF image",info.get("width"),info.get("height")))
                conn.commit()
            finally:
                conn.close()
            stored.append({"key":key,"description":"Embedded PDF image","page":page_no})
    finally:
        doc.close()
    return stored

def process_pdf_pages(raw_pdf: bytes, reader, records_meta, source_file: str, subject: str):
    """Extract text/images from normal PDFs and Gemini-OCR scanned pages."""
    page_texts = {}
    page_images = {}
    for page_no, page in enumerate(reader.pages, 1):
        page_meta = metadata_for_page(records_meta, page_no)
        extracted = (page.extract_text() or "").strip()
        text_len = len(re.sub(r"\s+", "", extracted))

        if text_len >= 30:
            page_texts[page_no] = extracted
            embedded = extract_embedded_images(raw_pdf, page_no, source_file, subject, page_meta)
            if embedded:
                page_images[page_no] = embedded
            continue

        # Scan/image page: render it and let Gemini Vision OCR the page and
        # identify meaningful educational images by bounding box.
        png = render_pdf_page(raw_pdf, page_no, dpi=150)
        ocr_text, detected = gemini_ocr_page(png, page_no)
        page_texts[page_no] = ocr_text
        stored = []
        for idx, item in enumerate(detected, 1):
            cropped = crop_image_from_page(png, item.get("box"))
            if not cropped:
                continue
            image_bytes, (width, height) = cropped
            primary = page_meta[0] if page_meta else {}
            key = f"images/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}/page_{page_no:04d}/img_{idx:02d}.jpg"
            b2_put_bytes(key, image_bytes, "image/jpeg")
            description = str(item.get("description") or "").strip()
            conn = db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO knowledge_images
                        (source_file,subject,content_type,lesson,topic,page,image_key,image_url,description,width,height)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (source_file, subject, primary.get("content_type","Từ vựng"), primary.get("lesson"), primary.get("topic"),
                         page_no, key, b2_url(key), description, width, height))
                conn.commit()
            finally:
                conn.close()
            stored.append({"key": key, "description": description, "page": page_no})
        if stored:
            page_images[page_no] = stored
    return page_texts, page_images

@app.post("/admin/api/knowledge/upload")
async def admin_knowledge_upload(
    password: str = Form(""),
    file: UploadFile = File(...),
    subject: str = Form(""),
    metadata_json: str = Form("[]"),
    chunk_size: int = Form(1200),
    overlap: int = Form(200)
):
    check_admin(password)
    subject=subject.strip()
    if not subject:
        raise HTTPException(400, "Môn học là bắt buộc.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Vui lòng chọn file PDF.")
    if not gemini:
        raise HTTPException(500, "GEMINI_API_KEY chưa được cấu hình.")
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    if chunk_size < 300 or chunk_size > 5000:
        raise HTTPException(400, "chunk_size phải từ 300 đến 5000.")
    if overlap < 0 or overlap >= chunk_size:
        raise HTTPException(400, "overlap phải >= 0 và nhỏ hơn chunk_size.")

    raw=await file.read()
    if len(raw)>50*1024*1024:
        raise HTTPException(400, "File quá lớn. Giới hạn 50 MB.")
    try:
        reader=PdfReader(io.BytesIO(raw))
        records_meta=normalize_kb_records(metadata_json,len(reader.pages))
    except ValueError as e:
        raise HTTPException(400,str(e))
    except Exception as e:
        raise HTTPException(400,f"Không đọc được PDF: {e}")

    source_file=os.path.basename(file.filename)
    namespace="__default__"

    # Save original PDF to B2 when configured.
    pdf_key = f"pdf/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}"
    pdf_url = None
    if b2_ready():
        try:
            pdf_url = b2_put_bytes(pdf_key, raw, "application/pdf")
        except Exception as e:
            raise HTTPException(500, f"Không lưu được PDF vào B2: {e}")

    # Extract text normally; scanned/empty pages go through Gemini OCR.
    try:
        page_texts, page_images = process_pdf_pages(raw, reader, records_meta, source_file, subject)
    except Exception as e:
        raise HTTPException(500, f"OCR Gemini/xử lý ảnh thất bại: {e}")

    vectors=[]
    total=0
    try:
        for page_no in range(1, len(reader.pages)+1):
            text = page_texts.get(page_no, "")
            chunks=kb_chunk_text(text,chunk_size,overlap)
            page_meta=metadata_for_page(records_meta,page_no)
            image_keys=[x["key"] for x in page_images.get(page_no, [])]
            for chunk_no,chunk in enumerate(chunks):
                md_list=[{
                    "content_type":r["content_type"],"lesson":r["lesson"],"lesson_pages":r["lesson_pages"],
                    "topic":r["topic"],"topic_pages":r["topic_pages"],
                    "question_pages":r["question_pages"],"answer_pages":r["answer_pages"]
                } for r in page_meta]
                primary=page_meta[0] if page_meta else None
                content_type=primary["content_type"] if primary else (records_meta[0]["content_type"] if records_meta else "Từ vựng")
                md={
                    "text":chunk,"course":subject,"subject":subject,"content_type":content_type,
                    "source_file":source_file,"page":page_no,"chunk_index":chunk_no,
                    "metadata_records":json.dumps(md_list,ensure_ascii=False),
                    "image_keys":json.dumps(image_keys,ensure_ascii=False)
                }
                if primary:
                    md.update({
                        "lesson":primary["lesson"],"lesson_pages":primary["lesson_pages"],
                        "topic":primary["topic"],"topic_pages":primary["topic_pages"],
                        "question_pages":primary["question_pages"],"answer_pages":primary["answer_pages"]
                    })
                vectors.append({"id":uuid.uuid4().hex,"values":embed_text(chunk),"metadata":md})
                total+=1
                if len(vectors)>=50:
                    index.upsert(vectors=vectors,namespace=namespace)
                    vectors=[]
        if vectors:
            index.upsert(vectors=vectors,namespace=namespace)
    except Exception as e:
        raise HTTPException(500,f"Lỗi embedding/Pinecone: {e}")

    conn=db()
    try:
        with conn.cursor() as cur:
            for r in records_meta:
                cur.execute("""INSERT INTO knowledge_documents
                    (source_file,subject,content_type,lesson,lesson_pages,topic,topic_pages,question_pages,answer_pages,namespace)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (source_file,subject,r["content_type"],r["lesson"],r["lesson_pages"],r["topic"],
                     r["topic_pages"],r["question_pages"],r["answer_pages"],namespace))
        conn.commit()
    finally:
        conn.close()

    image_count=sum(len(v) for v in page_images.values())
    scanned_pages=sum(1 for p in range(1,len(reader.pages)+1) if len(re.sub(r"\s+","",(reader.pages[p-1].extract_text() or ""))) < 30)
    return {"success":True,"filename":source_file,"subject":subject,
            "pages":len(reader.pages),"scanned_pages_ocr":scanned_pages,"chunks":total,"records":len(records_meta),
            "images":image_count,"pdf_url":pdf_url,"dimension":768,"index":PINECONE_INDEX,"namespace":namespace}

@app.get("/admin/api/knowledge/images")
def admin_knowledge_images(password: str, source_file: str = ""):
    check_admin(password)
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source_file:
                cur.execute("SELECT * FROM knowledge_images WHERE source_file=%s ORDER BY page,id", (source_file,))
            else:
                cur.execute("SELECT * FROM knowledge_images ORDER BY created_at DESC LIMIT 500")
            rows=[dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    for row in rows:
        row["url"]=b2_url(row.get("image_key"))
    return {"success":True,"images":rows}

@app.get("/admin/api/users")
def admin_users(password: str):
    check_admin(password)
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT u.id,u.phone,u.nickname,u.status,u.created_at,
                    s.id subscription_id,s.plan,s.started_at,s.expires_at,s.status subscription_status
                    FROM users u LEFT JOIN LATERAL
                    (SELECT * FROM subscriptions WHERE user_id=u.id ORDER BY id DESC LIMIT 1) s ON TRUE
                    ORDER BY u.id DESC""")
            rows=cur.fetchall()
    finally: conn.close()
    return {"users":[{"id":r["id"],"phone":r["phone"],"nickname":r["nickname"],"status":r["status"],
        "created_at":r["created_at"],
        "subscription":None if r["subscription_id"] is None else
        {"id":r["subscription_id"],"plan":r["plan"],"started_at":r["started_at"],
         "expires_at":r["expires_at"],"status":r["subscription_status"]}} for r in rows]}

@app.post("/admin/api/users/{user_id}/activate")
def admin_activate(user_id:int,data:dict):
    check_admin(str(data.get("password",""))); months=int(data.get("months",1))
    if months not in (1,3,6,12): raise HTTPException(400,"Thời hạn phải 1, 3, 6 hoặc 12 tháng.")
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE id=%s",(user_id,))
            if not cur.fetchone(): raise HTTPException(404,"Không tìm thấy user.")
            cur.execute("SELECT id,expires_at FROM subscriptions WHERE user_id=%s ORDER BY id DESC LIMIT 1",(user_id,))
            old=cur.fetchone(); now=datetime.now(timezone.utc)
            start=old["expires_at"] if old and old["expires_at"] and old["expires_at"]>now else now
            exp=start+timedelta(days=30*months)
            if old:
                cur.execute("UPDATE subscriptions SET plan=%s,started_at=COALESCE(started_at,%s),expires_at=%s,status='ACTIVE' WHERE id=%s",
                            ("N5",now,exp,old["id"]))
            else:
                cur.execute("INSERT INTO subscriptions(user_id,plan,started_at,expires_at,status) VALUES(%s,'N5',%s,%s,'ACTIVE')",
                            (user_id,now,exp))
            cur.execute("UPDATE users SET status='ACTIVE' WHERE id=%s",(user_id,))
        conn.commit()
    finally: conn.close()
    return {"success":True,"expires_at":exp}



@app.post("/admin/api/chat/send")
def admin_send_chat(data: dict):
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
            cur.execute("""INSERT INTO admin_messages(user_id,sender,message)
                           VALUES(%s,'admin',%s)
                           RETURNING id,user_id,sender,message,created_at,is_read""",
                        (user_id, msg))
            row = dict(cur.fetchone())
        conn.commit()
    finally:
        conn.close()
    return {"message": row}

@app.get("/admin/api/chat/history")
def admin_chat_history(user_id: int, password: str, limit: int = 200, after_id: int = 0):
    check_admin(password)
    limit = max(1, min(limit, 500))
    after_id = max(0, int(after_id or 0))
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if after_id > 0:
                cur.execute("""SELECT id,user_id,sender,message,created_at,is_read
                               FROM admin_messages
                               WHERE user_id=%s AND id>%s
                               ORDER BY id ASC LIMIT %s""",
                            (user_id, after_id, limit))
            else:
                cur.execute("""SELECT id,user_id,sender,message,created_at,is_read
                               FROM admin_messages
                               WHERE user_id=%s
                               ORDER BY id ASC LIMIT %s""",
                            (user_id, limit))
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                cur.execute("""UPDATE admin_messages SET is_read=TRUE
                               WHERE user_id=%s AND sender='user' AND id<=%s""",
                            (user_id, rows[-1]["id"]))
        conn.commit()
    finally:
        conn.close()
    return {"messages": rows, "last_id": rows[-1]["id"] if rows else after_id}

@app.get("/admin/api/ws-token")
def admin_ws_token(password: str):
    check_admin(password)
    if not ADMIN_WS_TOKEN:
        raise HTTPException(500, "ADMIN_WS_TOKEN chưa được cấu hình trên Render.")
    return {"token": ADMIN_WS_TOKEN}

@app.post("/admin/api/users/{user_id}/status")
def admin_status(user_id:int,data:dict):
    check_admin(str(data.get("password",""))); status=str(data.get("status","")).upper()
    if status not in ("ACTIVE","LOCKED","PENDING"): raise HTTPException(400,"Trạng thái không hợp lệ.")
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET status=%s WHERE id=%s",(status,user_id))
            if cur.rowcount==0: raise HTTPException(404,"Không tìm thấy user.")
        conn.commit()
    finally: conn.close()
    return {"success":True,"status":status}


@app.get("/")
def root():
    return {"service":"Doraemon Server","status":"ok","version":SERVER_VERSION}

@app.get("/health")
def health():
    return {"status":"ok","pinecone":index is not None,"gemini":gemini is not None,
            "database":bool(DATABASE_URL),"learning_engine":True,"content_types":sorted(CONTENT_TYPES),"gemini_model":GEMINI_MODEL}
