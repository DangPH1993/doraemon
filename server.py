import os
import ast
import io
import uuid
import re
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional
import json
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
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
SERVER_VERSION = "2026-08-17-doraemon-baseline-v7-packages-free-limit"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pc = None
index = None
gemini = None
connected_users = {}
admin_connections = set()

# Short-lived in-process cache for the learning catalog. The catalog changes only
# when knowledge documents are uploaded, so there is no reason to query the full
# table on every chat request. A short TTL also keeps the cache safe if an admin
# changes data directly in PostgreSQL.
CATALOG_CACHE_TTL = 300.0
_catalog_cache = None
_catalog_cache_at = 0.0
_catalog_cache_lock = threading.Lock()

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
                term TEXT, reading TEXT, meaning TEXT, associated_text TEXT,
                bbox TEXT, width INTEGER, height INTEGER, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS user_learning_state (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                welcome_seen BOOLEAN NOT NULL DEFAULT FALSE,
                reset_count INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );""")
            cur.execute("""CREATE TABLE IF NOT EXISTS daily_question_usage (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                usage_date DATE NOT NULL,
                question_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(user_id, usage_date)
            );""")

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
                "ALTER TABLE knowledge_images ADD COLUMN IF NOT EXISTS term TEXT;",
                "ALTER TABLE knowledge_images ADD COLUMN IF NOT EXISTS reading TEXT;",
                "ALTER TABLE knowledge_images ADD COLUMN IF NOT EXISTS meaning TEXT;",
                "ALTER TABLE knowledge_images ADD COLUMN IF NOT EXISTS associated_text TEXT;",
                "ALTER TABLE knowledge_images ADD COLUMN IF NOT EXISTS bbox TEXT;",
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
            # New accounts (and legacy accounts without a subscription row) receive Free access.
            cur.execute("""INSERT INTO subscriptions(user_id,plan,started_at,expires_at,status)
                           SELECT u.id,'Free',u.created_at,NULL,'ACTIVE'
                           FROM users u
                           WHERE NOT EXISTS (SELECT 1 FROM subscriptions s WHERE s.user_id=u.id)""")
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

def _now_local():
    return datetime.now(ZoneInfo(os.getenv("DAILY_USAGE_TIMEZONE", "Asia/Ho_Chi_Minh")))

def _package_info(user_id):
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id,plan,started_at,expires_at,status FROM subscriptions
                           WHERE user_id=%s ORDER BY id DESC LIMIT 1""", (user_id,))
            sub = cur.fetchone()
            if not sub:
                # Defensive fallback for legacy records before the migration runs.
                sub = {"id": None, "plan": "Free", "started_at": None, "expires_at": None, "status": "ACTIVE"}
            cur.execute("""SELECT question_count FROM daily_question_usage
                           WHERE user_id=%s AND usage_date=%s""", (user_id, _now_local().date()))
            row = cur.fetchone()
            used = int(row["question_count"]) if row else 0
    finally:
        conn.close()

    plan = str(sub.get("plan") or "Free")
    expires_at = sub.get("expires_at")
    active_paid = plan != "Free" and str(sub.get("status") or "").upper() == "ACTIVE" and expires_at and expires_at > datetime.now(timezone.utc)
    if active_paid:
        return {
            "id": sub.get("id"), "plan": plan, "started_at": sub.get("started_at"),
            "expires_at": expires_at, "status": "ACTIVE",
            "daily_limit": None, "used_today": used, "remaining_today": None, "unlimited": True
        }

    return {
        "id": sub.get("id"), "plan": "Free", "started_at": sub.get("started_at"),
        "expires_at": None, "status": "ACTIVE",
        "daily_limit": 5, "used_today": used, "remaining_today": max(0, 5-used), "unlimited": False
    }

def subscription_status(user_id):
    info = _package_info(user_id)
    return {k: info.get(k) for k in ("id","plan","started_at","expires_at","status")}, None

def enforce_question_limit(user_id):
    info = _package_info(user_id)
    if info.get("unlimited"):
        return info
    today = _now_local().date()
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""INSERT INTO daily_question_usage(user_id,usage_date,question_count)
                           VALUES(%s,%s,1)
                           ON CONFLICT(user_id,usage_date) DO UPDATE
                           SET question_count=daily_question_usage.question_count+1
                           RETURNING question_count""", (user_id, today))
            used = int(cur.fetchone()["question_count"])
            if used > 5:
                conn.rollback()
                raise HTTPException(429, detail={
                    "code": "FREE_DAILY_LIMIT",
                    "message": "Gói Free đã dùng hết 5 lượt hỏi hôm nay. Vui lòng thử lại vào ngày mai hoặc nâng cấp gói.",
                    "plan": "Free", "daily_limit": 5, "used_today": 5, "remaining_today": 0
                })
        conn.commit()
    finally:
        conn.close()
    info["used_today"] = used
    info["remaining_today"] = max(0, 5-used)
    return info


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
                           VALUES(%s,%s,%s,'ACTIVE') RETURNING id""",
                        (phone, nickname, hash_password(password)))
            uid = cur.fetchone()[0]
            cur.execute("""INSERT INTO subscriptions(user_id,plan,started_at,expires_at,status)
                           VALUES(%s,'Free',NOW(),NULL,'ACTIVE')""", (uid,))
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "user_id": uid, "status": "ACTIVE",
            "subscription": _package_info(uid),
            "message": "Đăng ký thành công. Bạn đang sử dụng gói Free (5 lượt hỏi/ngày)."}

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
            "subscription": _package_info(user["id"]), "subscription_message": msg}

@app.get("/auth/me")
def me(authorization: Optional[str] = Header(default=None)):
    user = current_user(bearer(authorization))
    sub, msg = subscription_status(user["id"])
    return {"user": user, "subscription": _package_info(user["id"]), "subscription_message": msg}

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
    """
    Persist a user->admin chat message with client-side idempotency.

    The desktop client may retry the same request after a timeout or network
    outage. `client_message_id` makes those retries safe: the same logical
    message is inserted into admin_messages at most once and subsequent retries
    return the original row.
    """
    user = current_user(bearer(authorization))
    msg = str(data.get("message", "")).strip()
    client_message_id = str(data.get("client_message_id", "")).strip()[:128]
    if not msg:
        raise HTTPException(400, "Tin nhắn trống.")
    if not client_message_id:
        raise HTTPException(400, "client_message_id là bắt buộc.")

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_message_dedup (
                    client_message_id VARCHAR(128) PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    admin_message_id INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            cur.execute("""
                SELECT admin_message_id
                FROM admin_message_dedup
                WHERE client_message_id=%s AND user_id=%s
                LIMIT 1
            """, (client_message_id, user["id"]))
            existing = cur.fetchone()
            if existing:
                cur.execute("""
                    SELECT id,user_id,sender,message,created_at,is_read
                    FROM admin_messages WHERE id=%s LIMIT 1
                """, (existing["admin_message_id"],))
                row = dict(cur.fetchone())
                conn.commit()
                return {"message": row, "duplicate": True}

            cur.execute("""INSERT INTO admin_messages(user_id,sender,message)
                           VALUES(%s,'user',%s)
                           RETURNING id,user_id,sender,message,created_at,is_read""",
                        (user["id"], msg))
            row = dict(cur.fetchone())
            cur.execute("""INSERT INTO admin_message_dedup(client_message_id,user_id,admin_message_id)
                           VALUES(%s,%s,%s)""",
                        (client_message_id, user["id"], row["id"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"message": row, "duplicate": False}

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

def _load_catalog_cached():
    """Load the full knowledge catalog once per short TTL instead of per chat."""
    global _catalog_cache, _catalog_cache_at
    now = time.monotonic()
    if _catalog_cache is not None and (now - _catalog_cache_at) < CATALOG_CACHE_TTL:
        return _catalog_cache

    with _catalog_cache_lock:
        now = time.monotonic()
        if _catalog_cache is not None and (now - _catalog_cache_at) < CATALOG_CACHE_TTL:
            return _catalog_cache
        conn = db()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT subject,content_type,lesson,lesson_pages,topic,topic_pages,
                           question_pages,answer_pages,source_file,namespace
                    FROM knowledge_documents
                    ORDER BY subject,content_type,lesson,topic,id
                """)
                _catalog_cache = [dict(x) for x in cur.fetchall()]
                _catalog_cache_at = now
        finally:
            conn.close()
    return _catalog_cache or []

def _invalidate_catalog_cache():
    global _catalog_cache, _catalog_cache_at
    with _catalog_cache_lock:
        _catalog_cache = None
        _catalog_cache_at = 0.0

def _compact_catalog_for_prompt(catalog, active_scope=None, limit=30):
    """Keep the Gemini prompt small without changing routing data."""
    active_scope = active_scope or {}
    ct = active_scope.get("content_type")
    course = _clean_scope_value(active_scope.get("course"))
    lesson = _clean_scope_value(active_scope.get("lesson"))
    topic = _clean_scope_value(active_scope.get("topic"))

    rows = []
    for item in catalog or []:
        item_ct = _normalize_content_type(item.get("content_type"))
        item_course = _clean_scope_value(item.get("course") or item.get("course_name"))
        item_lesson = _clean_scope_value(item.get("lesson"))
        item_topic = _clean_scope_value(item.get("topic"))

        if ct and item_ct != ct:
            continue
        if course and item_course and item_course != course:
            continue
        if lesson and item_lesson and item_lesson != lesson:
            continue
        if topic and item_topic and item_topic != topic:
            continue

        rows.append({
            "subject": item.get("subject"),
            "content_type": item.get("content_type"),
            "lesson": item.get("lesson"),
            "topic": item.get("topic"),
            "lesson_pages": item.get("lesson_pages"),
            "topic_pages": item.get("topic_pages"),
            "question_pages": item.get("question_pages"),
            "answer_pages": item.get("answer_pages"),
            "source_file": item.get("source_file"),
        })
        if len(rows) >= limit:
            break

    # For an entirely generic request, show a compact sample of the catalog
    # rather than dumping every document row into Gemini. Routing itself still
    # uses the full cached catalog above.
    if not rows:
        for item in catalog or []:
            rows.append({
                "subject": item.get("subject"),
                "content_type": item.get("content_type"),
                "lesson": item.get("lesson"),
                "topic": item.get("topic"),
                "lesson_pages": item.get("lesson_pages"),
                "topic_pages": item.get("topic_pages"),
                "question_pages": item.get("question_pages"),
                "answer_pages": item.get("answer_pages"),
                "source_file": item.get("source_file"),
            })
            if len(rows) >= limit:
                break
    return rows

def _save_learning_event_background(user_id, user_text, reply, catalog, learning, source_meta, active_scope):
    """Persist learning progress after the HTTP response has been sent."""
    try:
        event = infer_learning_event(
            user_id, user_text, reply, catalog, learning, source_meta, active_scope=active_scope
        )
        if event:
            record_learning_event(user_id, event)
    except Exception as e:
        print("Learning progress background save skipped:", type(e).__name__, str(e))

def require_active_user(authorization):
    user = current_user(bearer(authorization))
    info = _package_info(user["id"])
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


CONTENT_TYPES = {"Giáo trình", "Từ vựng", "Ngữ pháp", "Bài tập", "Truyện đọc"}


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



def _clean_scope_value(value):
    """Normalize course/content/lesson/topic metadata for matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _metadata_scope(md):
    """Return the hierarchy used to route a learning query."""
    return {
        "course": str(md.get("course") or md.get("course_name") or "").strip(),
        "content_type": _normalize_content_type(md.get("content_type")),
        "lesson": str(md.get("lesson") or "").strip(),
        "topic": str(md.get("topic") or "").strip(),
    }


def _scope_matches(query_text, md, require_lesson=False, require_topic=False):
    """
    Match a document against the requested hierarchy.

    Course -> content_type -> lesson -> topic.
    Empty lesson/topic metadata is allowed for legacy uploads.
    """
    q = _clean_scope_value(query_text)
    scope = _metadata_scope(md)

    course = _clean_scope_value(scope["course"])
    lesson = _clean_scope_value(scope["lesson"])
    topic = _clean_scope_value(scope["topic"])

    if course and course not in q:
        return False

    if require_lesson and lesson and lesson not in q:
        return False

    if require_topic and topic and topic not in q:
        return False

    return True


def _explicit_lesson_topic(query_text, catalog):
    """
    Resolve the user's named lesson/topic BEFORE semantic text retrieval.

    Priority:
      1. exact/longest topic name
      2. exact/longest lesson name
      3. course/content type as supporting scope

    Kanji and Bộ thủ are lesson values under Từ vựng.
    """
    q = _clean_scope_value(query_text)
    if not q:
        return None

    candidates = []
    for item in catalog or []:
        lesson = str(item.get("lesson") or "").strip()
        topic = str(item.get("topic") or "").strip()
        course = str(item.get("course") or item.get("course_name") or "").strip()
        content_type = _normalize_content_type(item.get("content_type"))

        lesson_n = _clean_scope_value(lesson)
        topic_n = _clean_scope_value(topic)
        course_n = _clean_scope_value(course)

        topic_hit = bool(topic_n and (
            topic_n == q or
            re.search(rf"(?<!\w){re.escape(topic_n)}(?!\w)", q) or
            topic_n in q
        ))
        lesson_hit = bool(lesson_n and (
            lesson_n == q or
            re.search(rf"(?<!\w){re.escape(lesson_n)}(?!\w)", q) or
            lesson_n in q
        ))
        course_hit = bool(course_n and course_n in q)

        if not topic_hit and not lesson_hit:
            continue

        # Topic is more specific than lesson. Longer names win ties.
        score = 0
        if topic_hit:
            score += 1000 + len(topic_n) * 10
        if lesson_hit:
            score += 500 + len(lesson_n) * 5
        if course_hit:
            score += 50

        candidates.append((score, item))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]



def _focus_metadata_matches(matches, query_text, lesson=None, topic=None,
                            content_type=None):
    """
    Content-type-aware second-stage selection.

    The first stage scopes by lesson/topic. The second stage behaves differently:
      - Vocabulary with a specific item (reading/term/meaning such as "Bộ Vi"):
        narrow to that item.
      - Reading/Grammar/Exercise: keep all chunks belonging to the identified
        lesson/topic. These are multi-chunk learning materials and must not be
        collapsed to one chunk merely because one word appears in the query.

    Images remain locked to the selected chunk later in the pipeline.
    """
    if not matches:
        return []

    ct = _clean_scope_value(content_type)
    # "Từ vựng" is the only content type where lesson names such as Bộ thủ or
    # Kanji represent individual vocabulary items.
    is_vocabulary = ct in {"từ vựng", "tu vung", "vocabulary"}

    if not is_vocabulary:
        return list(matches)

    q = _clean_scope_value(query_text)

    generic = {
        "bộ thủ", "kanji", "từ vựng", "từ mới", "học", "học về", "cho tôi biết",
        "giải thích", "là gì", "nghĩa là gì", "ý nghĩa", "thông tin",
        "có nghĩa gì", "hãy dạy", "dạy", "về", "của", "cho", "biết",
    }
    q_tokens = [
        t for t in re.findall(r"[\wÀ-ỹ一-龥ぁ-んァ-ンー]+", q)
        if t not in generic
    ]

    scored = []
    for idx, m in enumerate(matches):
        md = m.metadata or {}

        fields = {
            "topic": _clean_scope_value(md.get("topic")),
            "reading": _clean_scope_value(md.get("reading")),
            "term": _clean_scope_value(md.get("term")),
            "meaning": _clean_scope_value(md.get("meaning")),
            "associated_text": _clean_scope_value(md.get("associated_text")),
        }

        item_fields = [
            fields["topic"], fields["reading"], fields["term"],
            fields["meaning"], fields["associated_text"]
        ]

        score = 0
        exact_hits = 0

        for value in item_fields:
            if not value:
                continue
            if value in q or q in value:
                exact_hits += 1
                score = max(score, 1000 + len(value))

        for token in q_tokens:
            if len(token) < 2:
                continue
            for value in item_fields:
                if value and (token == value or token in value):
                    exact_hits += 1
                    score = max(score, 900 + len(token))

        if exact_hits:
            score += exact_hits * 50

        scored.append((score, float(getattr(m, "score", 0) or 0), idx, m))

    strong = [x for x in scored if x[0] >= 900]
    if not strong:
        return list(matches)

    strong.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score = strong[0][0]
    focused = [x[3] for x in strong if x[0] == best_score]

    print(
        "[RAG item-focus] "
        f"content_type={content_type!r} lesson={lesson!r} topic={topic!r} "
        f"query={query_text!r} candidates={len(matches)} "
        f"selected={len(focused)} best_score={best_score}"
    )
    return focused



def _select_active_scope(query_text, text_matches, catalog):
    """
    Determine the active learning scope in strict order:
    Course -> content type -> lesson -> topic.

    Kanji and Bộ thủ are LESSONS under content_type="Từ vựng",
    not content types themselves.
    """
    q = _clean_scope_value(query_text)

    # Course
    course = None
    for item in catalog or []:
        c = str(item.get("course") or item.get("course_name") or "").strip()
        if c and _clean_scope_value(c) in q:
            course = c
            break

    # Content type: five peer content types.
    explicit_type = None
    explicit_patterns = [
        ("Giáo trình", ["giáo trình", "học theo giáo trình", "học giáo trình", "theo giáo trình", "trong giáo trình"]),
        ("Truyện đọc", ["truyện đọc", "đọc truyện", "câu chuyện", "học truyện"]),
        ("Bài tập", ["bài tập", "làm bài", "bài quiz", "quiz"]),
        ("Ngữ pháp", ["ngữ pháp", "học ngữ pháp", "ôn ngữ pháp", "grammar"]),
        ("Từ vựng", ["từ vựng", "học từ vựng", "từ mới", "học từ mới", "vocabulary"]),
    ]
    for typ, keys in explicit_patterns:
        if any(k in q for k in keys):
            explicit_type = typ
            break

    # Lesson/topic from catalog.
    # "Kanji" / "Bộ thủ" are matched here as lesson values.
    lesson = None
    topic = None
    best_score = -1
    for item in catalog or []:
        item_course = str(item.get("course") or item.get("course_name") or "").strip()
        item_type = _normalize_content_type(item.get("content_type"))
        item_lesson = str(item.get("lesson") or "").strip()
        item_topic = str(item.get("topic") or "").strip()

        score = 0
        if item_course and _clean_scope_value(item_course) in q:
            score += 10

        if explicit_type and item_type == explicit_type:
            score += 20

        if item_lesson and _clean_scope_value(item_lesson) in q:
            score += 40
            # A named lesson is authoritative. If it is Kanji/Bộ thủ,
            # its content type must come from the catalog and therefore be Từ vựng.
            if not explicit_type:
                explicit_type = item_type

        if item_topic and _clean_scope_value(item_topic) in q:
            score += 50
            if not explicit_type:
                explicit_type = item_type

        if score > best_score:
            best_score = score
            lesson = item_lesson or lesson
            topic = item_topic or topic
            if score > 0:
                if item_course and not course:
                    course = item_course
                if item_type and not explicit_type:
                    explicit_type = item_type

    # If the user explicitly names Kanji/Bộ thủ but the catalog has not
    # provided a matching lesson row, still route them as Từ vựng lessons.
    # This prevents RAG similarity from selecting another content type.
    if any(k in q for k in ["kanji", "học kanji"]):
        if not lesson:
            lesson = "Kanji"
        explicit_type = "Từ vựng"
    elif any(k in q for k in ["bộ thủ", "học bộ thủ", "radical"]):
        if not lesson:
            lesson = "Bộ thủ"
        explicit_type = "Từ vựng"

    # If no explicit routing was possible, use top RAG metadata as fallback.
    if not explicit_type:
        for m in text_matches or []:
            md = m.metadata or {}
            typ = _normalize_content_type(md.get("content_type"))
            if typ:
                explicit_type = typ
                if not course:
                    course = str(md.get("course") or md.get("course_name") or "").strip() or None
                if not lesson:
                    lesson = str(md.get("lesson") or "").strip() or None
                if not topic:
                    topic = str(md.get("topic") or "").strip() or None
                break

    return {
        "course": course,
        "content_type": explicit_type,
        "lesson": lesson,
        "topic": topic,
    }


def infer_learning_event(user_id, user_text, reply, catalog, learning, source_meta=None, active_scope=None):
    """Infer only learning progress, never a score. Exercises are scored via /learning/progress."""
    text = (user_text or "").strip()
    low = text.lower()
    source_meta = source_meta or []
    active_scope = active_scope or {}
    active_content_type = active_scope.get("content_type")
    active_course = active_scope.get("course")
    active_lesson = active_scope.get("lesson")
    active_topic = active_scope.get("topic")

    chosen = None

    # IMPORTANT:
    # Learning progress must follow the user's explicit learning intent first.
    # RAG ranking is NOT reliable enough to decide whether the user is studying
    # vocabulary or grammar. For example, a vocabulary request may retrieve a
    # grammar page because the two documents share words/context.
    explicit_type = None
    explicit_patterns = [
        ("Giáo trình", [
            "giáo trình", "học theo giáo trình", "học giáo trình",
            "theo giáo trình", "trong giáo trình"
        ]),
        ("Truyện đọc", [
            "truyện đọc", "đọc truyện", "câu chuyện", "học truyện",
            "muốn đọc truyện", "học truyện"
        ]),
        ("Bài tập", [
            "bài tập", "làm bài", "làm bài tập", "bài quiz", "quiz",
            "bài kiểm tra"
        ]),
        ("Ngữ pháp", [
            "ngữ pháp", "học ngữ pháp", "ôn ngữ pháp", "grammar"
        ]),
        ("Từ vựng", [
            "từ vựng", "học từ vựng", "ôn từ vựng", "từ mới",
            "học từ mới", "vocabulary"
        ]),
    ]

    for typ, keys in explicit_patterns:
        if any(k in low for k in keys):
            explicit_type = typ
            break

    # Kanji and Bộ thủ are lessons under Từ vựng, never content types.
    # If explicitly requested, lock the event to Từ vựng and the corresponding lesson.
    explicit_lesson = None
    if any(k in low for k in ["kanji", "học kanji"]):
        explicit_type = "Từ vựng"
        explicit_lesson = "Kanji"
    elif any(k in low for k in ["bộ thủ", "học bộ thủ", "radical"]):
        explicit_type = "Từ vựng"
        explicit_lesson = "Bộ thủ"

    if explicit_lesson:
        chosen = {
            "content_type": "Từ vựng",
            "subject": "Tiếng Nhật",
            "lesson": explicit_lesson,
        }

    # Phrases such as "chỉ học từ vựng" are an even stronger signal.
    # Keep the explicit type and never let RAG override it.
    if explicit_type:
        chosen = {"content_type": explicit_type, "subject": "Tiếng Nhật"}

        # If the user explicitly named a lesson/topic in the catalog, attach it
        # to the chosen type only when the catalog item has the same type.
        for item in catalog:
            item_type = _normalize_content_type(item.get("content_type"))
            if item_type != explicit_type:
                continue
            hay = " ".join(
                str(item.get(k) or "")
                for k in ("lesson", "topic", "subject")
            ).strip()
            parts = [
                str(item.get("lesson") or ""),
                str(item.get("topic") or ""),
            ]
            if hay and any(part and part.lower() in low for part in parts):
                chosen = item
                break

    # If a hierarchy was identified, choose a catalog item only from that
    # exact scope. Never let a generic/high-similarity Kanji item override an
    # explicit Bộ thủ lesson request.
    if not chosen and (active_content_type or active_lesson or active_topic):
        for item in catalog:
            item_type = _normalize_content_type(item.get("content_type"))
            item_course = str(item.get("course") or item.get("course_name") or "").strip()
            item_lesson = str(item.get("lesson") or "").strip()
            item_topic = str(item.get("topic") or "").strip()

            if active_content_type and item_type != active_content_type:
                continue
            if active_course and item_course and _clean_scope_value(item_course) != _clean_scope_value(active_course):
                continue
            if active_lesson and item_lesson and _clean_scope_value(item_lesson) != _clean_scope_value(active_lesson):
                continue
            if active_topic and item_topic and _clean_scope_value(item_topic) != _clean_scope_value(active_topic):
                continue

            chosen = item
            break

    # If the user did NOT explicitly choose a content type/lesson/topic, then
    # use catalog/source metadata as a fallback.
    if not chosen:
        for item in catalog:
            hay = " ".join(
                str(item.get(k) or "")
                for k in ("lesson", "topic", "subject")
            ).strip()
            if hay and any(
                part.lower() in low
                for part in [
                    str(item.get("lesson") or ""),
                    str(item.get("topic") or "")
                ] if part
            ):
                chosen = item
                break

    if not chosen:
        for md in source_meta:
            if md.get("content_type") in CONTENT_TYPES:
                chosen = md
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

def _find_phrase_position(text: str, phrase: str):
    """Find a meaningful exact occurrence, avoiding tiny Latin substring matches."""
    text = text or ""
    phrase = str(phrase or "").strip()
    if not phrase:
        return None

    # For Latin/number-only phrases, require word boundaries.
    # This prevents a term such as "Vi" from matching inside unrelated
    # Vietnamese words such as "vì".
    if re.fullmatch(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 _-]*", phrase):
        pattern = re.compile(
            rf"(?<![A-Za-zÀ-ỹ0-9]){re.escape(phrase)}(?![A-Za-zÀ-ỹ0-9])",
            re.IGNORECASE
        )
        match = pattern.search(text)
        return match.start() if match else None

    # Japanese/CJK and mixed strings: exact case-insensitive substring match.
    pos = text.casefold().find(phrase.casefold())
    return pos if pos >= 0 else None


def _find_structural_phrase_position(text: str, phrase: str):
    """
    Find a term only in a structural/title part of Doraemon's answer.

    We do NOT search the whole answer: a term such as "Bao" can appear inside
    the explanation of another radical, e.g. "Ý nghĩa: bao quanh". That must
    not cause the image for Bộ Bao to be inserted while teaching Bộ Vi.
    """
    text = text or ""
    phrase = str(phrase or "").strip()
    if not text or not phrase:
        return None

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        is_heading = bool(re.match(r"^#{1,6}\s+", stripped))
        is_bold_title = stripped.startswith("**") and stripped.endswith("**")

        if is_heading or is_bold_title:
            pos = _find_phrase_position(line, phrase)
            if pos is not None:
                return offset + pos

        offset += len(line)

    return None


def _find_explicit_term_in_query(query: str, phrase: str):
    """
    Match a term in the user's query only when it is explicitly named.

    This prevents a Vietnamese meaning such as "bao quanh" from being
    interpreted as a request for the vocabulary/radical "Bao".
    """
    query = query or ""
    phrase = str(phrase or "").strip()
    if not query or not phrase:
        return None

    q = query.casefold()
    p = phrase.casefold()

    patterns = [
        rf"\bbộ\s+{re.escape(p)}\b",
        rf"\bbộ\s+thủ\s*[:：-]?\s*{re.escape(p)}\b",
        rf"\btừ\s+{re.escape(p)}\b",
        rf"\bkanji\s*[:：-]?\s*{re.escape(p)}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            pos = q.find(p, m.start())
            return pos if pos >= 0 else m.start()

    # Japanese/Chinese characters are distinctive enough to be explicit.
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff]", phrase):
        return _find_phrase_position(query, phrase)

    return None



def _parse_image_keys(raw):
    """Return normalized image keys from text-chunk metadata."""
    if raw is None:
        return []
    values = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            values.extend(_parse_image_keys(item))
        return list(dict.fromkeys(values))
    if isinstance(raw, dict):
        for field in ("key", "image_key", "image_keys", "path"):
            if field in raw:
                values.extend(_parse_image_keys(raw.get(field)))
        return list(dict.fromkeys(values))

    value = str(raw).strip()
    if not value:
        return []
    if value.startswith("[") or value.startswith("{"):
        try:
            return _parse_image_keys(json.loads(value))
        except Exception:
            try:
                return _parse_image_keys(ast.literal_eval(value))
            except Exception:
                pass
    value = value.strip().strip('"').strip("'")
    return [value] if value else []


def _split_exercise_logical_chunks(text: str):
    """
    Split a retrieved Bài tập chunk into logical sub-chunks when a single
    Pinecone text record contains multiple customer/order blocks.

    This is intentionally deterministic and text-only. It never uses image
    similarity or image metadata to decide where a text chunk starts/ends.
    The splitter is conservative: when no clear boundary is found, the
    original text remains one chunk.
    """
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    # First prefer blank-line separated blocks. OCR/PDF extraction for
    # conversation-style exercises commonly preserves this structure.
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw) if b.strip()]
    if len(blocks) > 1 and len(blocks) <= 20:
        return blocks

    # Then detect common customer/speaker/order headers. The lookahead keeps
    # the header attached to the following text.
    header_re = re.compile(
        r"(?im)^(?=\s*(?:"
        r"khách(?:\s*hàng)?(?:\s*(?:số\s*)?\d+)?\s*[:：.-]?|"
        r"kh\s*\d+\s*[:：.-]|"
        r"người\s*(?:khách|mua)\s*(?:số\s*)?\d+\s*[:：.-]?|"
        r"[A-H]\s*[:：.-]|"
        r"お客(?:さん|様)?(?:\s*\d+)?\s*[:：.-]?|"
        r"客\s*\d+\s*[:：.-]|"
        r"[①-⑩]"
        r"))"
    )
    starts = [m.start() for m in header_re.finditer(raw)]
    if len(starts) > 1:
        chunks = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(raw)
            part = raw[start:end].strip()
            if part:
                chunks.append(part)
        if 1 < len(chunks) <= 20:
            return chunks

    return [raw]


def _chunk_identity(md):
    """
    Identity of one RAG text chunk.

    chunk_index is intentionally included when present. We never use a
    page-level image as a substitute for a different chunk on the same page.
    """
    source_file = str(md.get("source_file") or "").strip()
    page = str(md.get("page") or "").strip()
    raw_chunk = md.get("chunk_index")
    chunk_index = None
    if raw_chunk not in (None, ""):
        try:
            chunk_index = int(raw_chunk)
        except Exception:
            chunk_index = str(raw_chunk).strip()
    return source_file, page, chunk_index


def _normalize_chunk_text_for_match(value):
    value = str(value or "").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


def _image_belongs_to_text_chunk(md, chunk_md, chunk_text):
    """
    Strict text-chunk -> image-chunk association.

    Priority:
      1) exact source_file + page + chunk_index;
      2) legacy records with no chunk_index: exact OCR/associated text;
      3) legacy vocabulary records: exact source_file + page + lesson and
         exact reading/term/meaning identity.

    IMPORTANT:
      We never use page-only, lesson-only, or vector similarity to attach an
      image. The vocabulary identity fallback exists because the older
      Pinecone uploader stored image records without chunk_index. For example,
      a Bộ Vi image can have reading="Vi", meaning="Vây quanh", lesson="Bộ thủ"
      while the corresponding text chunk has the same identity.
    """
    sf, page, chunk_index = _chunk_identity(chunk_md)
    img_sf = str(md.get("source_file") or "").strip()
    img_page = str(md.get("page") or "").strip()

    if not sf or not page or img_sf != sf or img_page != page:
        return False

    raw_img_chunk = md.get("chunk_index")
    img_chunk_index = None
    if raw_img_chunk not in (None, ""):
        try:
            img_chunk_index = int(raw_img_chunk)
        except Exception:
            img_chunk_index = str(raw_img_chunk).strip()

    chunk_type = _normalize_chunk_text_for_match(chunk_md.get("content_type"))
    is_exercise = chunk_type == "bài tập"
    is_virtual_exercise = bool(chunk_md.get("_virtual_exercise_split"))

    # For exercise material, associated_text is the safest bridge when older
    # image records were created before per-chunk image_index metadata existed.
    # It MUST win over a stale/shared chunk_index (for example, several images
    # accidentally carrying chunk_index=0 on the same page).
    img_associated = _normalize_chunk_text_for_match(md.get("associated_text"))
    chunk_norm = _normalize_chunk_text_for_match(chunk_text)
    if is_exercise and img_associated and chunk_norm:
        if img_associated == chunk_norm:
            return True
        if len(img_associated) >= 20 and len(chunk_norm) >= 20:
            shorter, longer = sorted((img_associated, chunk_norm), key=len)
            if shorter in longer and len(shorter) / len(longer) >= 0.65:
                return True
        # An explicitly associated exercise image that does not match this
        # logical customer/order must never fall through to page/chunk matching.
        return False

    # Strongest identity: both sides have chunk_index and it is identical.
    if chunk_index is not None:
        if img_chunk_index is not None:
            if is_virtual_exercise:
                # Virtual exercise chunks are mapped by associated_text above.
                # A stale shared index is not enough.
                return False
            return img_chunk_index == chunk_index
    elif img_chunk_index is not None:
        # A chunked image cannot be attached to an unchunked text record.
        return False

    # Legacy exact text identity.
    img_text = _normalize_chunk_text_for_match(
        md.get("associated_text") or md.get("text") or md.get("content")
    )
    if img_text and chunk_norm:
        if img_text == chunk_norm:
            return True

        # OCR can differ in whitespace/punctuation. Keep this conservative:
        # only accept a substantial containment relationship.
        if len(img_text) >= 20 and len(chunk_norm) >= 20:
            shorter, longer = sorted((img_text, chunk_norm), key=len)
            if shorter in longer and len(shorter) / len(longer) >= 0.65:
                return True

    # Legacy vocabulary identity. This is the missing case for records such
    # as Bộ Vi where image records have no chunk_index.
    def norm_field(value):
        return _normalize_chunk_text_for_match(value)

    img_lesson = norm_field(md.get("lesson"))
    chunk_lesson = norm_field(chunk_md.get("lesson"))
    if img_lesson and chunk_lesson and img_lesson != chunk_lesson:
        return False

    # IMPORTANT: no page+lesson fallback for Bài tập.
    # When chunk_index is unavailable, an exercise image must have exact/strong
    # associated_text identity; otherwise it is not safe to attach.

    identity_pairs = [
        ("reading", "reading"),
        ("term", "term"),
        ("meaning", "meaning"),
    ]

    matched = 0
    available = 0
    for img_field, chunk_field in identity_pairs:
        img_value = norm_field(md.get(img_field))
        chunk_value = norm_field(chunk_md.get(chunk_field))
        if not img_value or not chunk_value:
            continue
        available += 1
        if img_value == chunk_value:
            matched += 1

    # At least one exact vocabulary identity must match, and when both
    # reading+meaning (or term+reading) exist they must not conflict.
    if matched >= 1:
        for img_field, chunk_field in identity_pairs:
            img_value = norm_field(md.get(img_field))
            chunk_value = norm_field(chunk_md.get(chunk_field))
            if img_value and chunk_value and img_value != chunk_value:
                return False
        return True

    return False

def _same_chunk_image(md, chunk_md):
    """Strictly determine whether an image record belongs to a text chunk."""
    sf, page, chunk_index = _chunk_identity(chunk_md)
    if not sf or not page:
        return False
    if str(md.get("source_file") or "").strip() != sf:
        return False
    if str(md.get("page") or "").strip() != page:
        return False

    # If the text chunk has chunk_index, the image MUST have the same one.
    # This is the core rule: no page-only fallback can leak another chunk's image.
    if chunk_index is not None:
        raw_img_chunk = md.get("chunk_index")
        if raw_img_chunk in (None, ""):
            return False
        try:
            img_chunk = int(raw_img_chunk)
        except Exception:
            img_chunk = str(raw_img_chunk).strip()
        return img_chunk == chunk_index

    # Legacy text chunks without chunk_index can only match image records that
    # also lack chunk_index. This prevents accidentally attaching a chunk-0 image
    # to an unrelated legacy chunk on the same page.
    return md.get("chunk_index") in (None, "")


def _image_payload_from_metadata(md, score=0.0, chunk_order=None, chunk_text=""):
    keys = _parse_image_keys(md.get("image_key"))
    if not keys:
        keys = _parse_image_keys(md.get("image_keys"))
    if not keys:
        return []

    payloads = []
    for key in keys:
        url = b2_url(key) or md.get("image_url")
        if not url:
            continue
        payloads.append({
            "key": key,
            "url": url,
            "term": str(md.get("term") or "").strip(),
            "reading": str(md.get("reading") or "").strip(),
            "meaning": str(md.get("meaning") or "").strip(),
            "page": md.get("page"),
            "source_file": str(md.get("source_file") or "").strip(),
            "score": float(score or 0),
            "_chunk_order": chunk_order,
            "_chunk_text": chunk_text,
            "_chunk_key": (
                str(md.get("source_file") or "").strip(),
                str(md.get("page") or "").strip(),
                str(md.get("chunk_index") if md.get("chunk_index") not in (None, "") else ""),
            ),
        })
    return payloads


def _retrieve_images_for_text_chunks(text_chunks, index, namespace, query_vector):
    """
    The only image retrieval path.

    For every text chunk actually selected into RAG:
      1. use image_key/image_keys directly if present on the chunk;
      2. otherwise query Pinecone IMAGE records by the exact same
         source_file + page + chunk_index;
      3. if no exact match exists, return no image.

    We deliberately do NOT run a semantic image query. This prevents an image
    from another chunk/lesson/page from being injected just because its vector
    happens to be similar to the user's question.
    """
    if not text_chunks or not index:
        return []

    results = []
    jobs = []

    # Direct image keys are strongest and require no Pinecone round trip.
    for order, chunk in enumerate(text_chunks):
        md = chunk["metadata"]
        chunk_text = chunk["text"]
        direct_keys = _parse_image_keys(md.get("image_key")) + _parse_image_keys(md.get("image_keys"))
        direct_keys = list(dict.fromkeys(direct_keys))
        if direct_keys:
            direct_md = dict(md)
            direct_md["image_key"] = direct_keys
            direct_payload = _image_payload_from_metadata(
                direct_md, chunk.get("score", 0), order, chunk_text
            )
            print(
                "[IMAGE direct] "
                f"order={order} keys={len(direct_keys)} urls={len(direct_payload)}"
            )
            results.extend(direct_payload)
            continue

        sf, page, chunk_index = _chunk_identity(md)
        if sf and page:
            jobs.append((order, chunk, sf, page, chunk_index))

    def fetch_exact(job):
        order, chunk, sf, page, chunk_index = job
        filt = {
            "record_type": {"$eq": "image"},
            "source_file": {"$eq": sf},
            "page": {"$eq": int(page) if str(page).isdigit() else page},
        }
        is_exercise = _normalize_chunk_text_for_match(chunk["metadata"].get("content_type")) == "bài tập"
        is_virtual_exercise = bool(chunk["metadata"].get("_virtual_exercise_split"))
        if chunk_index is not None and not (is_exercise and is_virtual_exercise):
            filt["chunk_index"] = {"$eq": chunk_index}

        # IMPORTANT: do not add reading/term/meaning as Pinecone filter fields
        # here. Older image records are not schema-identical to text records
        # (some omit term or chunk_index). Query the exact source/page and let
        # _image_belongs_to_text_chunk perform the strict identity check in
        # Python. This avoids false zero-result filters for Bộ Vi/Kanji.
        try:
            res = index.query(
                vector=query_vector,
                top_k=50 if is_exercise else 20,
                include_metadata=True,
                namespace=namespace,
                filter=filt,
            )
            # Legacy image records may not have record_type="image". Retry the
            # exact same source/page identity without record_type.
            if not res.matches:
                legacy_filt = {
                    "source_file": {"$eq": sf},
                    "page": {"$eq": int(page) if str(page).isdigit() else page},
                }
                res = index.query(
                    vector=query_vector,
                    top_k=50 if is_exercise else 20,
                    include_metadata=True,
                    namespace=namespace,
                    filter=legacy_filt,
                )
            found = []
            rejected = 0
            for m in res.matches:
                md = m.metadata or {}
                if not _image_belongs_to_text_chunk(md, chunk["metadata"], chunk["text"]):
                    rejected += 1
                    continue
                payload = _image_payload_from_metadata(
                    md, getattr(m, "score", 0), order, chunk["text"]
                )
                found.extend(payload)

            print(
                "[IMAGE chunk-exact] "
                f"order={order} source={sf!r} page={page!r} chunk_index={chunk_index!r} "
                f"lesson={chunk['metadata'].get('lesson')!r} "
                f"content_type={chunk['metadata'].get('content_type')!r} "
                f"candidates={len(res.matches)} accepted={len(found)} rejected={rejected}"
            )
            return found
        except Exception as exc:
            print("[IMAGE chunk-exact] failed:", sf, page, chunk_index, type(exc).__name__, str(exc))
            return []

    if jobs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as executor:
            for found in executor.map(fetch_exact, jobs):
                results.extend(found)

    # Deduplicate only by actual object key, never by page/chunk position.
    unique = []
    seen = set()
    for item in results:
        key = item.get("key")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique

def build_rich_content_blocks(reply: str, image_items: list) -> list:
    """
    Insert images only for the RAG chunks that supplied the answer.

    Gemini receives an internal marker instruction such as [[IMG_CHUNK_2]].
    The marker is removed before returning the reply and becomes the exact
    insertion point for images belonging to that chunk.

    If Gemini omits a marker, that chunk's image is appended after the nearest
    answer text rather than being attached to a different chunk.
    """
    reply = reply or ""
    if not reply:
        return []

    marker_re = re.compile(r"\[\[IMG_CHUNK_(\d+)\]\]")
    chunks = {}
    for item in image_items:
        order = item.get("_chunk_order")
        if order is None:
            continue
        chunks.setdefault(int(order), []).append(item)

    # First use explicit markers. This is deterministic and does not depend on
    # term/meaning words accidentally appearing in the answer.
    marker_positions = []
    for m in marker_re.finditer(reply):
        marker_positions.append((m.start(), int(m.group(1)), m.end()))

    # Remove markers from user-visible text.
    clean_reply = marker_re.sub("", reply)

    blocks = []
    cursor = 0
    inserted_keys = set()

    # Convert positions from original reply to clean-reply positions.
    removed_before = 0
    for pos, chunk_order, marker_end in marker_positions:
        clean_pos = pos - removed_before
        removed_before += marker_end - pos

        if clean_pos > cursor:
            blocks.append({"type": "text", "text": clean_reply[cursor:clean_pos]})

        for item in chunks.get(chunk_order, []):
            key = item.get("key")
            if not key or key in inserted_keys:
                continue
            blocks.append({
                "type": "image",
                "key": key,
                "url": item.get("url"),
                "term": item.get("term", ""),
                "reading": item.get("reading", ""),
                "meaning": item.get("meaning", ""),
                "page": item.get("page"),
            })
            inserted_keys.add(key)
        cursor = clean_pos

    # If Gemini did not emit a marker, fall back to chunk-order placement:
    # put each unmarked chunk's images after the answer text, in the same order
    # as the retrieved chunks. Never borrow an image from another chunk.
    if not marker_positions:
        if clean_reply:
            blocks.append({"type": "text", "text": clean_reply})
        for order in sorted(chunks):
            for item in chunks[order]:
                key = item.get("key")
                if not key or key in inserted_keys:
                    continue
                blocks.append({
                    "type": "image",
                    "key": key,
                    "url": item.get("url"),
                    "term": item.get("term", ""),
                    "reading": item.get("reading", ""),
                    "meaning": item.get("meaning", ""),
                    "page": item.get("page"),
                })
                inserted_keys.add(key)
        return blocks

    if cursor < len(clean_reply):
        blocks.append({"type": "text", "text": clean_reply[cursor:]})

    # If some image chunks were not marked, append only those exact chunk images.
    marked_orders = {order for _, order, _ in marker_positions}
    for order in sorted(chunks):
        if order in marked_orders:
            continue
        for item in chunks[order]:
            key = item.get("key")
            if not key or key in inserted_keys:
                continue
            blocks.append({
                "type": "image",
                "key": key,
                "url": item.get("url"),
                "term": item.get("term", ""),
                "reading": item.get("reading", ""),
                "meaning": item.get("meaning", ""),
                "page": item.get("page"),
            })
            inserted_keys.add(key)

    return blocks


_GREETING_EXACT = {
    "chào", "chào bạn", "chào cậu", "chào doraemon",
    "xin chào", "xin chào bạn", "xin chào doraemon",
    "hello", "hello doraemon", "hi", "hi doraemon",
    "hey", "hey doraemon", "alo", "alo doraemon",
    "doraemon ơi", "doraemon ơi chào",
}

def _is_pure_greeting(text: str) -> bool:
    """True only for a standalone greeting, not for a greeting + request."""
    low = str(text or "").strip().casefold()
    if not low:
        return False
    normalized = re.sub(r"[\W_]+", " ", low, flags=re.UNICODE).strip()
    return normalized in _GREETING_EXACT


def _build_welcome_for_user(user, mark_seen: bool = False):
    """
    Build the same concise onboarding/returning-user message for both
    /session/welcome and a standalone 'Chào' sent through /api/proxy-chat.

    Important:
    - The curriculum is always shown.
    - Only unfinished / reviewable learning is shown.
    - Completed learning is NOT presented as 'đang học dở'.
    - No Gemini/Pinecone call is needed for a greeting.
    """
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT welcome_seen, reset_count
                FROM user_learning_state
                WHERE user_id=%s
            """, (user["id"],))
            state = cur.fetchone()

            cur.execute("""
                SELECT content_type, lesson, topic, status, current_position,
                       current_page, score, last_studied_at
                FROM learning_progress
                WHERE user_id=%s
                  AND LOWER(COALESCE(status,'')) IN
                      ('in_progress','active','review','needs_review')
                ORDER BY
                    CASE
                        WHEN LOWER(COALESCE(status,'')) IN ('in_progress','active') THEN 0
                        ELSE 1
                    END,
                    last_studied_at DESC
                LIMIT 12
            """, (user["id"],))
            unfinished_rows = [dict(r) for r in cur.fetchall()]

            all_progress_exists = False
            cur.execute("""
                SELECT 1
                FROM learning_progress
                WHERE user_id=%s
                LIMIT 1
            """, (user["id"],))
            all_progress_exists = cur.fetchone() is not None

            if state is None:
                is_new = not all_progress_exists
                if mark_seen:
                    cur.execute("""
                        INSERT INTO user_learning_state(user_id,welcome_seen,reset_count)
                        VALUES(%s,TRUE,0)
                    """, (user["id"],))
            else:
                # A reset explicitly sets welcome_seen=False and should behave
                # like a fresh learner. Otherwise an account with no progress
                # is still new, while an account with progress is returning.
                is_new = (not bool(state["welcome_seen"])) and not all_progress_exists
                if not all_progress_exists and not bool(state["welcome_seen"]):
                    is_new = True
                elif all_progress_exists:
                    is_new = False

                if mark_seen:
                    cur.execute("""
                        UPDATE user_learning_state
                        SET welcome_seen=TRUE, updated_at=NOW()
                        WHERE user_id=%s
                    """, (user["id"],))
        if mark_seen:
            conn.commit()
    finally:
        conn.close()

    nickname = user.get("nickname") or "bạn"

    curriculum = (
        "📚 Doraemon hỗ trợ 5 loại nội dung:\n"
        "1. Giáo trình\n"
        "2. Ngữ pháp\n"
        "3. Bài tập\n"
        "4. Từ vựng\n"
        "   • Kanji và Bộ thủ là các lesson bên trong Từ vựng\n"
        "5. Truyện đọc"
    )

    if is_new:
        message = (
            f"Chào {nickname}! 👋 Tớ là Doraemon, gia sư tiếng Nhật của cậu. 🤖\n\n"
            f"{curriculum}\n\n"
            "Cậu muốn bắt đầu học phần nào? Nếu chưa biết nên bắt đầu từ đâu, "
            "tớ có thể gợi ý lộ trình cho cậu nhé! 😊"
        )
        return {
            "success": True,
            "mode": "new",
            "message": message,
            "learning_history": [],
        }

    parts = []
    seen = set()
    for row in unfinished_rows:
        key = (
            row.get("content_type"),
            row.get("lesson"),
            row.get("topic"),
        )
        if key in seen:
            continue
        seen.add(key)

        label = str(row.get("content_type") or "Nội dung")
        detail = " ".join(
            str(x).strip()
            for x in (row.get("lesson"), row.get("topic"))
            if x and str(x).strip()
        )
        status = str(row.get("status") or "").strip().lower()
        page = row.get("current_page")
        position = row.get("current_position")

        if status in {"needs_review", "review"}:
            state_text = "cần ôn lại"
        else:
            state_text = "đang học dở"

        extras = []
        if page:
            extras.append(f"trang {page}")
        if position not in (None, "", 0):
            extras.append(f"vị trí {position}")
        suffix = f" – {', '.join(extras)}" if extras else ""

        parts.append(
            f"• {label}{(': ' + detail) if detail else ''} – {state_text}{suffix}"
        )
        if len(parts) >= 6:
            break

    if parts:
        unfinished_summary = "\n".join(parts)
        progress_text = (
            "📖 Những phần cậu đang học dở/cần ôn:\n"
            f"{unfinished_summary}"
        )
        closing = (
            "\n\nCậu muốn học tiếp từ chỗ đang dở hay chọn một phần khác? 😊"
        )
    else:
        progress_text = (
            "📖 Hiện tại cậu không có phần nào đang học dở hoặc cần ôn "
            "được lưu trong tiến độ."
        )
        closing = "\n\nCậu muốn bắt đầu hoặc chọn một phần để học tiếp? 😊"

    message = (
        f"Chào {nickname}! 👋 Mừng cậu quay lại với Doraemon. 🤖\n\n"
        f"{curriculum}\n\n"
        f"{progress_text}"
        f"{closing}"
    )
    return {
        "success": True,
        "mode": "returning",
        "message": message,
        "learning_history": unfinished_rows,
    }


@app.post("/api/proxy-chat")
def proxy_chat(
    data: ChatRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    perf_total = time.perf_counter()
    user = require_active_user(authorization)
    perf_auth = time.perf_counter()

    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    if not data.text:
        raise HTTPException(400, "Tin nhắn không được để trống.")

    # Paid packages are unlimited. Free is limited to 5 accepted questions/day.
    # Standalone greetings are onboarding actions and do not consume a question.
    if not _is_pure_greeting(data.text):
        enforce_question_limit(user["id"])

    # A standalone greeting is a session/onboarding action, NOT a knowledge
    # question. Do not send "Chào" through embedding/Pinecone/Gemini, because
    # generic RAG similarity can accidentally make Doraemon start teaching a
    # random lesson (for example Bài tập) immediately after saying hello.
    if _is_pure_greeting(data.text):
        welcome = _build_welcome_for_user(user, mark_seen=False)
        return {
            "reply": welcome["message"],
            "model": GEMINI_MODEL,
            "sources": [],
            "images": [],
            "content_blocks": [{"type": "text", "text": welcome["message"]}],
            "learning_history_count": len(welcome.get("learning_history") or []),
            "learning_progress": None,
            "welcome": True,
            "welcome_mode": welcome.get("mode"),
        }

    namespace = data.knowledge_namespace or "__default__"
    query_text = data.text

    # Keep enough history for continuity, but do not send 100 rows to Gemini.
    # The full learning state remains in PostgreSQL; this is only the prompt view.
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT subject,content_type,content_id,lesson,topic,item_key,score,status,
                       current_position,current_page,attempt_count,correct_count,wrong_count,
                       last_studied_at,next_review_at,completed_at
                FROM learning_progress
                WHERE user_id=%s
                ORDER BY last_studied_at DESC LIMIT 8
            """, (user["id"],))
            learning = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    catalog = _load_catalog_cached()
    perf_state = time.perf_counter()

    low = query_text.strip().lower()

    # Conversational memory: keep the recent turns so short follow-ups such as
    # "A", "câu tiếp theo", "giải thích câu này" still have their real context.
    # The client already sends chat_history; this is deliberately kept compact
    # to control latency while preserving the current exercise/lesson context.
    recent_history = []
    for item in (data.chat_history or [])[-4:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        parts = item.get("parts")
        text = ""
        if isinstance(parts, list):
            texts = []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    texts.append(str(part.get("text")))
            text = " ".join(texts).strip()
        elif item.get("text"):
            text = str(item.get("text")).strip()
        elif item.get("content"):
            text = str(item.get("content")).strip()
        if role in {"user", "model", "assistant"} and text:
            role = "model" if role == "assistant" else role
            recent_history.append({"role": role, "text": text[-900:]})

    # Resolve explicit intent before semantic retrieval. Kanji/Bộ thủ are
    # lessons under Từ vựng, never standalone content types.
    named_lesson_topic = _explicit_lesson_topic(low, catalog)
    if named_lesson_topic:
        requested_scope = {
            "course": str(
                named_lesson_topic.get("course")
                or named_lesson_topic.get("course_name")
                or ""
            ).strip() or None,
            "content_type": _normalize_content_type(
                named_lesson_topic.get("content_type")
            ),
            "lesson": str(named_lesson_topic.get("lesson") or "").strip() or None,
            "topic": str(named_lesson_topic.get("topic") or "").strip() or None,
        }
    else:
        requested_scope = _select_active_scope(low, [], catalog)

    # Generic vocabulary identity routing for names such as "Bộ Vi" where the
    # catalog stores lesson="Bộ thủ" but does not store the individual term as
    # topic. The lesson is still authoritative; the actual term is left in the
    # semantic text query so the text chunk is selected inside that lesson.
    if not named_lesson_topic:
        if re.search(r"\bbộ\s+[^\s]+", low, flags=re.UNICODE):
            requested_scope["content_type"] = "Từ vựng"
            requested_scope["lesson"] = "Bộ thủ"
        elif re.search(r"\bkanji\s+[^\s]+", low, flags=re.UNICODE):
            requested_scope["content_type"] = "Từ vựng"
            requested_scope["lesson"] = "Kanji"

    requested_content_type = requested_scope.get("content_type")
    requested_course = requested_scope.get("course")
    requested_lesson = requested_scope.get("lesson")
    requested_topic = requested_scope.get("topic")

    # Continue the most recent in-progress lesson for short follow-ups. This
    # applies to ALL content types (especially exercises), not only Kanji/Bộ thủ.
    # PostgreSQL is the durable state; chat history is only the conversational hint.
    recommendation_words = ("học gì", "học gì hôm nay", "gợi ý", "đề xuất", "chọn bài", "nên học")
    wants_recommendation = any(w in low for w in recommendation_words)
    active_learning = None
    if not (requested_content_type or requested_course or requested_lesson or requested_topic) and not wants_recommendation:
        for lp in learning:
            if str(lp.get("status") or "").strip().lower() in {"in_progress", "review", "active"}:
                active_learning = lp
                break
        if active_learning is None and learning:
            active_learning = learning[0]

        if active_learning:
            requested_content_type = _normalize_content_type(active_learning.get("content_type")) or None
            requested_course = str(active_learning.get("subject") or "").strip() or None
            requested_lesson = str(active_learning.get("lesson") or "").strip() or None
            requested_topic = str(active_learning.get("topic") or "").strip() or None

    def build_scope_filter(record_type, content_type=None, course=None, lesson=None, topic=None):
        scope_filter = {"record_type": {"$eq": record_type}}
        if content_type:
            scope_filter["content_type"] = {"$eq": content_type}
        if course:
            scope_filter["course"] = {"$eq": course}
        if lesson:
            scope_filter["lesson"] = {"$eq": lesson}
        if topic:
            scope_filter["topic"] = {"$eq": topic}
        return scope_filter

    explicit_scope = bool(
        requested_content_type or requested_course or requested_lesson or requested_topic
    )
    active_scope = requested_scope if explicit_scope else None

    # Build a tiny semantic query. If durable learning state already identifies
    # the active lesson/exercise, do NOT embed the whole chat history. Only the
    # active scope/state plus the new user message is needed. For a genuinely
    # unscoped query, use at most the two latest chat turns as a fallback hint.
    rag_query_text = query_text
    if active_scope:
        scope_parts = [
            str(active_scope.get("content_type") or ""),
            str(active_scope.get("course") or ""),
            str(active_scope.get("lesson") or ""),
            str(active_scope.get("topic") or ""),
        ]
        scope_label = " / ".join(x for x in scope_parts if x)
        position_label = ""
        if active_learning:
            position_label = (
                f" Vị trí hiện tại: {active_learning.get('current_position') or ''};"
                f" trang: {active_learning.get('current_page') or ''}."
            )
        rag_query_text = f"Ngữ cảnh học tập: {scope_label}.{position_label}\nTin nhắn: {query_text}"
    elif recent_history:
        history_tail = recent_history[-2:]
        history_context = " ".join(f"{h['role']}: {h['text'][-500:]}" for h in history_tail)
        rag_query_text = f"Ngữ cảnh gần đây: {history_context}\nTin nhắn: {query_text}"

    # One embedding request per chat.
    # embed_text() twice before retrieval, which added an unnecessary Gemini call.
    embed_started = time.perf_counter()
    query_vector = embed_text(rag_query_text)
    perf_embed = time.perf_counter()

    # Default 8 text matches is enough for the compact prompt and keeps RAG fast.
    # Never exceed 10 unless the client explicitly sends a smaller value.
    retrieval_k = min(50 if requested_content_type == "Bài tập" and (requested_lesson or requested_topic) else 10,
                      max(4, int(data.top_k or 8)))

    text_filter = build_scope_filter(
        "text",
        requested_content_type,
        requested_course,
        requested_lesson,
        requested_topic,
    )

    def query_text_matches(filter_override=None):
        return index.query(
            vector=query_vector,
            top_k=retrieval_k,
            include_metadata=True,
            namespace=namespace,
            filter=filter_override if filter_override is not None else text_filter,
        )

    # IMPORTANT: metadata identity wins over semantic similarity.
    # If the user named a lesson/topic, first select chunks from that exact
    # lesson/topic. Only when those metadata-scoped queries produce no usable
    # text do we fall back to broader semantic text retrieval.
    result = None
    priority_filters = []

    # Metadata identity is authoritative. We progressively relax ONLY the
    # metadata filter, never the lesson/topic intent:
    #   1) lesson + topic + content_type (+ course when available)
    #   2) lesson + content_type
    #   3) topic + content_type
    #   4) lesson
    #   5) topic
    #   6) content_type
    # Then, and only then, semantic text retrieval is used inside the best
    # available lesson/topic scope.
    def add_priority_filter(content_type=None, course=None, lesson=None, topic=None):
        if not any(x is not None and x != "" for x in (content_type, course, lesson, topic)):
            return
        priority_filters.append(build_scope_filter(
            "text", content_type, course, lesson, topic
        ))

    add_priority_filter(requested_content_type, requested_course, requested_lesson, requested_topic)
    if requested_lesson:
        add_priority_filter(requested_content_type, None, requested_lesson, None)
        add_priority_filter(None, None, requested_lesson, None)
    if requested_topic:
        add_priority_filter(requested_content_type, None, None, requested_topic)
        add_priority_filter(None, None, None, requested_topic)
    if requested_content_type:
        add_priority_filter(requested_content_type, None, None, None)

    # Remove duplicate filters while preserving priority.
    unique_filters = []
    seen_filter_repr = set()
    for pf in priority_filters:
        marker = repr(sorted(pf.items()))
        if marker not in seen_filter_repr:
            seen_filter_repr.add(marker)
            unique_filters.append(pf)
    priority_filters = unique_filters

    def _usable_matches(matches):
        usable = []
        for m in matches or []:
            md = m.metadata or {}
            rt = str(md.get("record_type") or "").strip().lower()
            txt = str(md.get("text", md.get("content", "")) or "").strip()
            associated = str(md.get("associated_text") or "").strip()

            if rt != "image":
                if txt:
                    usable.append(m)
                continue

            # Legacy schema: an image record can itself be the complete
            # text+image chunk (text + image_key on the same Pinecone record).
            image_keys = _parse_image_keys(md.get("image_key"))
            if not image_keys:
                image_keys = _parse_image_keys(md.get("image_keys"))
            if (txt or associated) and image_keys:
                usable.append(m)

        return usable

    for idx_priority, pf in enumerate(priority_filters):
        try:
            candidate = query_text_matches(pf)
            usable = _usable_matches(candidate.matches)

            legacy_filter = dict(pf)
            legacy_filter.pop("record_type", None)

            # Bài tập is frequently multi-chunk and older records may omit
            # record_type. Do not discard those valid text chunks just because
            # one modern record matched the strict filter. Merge both views
            # while staying inside the SAME metadata lesson/topic scope.
            should_merge_legacy = (
                requested_content_type == "Bài tập" and
                (requested_lesson or requested_topic)
            )
            if should_merge_legacy or not usable:
                legacy_candidate = index.query(
                    vector=query_vector,
                    top_k=50 if should_merge_legacy else retrieval_k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=legacy_filter,
                )
                legacy_usable = _usable_matches(legacy_candidate.matches)

                if usable and legacy_usable and should_merge_legacy:
                    merged = []
                    seen_keys = set()
                    for m in list(candidate.matches or []) + list(legacy_candidate.matches or []):
                        md = m.metadata or {}
                        key = (
                            str(getattr(m, "id", "")),
                            str(md.get("source_file") or ""),
                            str(md.get("page") or ""),
                            str(md.get("chunk_index") or ""),
                            str(md.get("text", md.get("content", "")) or "")[:240],
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        if m in usable or m in legacy_usable:
                            merged.append(m)
                    candidate.matches = merged
                    usable = _usable_matches(candidate.matches)
                    print(
                        "[RAG priority-merge] metadata match "
                        f"level={idx_priority + 1} strict={len(_usable_matches(candidate.matches))} "
                        f"legacy={len(legacy_usable)} total={len(usable)}"
                    )
                elif legacy_usable:
                    candidate = legacy_candidate
                    usable = legacy_usable
                    print(
                        "[RAG priority-legacy] metadata match "
                        f"level={idx_priority + 1} chunks={len(usable)}"
                    )

            if usable:
                result = candidate
                print(
                    "[RAG priority] metadata match "
                    f"level={idx_priority + 1} lesson={requested_lesson!r} "
                    f"topic={requested_topic!r} chunks={len(usable)}"
                )
                break
        except Exception as exc:
            print("[RAG priority] query failed:", type(exc).__name__, str(exc))

    # The metadata filter intentionally selects the lesson/topic first.
    # Now, if the user's question names a specific item inside that lesson
    # (e.g. "Bộ Vi"), reduce the lesson-wide result to that exact chunk.
    if result is not None and (requested_lesson or requested_topic):
        focused_matches = _focus_metadata_matches(
            result.matches,
            query_text,
            requested_lesson,
            requested_topic,
            requested_content_type,
        )
        if focused_matches:
            result.matches = focused_matches

    # If exact metadata identity did not find a chunk, do semantic text search
    # INSIDE the identified lesson/topic/content scope. This is the requested
    # order: lesson/topic first, text similarity second.
    if result is None:
        scoped_semantic_filter = build_scope_filter(
            "text",
            requested_content_type,
            None,  # course is deliberately relaxed first
            requested_lesson,
            requested_topic,
        ) if (requested_content_type or requested_lesson or requested_topic) else None

        try:
            candidate = query_text_matches(scoped_semantic_filter)
            if _usable_matches(candidate.matches):
                focused_matches = _focus_metadata_matches(
                    _usable_matches(candidate.matches),
                    query_text,
                    requested_lesson,
                    requested_topic,
                    requested_content_type,
                )
                candidate.matches = focused_matches or _usable_matches(candidate.matches)
                result = candidate
                print(
                    "[RAG semantic-scoped] "
                    f"content_type={requested_content_type!r} "
                    f"lesson={requested_lesson!r} topic={requested_topic!r} "
                    f"chunks={len(_usable_matches(candidate.matches))}"
                )
        except Exception as exc:
            print("[RAG semantic-scoped] query failed:", type(exc).__name__, str(exc))

    # Last semantic fallback only when no lesson/topic scope can be identified.
    if result is None:
        result = query_text_matches(
            build_scope_filter("text", requested_content_type, None, None, None)
            if requested_content_type else None
        )
        print(
            "[RAG semantic-fallback] "
            f"content_type={requested_content_type!r} chunks={len(_usable_matches(result.matches))}"
        )

    # IMPORTANT: retrieve TEXT first. Images are NOT searched independently.
    # They are resolved later from the exact text chunks that actually enter the
    # RAG context. This is the chunk-locked text↔image rule.

    # Compatibility fallback for older Pinecone records:
    # some documents uploaded by previous baselines do not contain
    # `record_type="text"` even though they are valid text chunks. A strict
    # record_type filter would therefore return zero rows and make Doraemon
    # incorrectly claim that the story/document does not exist.
    #
    # We keep the strict query as the fast/default path. Only when it returns
    # no usable text do we retry with the same lesson/content/page scope but
    # WITHOUT the record_type condition, then discard true image records in
    # Python. This preserves the chunk-locked architecture while remaining
    # compatible with old Pinecone data.
    def _usable_text_matches(matches):
        usable = []
        for m in matches or []:
            md = m.metadata or {}
            rt = str(md.get("record_type") or "").strip().lower()
            txt = str(md.get("text", md.get("content", "")) or "").strip()
            associated = str(md.get("associated_text") or "").strip()

            if rt != "image":
                if txt:
                    usable.append(m)
                continue

            image_keys = _parse_image_keys(md.get("image_key"))
            if not image_keys:
                image_keys = _parse_image_keys(md.get("image_keys"))
            if (txt or associated) and image_keys:
                usable.append(m)

        return usable

    usable_initial = _usable_text_matches(result.matches)

    if not usable_initial:
        compat_filter = {}
        if requested_content_type:
            compat_filter["content_type"] = {"$eq": requested_content_type}
        if requested_course:
            # course is historically stored as either `course` or `subject`;
            # Pinecone metadata filters cannot OR these fields portably, so
            # prefer course and use a second subject fallback below if needed.
            compat_filter["course"] = {"$eq": requested_course}
        if requested_lesson:
            compat_filter["lesson"] = {"$eq": requested_lesson}
        if requested_topic:
            compat_filter["topic"] = {"$eq": requested_topic}

        try:
            compat_result = index.query(
                vector=query_vector,
                top_k=retrieval_k,
                include_metadata=True,
                namespace=namespace,
                filter=compat_filter or None,
            )
            compat_usable = _usable_text_matches(compat_result.matches)

            # If the old data uses `subject` instead of `course`, retry once
            # with subject while preserving the exact lesson/content scope.
            if not compat_usable and requested_course:
                compat_filter2 = {
                    k: v for k, v in compat_filter.items() if k != "course"
                }
                compat_filter2["subject"] = {"$eq": requested_course}
                compat_result = index.query(
                    vector=query_vector,
                    top_k=retrieval_k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=compat_filter2 or None,
                )
                compat_usable = _usable_text_matches(compat_result.matches)

            if compat_usable:
                result = compat_result
                print(
                    "[RAG compat] strict record_type=text returned no text; "
                    f"using legacy metadata-compatible retrieval ({len(compat_usable)} chunks)"
                )
        except Exception as exc:
            print("[RAG compat] fallback failed:", type(exc).__name__, str(exc))

    if not explicit_scope:
        active_scope = _select_active_scope(low, result.matches, catalog)

        # If the active scope was inferred from the text result, the text query
        # above may have included mixed lessons. Keep the existing scope routing
        # behaviour, but do not perform a second independent image search.
    image_result = type("_EmptyImageResult", (), {"matches": []})()

    perf_rag = time.perf_counter()

    # active_state is the compact, prompt-facing snapshot of the durable
    # PostgreSQL learning state. V3.7 referenced it in the Gemini prompt but
    # forgot to construct it, which caused every normal chat request to fail
    # with NameError: active_state is not defined.
    active_content_type = (active_scope or {}).get("content_type")
    active_course = (active_scope or {}).get("course")
    active_lesson = (active_scope or {}).get("lesson")
    active_topic = (active_scope or {}).get("topic")

    active_state = {}
    if active_learning:
        active_state = {
            "content_type": _normalize_content_type(active_learning.get("content_type")),
            "subject": active_learning.get("subject"),
            "lesson": active_learning.get("lesson"),
            "topic": active_learning.get("topic"),
            "content_id": active_learning.get("content_id"),
            "item_key": active_learning.get("item_key"),
            "status": active_learning.get("status"),
            "current_position": active_learning.get("current_position"),
            "current_page": active_learning.get("current_page"),
            "attempt_count": active_learning.get("attempt_count"),
            "correct_count": active_learning.get("correct_count"),
            "wrong_count": active_learning.get("wrong_count"),
            "last_studied_at": active_learning.get("last_studied_at"),
            "next_review_at": active_learning.get("next_review_at"),
            "completed_at": active_learning.get("completed_at"),
        }

    # If there is no durable active record, still expose the scope inferred from
    # the current request/RAG so Gemini has a consistent compact state object.
    if active_scope:
        active_state["scope"] = {
            "content_type": active_content_type,
            "course": active_course,
            "lesson": active_lesson,
            "topic": active_topic,
        }

    # Select the exact chunks that will be sent to Gemini. These are the ONLY
    # chunks allowed to contribute images.
    #
    # Normal schema:
    #   record_type=text -> text chunk
    #
    # Legacy schema used by some vocabulary uploads:
    #   record_type=image + text/associated_text + image_key
    #   -> this ONE Pinecone record is itself the complete text+image chunk.
    #
    # For example, the supplied Bộ Vi record has:
    #   content_type=Từ vựng
    #   lesson=Bộ thủ
    #   reading=Vi
    #   meaning=Vây quanh
    #   text=...
    #   image_key=images/b_th_pdf/page_0001/img_07.jpg
    # Therefore we must use its text AND its image together.
    text_chunks = []
    seen_chunk_keys = set()
    exercise_scope = requested_content_type == "Bài tập" and (requested_lesson or requested_topic)

    for m in result.matches:
        md = dict(m.metadata or {})
        record_type = str(md.get("record_type") or "").strip().lower()
        txt = str(md.get("text", md.get("content", "")) or "").strip()
        associated = str(md.get("associated_text") or "").strip()

        if record_type == "image":
            image_keys = _parse_image_keys(md.get("image_key"))
            if not image_keys:
                image_keys = _parse_image_keys(md.get("image_keys"))

            if (txt or associated) and image_keys:
                txt = txt or associated
            else:
                continue

        if not txt:
            continue

        base_key = (
            str(md.get("source_file") or ""),
            str(md.get("page") or ""),
            str(md.get("chunk_index") if md.get("chunk_index") not in (None, "") else ""),
            _normalize_chunk_text_for_match(txt)[:280],
        )
        if base_key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(base_key)

        # A single stored PDF chunk can still contain several customer/order
        # blocks. Split those deterministically before resolving images so that
        # each customer gets its own text context and its own image marker.
        parts = _split_exercise_logical_chunks(txt) if exercise_scope else [txt]
        do_virtual = exercise_scope and len(parts) > 1
        for sub_index, part in enumerate(parts):
            sub_md = dict(md)
            if do_virtual:
                # Keep original identity for tracing, but mark the chunk as a
                # virtual exercise split. Image resolution will use associated
                # text identity instead of a potentially stale page chunk_index.
                sub_md["_parent_chunk_index"] = md.get("chunk_index")
                sub_md["_virtual_exercise_split"] = True
                sub_md["chunk_index"] = f"virtual:{sub_index}"
            text_chunks.append({
                "text": part,
                "metadata": sub_md,
                "score": float(getattr(m, "score", 0) or 0),
            })
            if len(text_chunks) >= (12 if exercise_scope else 6):
                break

        if len(text_chunks) >= (12 if exercise_scope else 6):
            break

    chunk_debug = []
    for c in text_chunks[:6]:
        md = c["metadata"]
        chunk_debug.append({
            "record_type": md.get("record_type"),
            "lesson": md.get("lesson"),
            "topic": md.get("topic"),
            "reading": md.get("reading"),
            "source_file": md.get("source_file"),
            "page": md.get("page"),
            "image_keys": len(_parse_image_keys(md.get("image_key"))),
            "virtual": bool(md.get("_virtual_exercise_split")),
            "chunk_index": md.get("chunk_index"),
        })
    print("[RAG chunks]", chunk_debug)


    # Resolve images ONLY for these exact text chunks.
    rich_images = _retrieve_images_for_text_chunks(
        text_chunks, index, namespace, query_vector
    )

    contexts = []
    source_meta = []
    for order, chunk in enumerate(text_chunks):
        md = chunk["metadata"]
        label = (
            f"[CHUNK_{order}] "
            f"[Loại: {md.get('content_type','') or 'Không rõ'} | "
            f"Môn: {md.get('subject',md.get('course',''))} | "
            f"Bài: {md.get('lesson','')} | Chủ đề: {md.get('topic','')} | "
            f"Trang: {md.get('page','')}]"
        )
        contexts.append(label + "\n" + chunk["text"])
        source_meta.append(md)

    active_text_pages = set()
    for chunk in text_chunks:
        md = chunk["metadata"]
        sf = str(md.get("source_file") or "").strip()
        page = str(md.get("page") or "").strip()
        if sf and page:
            active_text_pages.add((sf, page))

    # Mark which chunks actually have images. Gemini is asked to emit the
    # corresponding internal marker immediately after the relevant explanation.
    image_orders = sorted({
        int(item["_chunk_order"])
        for item in rich_images
        if item.get("_chunk_order") is not None
    })

    # Keep the RAG context compact: retrieval still uses up to 10 matches,
    # but Gemini only receives the best six selected text chunks.
    prompt_contexts = []
    for c in contexts:
        if len(c) > 1800:
            c = c[:1800] + "…"
        prompt_contexts.append(c)

    # Compact prompt-only catalog/history. V3.7 accidentally referenced
    # prompt_catalog/prompt_history without constructing them, causing
    # NameError before Gemini was called.
    prompt_history = recent_history[-4:] if recent_history else []

    # Do not send the full catalog on every request. Only expose a compact
    # catalog when the user is actually asking what to study / for a
    # recommendation. For ordinary questions this stays empty.
    prompt_catalog = []
    if wants_recommendation:
        for item in (catalog or []):
            if not isinstance(item, dict):
                continue
            prompt_catalog.append({
                "content_type": item.get("content_type"),
                "subject": item.get("subject") or item.get("course"),
                "lesson": item.get("lesson"),
                "topic": item.get("topic"),
            })
            if len(prompt_catalog) >= 24:
                break

    image_marker_rule = ""
    if image_orders:
        markers = ", ".join(f"[[IMG_CHUNK_{n}]]" for n in image_orders)
        image_marker_rule = (
            f"\n- Các chunk có ảnh tương ứng là: {markers}. "
            "Khi phần trả lời của cậu sử dụng nội dung của một chunk có ảnh, "
            "hãy đặt marker tương ứng NGAY SAU đúng đoạn/câu trả lời của chunk đó. "
            "Không gom nhiều marker về cuối câu trả lời. Không đổi thứ tự marker. "
            "Marker chỉ là kỹ thuật nội bộ, không được giải thích cho học sinh."
        )

    prompt = f"""Bạn là Doraemon, gia sư tiếng Nhật cá nhân.

NGUYÊN TẮC:
- Thực hiện ngay yêu cầu học tập cụ thể; không hỏi lại nếu đã rõ bài/chủ đề.
- Nội dung gồm đúng 5 loại ngang hàng: Giáo trình, Từ vựng, Ngữ pháp, Bài tập, Truyện đọc. Kanji và Bộ thủ là lesson của Từ vựng, không phải content type.
- Mỗi content type có thể có nhiều sách/tài liệu; chỉ sử dụng đúng nguồn mà RAG và ACTIVE LEARNING STATE xác định.
- Với Giáo trình: bám đúng lesson/phạm vi được RAG cung cấp; có thể vừa hướng dẫn/giải thích vừa cho học sinh làm các bài tập nằm trong chính giáo trình đó. Các bài tập nằm trong Giáo trình vẫn thuộc content type Giáo trình, không tự chuyển thành content type Bài tập.
- Khi người học yêu cầu học/trình bày trọn một bài của Giáo trình, sau phần nội dung chính hãy thêm một mục ngắn “🤖 Doraemon nhận xét” (khoảng 3-5 ý hoặc đoạn ngắn): nêu bài này trọng tâm gì, 1-3 điểm cần nhớ, một lỗi dễ nhầm hoặc mẹo học, và gợi ý bước luyện tiếp. Nhận xét phải được suy ra từ chính RAG CONTEXT/ACTIVE LEARNING STATE, không bịa thêm kiến thức ngoài nguồn.
- “Doraemon nhận xét” là phần hỗ trợ sư phạm, không thay thế hay viết lại toàn bộ giáo trình. Nếu người học chỉ hỏi một chi tiết nhỏ trong bài, không cần ép thêm một phần nhận xét dài; chỉ thêm khi phù hợp hoặc khi người học đang kết thúc/ôn lại toàn bài.
- Ưu tiên ACTIVE LEARNING STATE để tiếp tục đúng bài và vị trí đang học.
- Với Bài tập: để học sinh làm trước, chỉ chấm khi có đáp án; tiếp tục câu hiện tại/câu kế tiếp theo tiến độ.
- Với Truyện đọc: bám tài liệu được RAG cung cấp. Nếu chunk nguồn có OCR/text thì coi đó là văn bản nguồn hợp lệ.
- Không bịa nội dung/trang không có trong RAG.
- Quan trọng: ảnh không được tìm theo độ giống câu hỏi. Ảnh chỉ thuộc về đúng CHUNK có ảnh tương ứng trong RAG.
- Không được dùng ảnh của chunk khác, trang khác hoặc lesson khác chỉ vì nó có vẻ phù hợp.
{image_marker_rule}

ACTIVE LEARNING STATE:
{json.dumps(active_state, ensure_ascii=False, default=str, separators=(",", ":"))}

DANH MỤC (chỉ có khi cần gợi ý):
{json.dumps(prompt_catalog, ensure_ascii=False, default=str, separators=(",", ":"))}

RAG CONTEXT:
{chr(10).join(prompt_contexts)}

RECENT CHAT (chỉ để hiểu câu nói ngắn/đại từ):
{json.dumps(prompt_history, ensure_ascii=False, default=str, separators=(",", ":"))}

TIN NHẮN HIỆN TẠI:
{query_text}"""

    gen_started = time.perf_counter()
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    reply = response.text or ""
    perf_gen = time.perf_counter()

    # rich_images was resolved BEFORE Gemini from the exact text chunks.
    # Do not perform any second semantic image search here.
    content_blocks = build_rich_content_blocks(reply, rich_images)
    # The client still receives a flat images array for backwards compatibility.
    images = [{"key": item["key"], "url": item["url"]} for item in rich_images if item.get("url")]
    perf_blocks = time.perf_counter()

    # Learning progress is authoritative state for the next turn. Save it before
    # returning so a follow-up such as "câu tiếp theo" sees the updated position.
    # The write is kept after Gemini/image construction, so it remains a small
    # final PostgreSQL operation rather than part of the RAG/Gemini latency.
    try:
        event = infer_learning_event(
            user["id"], query_text, reply, catalog, learning, source_meta, active_scope=active_scope
        )
        tracked_event = record_learning_event(user["id"], event) if event else None
    except Exception as e:
        print("Learning progress save skipped:", type(e).__name__, str(e))
        tracked_event = None

    perf_total_done = time.perf_counter()
    print(
        "[PERF proxy_chat] auth=%.3fs state=%.3fs embed=%.3fs rag=%.3fs "
        "gemini=%.3fs blocks=%.3fs total=%.3fs text_k=%d image_k=%d prompt_catalog=%d prompt_active_state=%d"
        % (
            perf_auth - perf_total,
            perf_state - perf_auth,
            perf_embed - perf_state,
            perf_rag - perf_embed,
            perf_gen - perf_rag,
            perf_blocks - perf_gen,
            perf_total_done - perf_total,
            len(text_chunks),
            len(rich_images),
            len(prompt_catalog),
            1 if active_learning else 0,
        )
    )

    return {
        "reply": reply,
        "model": GEMINI_MODEL,
        "sources": source_meta[:10],
        "images": images,
        "content_blocks": content_blocks,
        "learning_history_count": len(learning),
        "learning_progress": tracked_event,
    }


@app.get("/session/welcome")
def session_welcome(authorization: Optional[str] = Header(default=None)):
    """
    Welcome/onboarding endpoint used after login.
    The same welcome logic is also used for a standalone "Chào" in chat.
    """
    user = require_active_user(authorization)
    return _build_welcome_for_user(user, mark_seen=True)


@app.post("/learning/reset")
def reset_learning(authorization: Optional[str] = Header(default=None)):
    """
    Xóa toàn bộ tiến độ học của user và đưa trạng thái giáo trình về như user mới.
    Không xóa tài khoản, gói dịch vụ, hay dữ liệu Knowledge Base.
    """
    user = require_active_user(authorization)
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM learning_progress WHERE user_id=%s", (user["id"],))
            deleted = cur.rowcount

            # Mark the account as a fresh learner. The next /session/welcome
            # therefore returns the same onboarding flow as a brand-new user.
            cur.execute("""
                INSERT INTO user_learning_state(user_id,welcome_seen,reset_count,updated_at)
                VALUES(%s,FALSE,1,NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    welcome_seen=FALSE,
                    reset_count=user_learning_state.reset_count + 1,
                    updated_at=NOW()
            """, (user["id"],))
        conn.commit()
    finally:
        conn.close()

    return {
        "success": True,
        "deleted_progress": deleted,
        "message": "Đã xóa lịch sử học và reset giáo trình về trạng thái ban đầu."
    }


@app.post("/api/learning/reset")
def reset_learning_compat(authorization: Optional[str] = Header(default=None)):
    """
    Compatibility alias for older clients.
    Canonical endpoint remains POST /learning/reset.
    """
    return reset_learning(authorization)


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
                ORDER BY last_studied_at DESC LIMIT 80
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
      <option value="Giáo trình" ${values.content_type==="Giáo trình"?"selected":""}>Giáo trình</option>
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
      <div><span class="status-${st}"><b>${st}</b></span> · Gói: <b>${esc(s.plan||"Free")}</b> · ${s.plan==='Free' ? `đã hỏi hôm nay: ${Number(s.used_today||0)}/5` : `hết hạn: ${ex}`}</div>
      <div class="small">Bấm để xem lịch sử và chat</div>
      <div style="margin-top:7px">
        <button onclick="event.stopPropagation();act(${u.id},1)">1 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},3)">3 tháng</button>

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

def _extract_image_key(raw_key):
    """
    Normalize Pinecone image_key metadata.

    Older/newer uploaders may store image_key as:
      - "images/file.pdf/page_0001/img_01.jpg"
      - ["images/file.pdf/page_0001/img_01.jpg"]
      - JSON string containing the list above
    """
    if raw_key is None:
        return ""

    if isinstance(raw_key, (list, tuple)):
        for item in raw_key:
            key = _extract_image_key(item)
            if key:
                return key
        return ""

    if isinstance(raw_key, dict):
        for field in ("key", "image_key", "path"):
            if field in raw_key:
                key = _extract_image_key(raw_key.get(field))
                if key:
                    return key
        return ""

    value = str(raw_key).strip()
    if not value:
        return ""

    # Try JSON first because Pinecone metadata often contains a JSON list.
    if value.startswith("[") or value.startswith("{"):
        try:
            parsed = json.loads(value)
            key = _extract_image_key(parsed)
            if key:
                return key
        except Exception:
            pass

    # Fallback for Python-literal list strings from older uploaders.
    if value.startswith("[") or value.startswith("("):
        try:
            parsed = ast.literal_eval(value)
            key = _extract_image_key(parsed)
            if key:
                return key
        except Exception:
            pass

    # Last fallback: remove accidental wrapping quotes only.
    return value.strip().strip('"').strip("'")

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
    prompt = f"""Đây là trang {page_no} của một sách học tập, có thể là trang từ vựng/giáo trình tiếng Nhật.

Hãy thực hiện ĐỒNG THỜI 2 việc:
1) OCR toàn bộ chữ nhìn thấy trên trang, giữ nguyên ngôn ngữ gốc, thứ tự đọc hợp lý và xuống dòng ở tiêu đề/ví dụ.
2) Phát hiện từng hình minh họa/ảnh/biểu đồ/sơ đồ có ý nghĩa giáo dục. Không gộp nhiều hình thành một hình. Không coi vùng chữ thuần túy là hình.

Đặc biệt, với MỖI hình, hãy xác định ĐÚNG nhãn/từ vựng gần hình nhất hoặc rõ ràng thuộc về hình đó. Nếu đây là trang từ vựng tiếng Nhật, mỗi hình phải được map 1-1 với nhãn của chính hình đó. TUYỆT ĐỐI KHÔNG gán một hình cho một từ khác chỉ vì nghĩa của chúng giống nhau, và không dùng nghĩa của cả trang để suy đoán.

Quy tắc quan trọng:
- term phải là TỪ/NHÃN thực sự gắn với chính hình đó, ưu tiên chữ nằm ngay cạnh/trên/dưới hình.
- Nếu không thể xác định chắc term của một hình, để term và reading là chuỗi rỗng thay vì đoán.
- Không được copy cùng một term cho nhiều hình.
- meaning chỉ là thông tin mô tả, KHÔNG dùng meaning để xác định hình.

Với mỗi hình trả về:
- box: [ymin, xmin, ymax, xmax] chuẩn hóa 0-1000
- term: từ/cụm từ tiếng Nhật gắn với CHÍNH hình này, nếu xác định được
- reading: cách đọc của CHÍNH term đó, nếu xác định được
- meaning: nghĩa tiếng Việt, nếu có thể xác định từ chính trang; nếu không chắc để chuỗi rỗng
- associated_text: đoạn chữ ngắn thực sự liên quan tới CHÍNH hình
- description: mô tả ngắn hình

Chỉ trả JSON đúng schema:
{{
  "text":"...",
  "images":[
    {{
      "box":[0,0,1000,1000],
      "term":"",
      "reading":"",
      "meaning":"",
      "associated_text":"",
      "description":""
    }}
  ]
}}"""
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
            term = str(item.get("term") or "").strip()
            reading = str(item.get("reading") or "").strip()
            meaning = str(item.get("meaning") or "").strip()
            associated_text = str(item.get("associated_text") or "").strip()
            bbox = json.dumps(item.get("box"), ensure_ascii=False) if isinstance(item.get("box"), (list, tuple)) else ""
            conn = db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO knowledge_images
                        (source_file,subject,content_type,lesson,topic,page,image_key,image_url,description,term,reading,meaning,associated_text,bbox,width,height)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (source_file, subject, primary.get("content_type","Từ vựng"), primary.get("lesson"), primary.get("topic"),
                         page_no, key, b2_url(key), description, term, reading, meaning, associated_text, bbox, width, height))
                conn.commit()
            finally:
                conn.close()
            stored.append({"key": key, "description": description, "term": term, "reading": reading,
                           "meaning": meaning, "associated_text": associated_text, "bbox": bbox, "page": page_no})
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
    image_vectors_total=0
    try:
        # Khi re-upload cùng source_file, xóa các vector cũ của tài liệu này
        # để tránh record V1 cũ tiếp tục cạnh tranh với record V2 mới.
        try:
            index.delete(filter={"source_file": source_file}, namespace=namespace)
        except Exception as e:
            print("Pinecone old-source cleanup skipped:", type(e).__name__, str(e))

        for page_no in range(1, len(reader.pages)+1):
            text = page_texts.get(page_no, "")
            chunks=kb_chunk_text(text,chunk_size,overlap)
            page_meta=metadata_for_page(records_meta,page_no)
            primary=page_meta[0] if page_meta else None
            content_type=primary["content_type"] if primary else (records_meta[0]["content_type"] if records_meta else "Từ vựng")
            for chunk_no,chunk in enumerate(chunks):
                md_list=[{
                    "content_type":r["content_type"],"lesson":r["lesson"],"lesson_pages":r["lesson_pages"],
                    "topic":r["topic"],"topic_pages":r["topic_pages"],
                    "question_pages":r["question_pages"],"answer_pages":r["answer_pages"]
                } for r in page_meta]
                md={
                    "record_type":"text",
                    "text":chunk,"course":subject,"subject":subject,"content_type":content_type,
                    "source_file":source_file,"page":page_no,"chunk_index":chunk_no,
                    "metadata_records":json.dumps(md_list,ensure_ascii=False),
                    "image_keys":json.dumps([],ensure_ascii=False)
                }
                if primary:
                    md.update({
                        "lesson":primary["lesson"],"lesson_pages":primary["lesson_pages"],
                        "topic":primary["topic"],"topic_pages":primary["topic_pages"],
                        "question_pages":primary["question_pages"],"answer_pages":primary["answer_pages"]
                    })
                vectors.append({"id":uuid.uuid4().hex,"values":embed_text(chunk),"metadata":md})
                total+=1

            # Mỗi ảnh là MỘT Pinecone record độc lập. Không còn nhét toàn bộ
            # ảnh của trang vào metadata của text chunk.
            page_chunk_count = len(chunks)
            for img in page_images.get(page_no, []):
                key = str(img.get("key") or "").strip()
                if not key:
                    continue
                term = str(img.get("term") or "").strip()
                reading = str(img.get("reading") or "").strip()
                meaning = str(img.get("meaning") or "").strip()
                associated_text = str(img.get("associated_text") or "").strip()
                description = str(img.get("description") or "").strip()
                search_text = " | ".join(x for x in [term, reading, meaning, associated_text, description, f"Trang {page_no}"] if x)
                if not search_text:
                    search_text = f"Hình minh họa trang {page_no}"
                image_md={
                    "record_type":"image",
                    "text":search_text,
                    "course":subject,"subject":subject,"content_type":content_type,
                    "source_file":source_file,"page":page_no,
                    "image_key":key,"image_url":b2_url(key),
                    # Strict chunk mapping: if the page contains exactly one
                    # text chunk, the image belongs unambiguously to chunk 0.
                    # For multi-chunk pages we leave chunk_index unset unless
                    # the extractor later supplies an explicit mapping.
                    "term":term,"reading":reading,"meaning":meaning,
                    "associated_text":associated_text,"description":description,
                    "bbox":str(img.get("bbox") or "")
                }
                if page_chunk_count == 1:
                    image_md["chunk_index"] = 0

                if img.get("chunk_index") not in (None, ""):
                    try:
                        image_md["chunk_index"] = int(img.get("chunk_index"))
                    except Exception:
                        image_md["chunk_index"] = str(img.get("chunk_index")).strip()

                if primary:
                    image_md.update({
                        "lesson":primary["lesson"],"topic":primary["topic"],
                        "lesson_pages":primary["lesson_pages"],"topic_pages":primary["topic_pages"]
                    })
                vectors.append({"id":uuid.uuid4().hex,"values":embed_text(search_text),"metadata":image_md})
                image_vectors_total += 1

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

    _invalidate_catalog_cache()

    image_count=sum(len(v) for v in page_images.values())
    scanned_pages=sum(1 for p in range(1,len(reader.pages)+1) if len(re.sub(r"\s+","",(reader.pages[p-1].extract_text() or ""))) < 30)
    return {"success":True,"filename":source_file,"subject":subject,
            "pages":len(reader.pages),"scanned_pages_ocr":scanned_pages,"chunks":total,"records":len(records_meta),
            "images":image_count,"image_vectors":image_vectors_total,"pdf_url":pdf_url,"dimension":768,"index":PINECONE_INDEX,"namespace":namespace}

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
                    s.id subscription_id,s.plan,s.started_at,s.expires_at,s.status subscription_status,
                    COALESCE(dq.question_count,0) AS used_today
                    FROM users u LEFT JOIN LATERAL
                    (SELECT * FROM subscriptions WHERE user_id=u.id ORDER BY id DESC LIMIT 1) s ON TRUE
                    LEFT JOIN daily_question_usage dq ON dq.user_id=u.id AND dq.usage_date=%s
                    ORDER BY u.id DESC""", (_now_local().date(),))
            rows=cur.fetchall()
    finally: conn.close()
    now = datetime.now(timezone.utc)
    users_out = []
    for r in rows:
        raw_plan = str(r["plan"] or "Free")
        raw_exp = r["expires_at"]
        paid_active = raw_plan != "Free" and str(r["subscription_status"] or "").upper() == "ACTIVE" and raw_exp and raw_exp > now
        plan = raw_plan if paid_active else "Free"
        users_out.append({
            "id": r["id"], "phone": r["phone"], "nickname": r["nickname"], "status": r["status"],
            "created_at": r["created_at"],
            "subscription": {
                "id": r["subscription_id"],
                "plan": plan,
                "started_at": r["started_at"] if paid_active else None,
                "expires_at": raw_exp if paid_active else None,
                "status": "ACTIVE",
                "used_today": int(r["used_today"] or 0),
                "daily_limit": None if paid_active else 5
            }
        })
    return {"users": users_out}

@app.post("/admin/api/users/{user_id}/activate")
def admin_activate(user_id:int,data:dict):
    check_admin(str(data.get("password",""))); months=int(data.get("months",1))
    if months not in (1,3,6): raise HTTPException(400,"Thời hạn phải 1, 3 hoặc 6 tháng.")
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE id=%s",(user_id,))
            if not cur.fetchone(): raise HTTPException(404,"Không tìm thấy user.")
            cur.execute("SELECT id,plan,expires_at FROM subscriptions WHERE user_id=%s ORDER BY id DESC LIMIT 1",(user_id,))
            old=cur.fetchone(); now=datetime.now(timezone.utc)
            old_plan = str(old.get("plan") or "Free") if old else "Free"
            start = old["expires_at"] if old and old_plan != "Free" and old["expires_at"] and old["expires_at"]>now else now
            exp=start+timedelta(days=30*months)
            plan_name=f"{months} tháng"
            if old:
                cur.execute("UPDATE subscriptions SET plan=%s,started_at=%s,expires_at=%s,status='ACTIVE' WHERE id=%s",
                            (plan_name,start,exp,old["id"]))
            else:
                cur.execute("INSERT INTO subscriptions(user_id,plan,started_at,expires_at,status) VALUES(%s,%s,%s,%s,'ACTIVE')",
                            (user_id,plan_name,start,exp))
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
