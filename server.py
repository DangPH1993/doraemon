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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "doraemon")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_RENDER")
ADMIN_WS_TOKEN = os.getenv("ADMIN_WS_TOKEN")
ADMIN_PANEL_PASSWORD = os.getenv("ADMIN_PANEL_PASSWORD", ADMIN_WS_TOKEN)
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

app = FastAPI(title="Doraemon SaaS Server")
SERVER_VERSION = "2026-08-13-content-type-v1"
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
                subject VARCHAR(255) NOT NULL, lesson VARCHAR(255), topic VARCHAR(255), item_key VARCHAR(500),
                score INTEGER, status VARCHAR(50) NOT NULL DEFAULT 'studied',
                last_studied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")

            # Migration: database cũ có thể đã có bảng learning_progress
            # nhưng chưa có các cột mới. ADD COLUMN IF NOT EXISTS giúp nâng cấp
            # schema mà không xóa dữ liệu user đã học.
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS subject VARCHAR(255);")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS lesson VARCHAR(255);")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS topic VARCHAR(255);")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS item_key VARCHAR(500);")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS score INTEGER;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'studied';")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS last_studied_at TIMESTAMPTZ DEFAULT NOW();")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS content_type VARCHAR(50);")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS content_id VARCHAR(500);")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS current_position INTEGER;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS current_page INTEGER;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS correct_count INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS wrong_count INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS next_review_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE learning_progress ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE learning_progress ALTER COLUMN status SET DEFAULT 'studied';")
            cur.execute("UPDATE learning_progress SET last_studied_at=COALESCE(last_studied_at,NOW());")
            cur.execute("UPDATE learning_progress SET attempt_count=COALESCE(attempt_count,0), correct_count=COALESCE(correct_count,0), wrong_count=COALESCE(wrong_count,0);")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_learning_progress_user_type
                           ON learning_progress(user_id,content_type,last_studied_at DESC);""")

            # Các cột mới có thể được thêm vào bảng cũ với NULL. Đảm bảo
            # last_studied_at luôn có giá trị trước khi tạo index.
            cur.execute("UPDATE learning_progress SET last_studied_at=NOW() WHERE last_studied_at IS NULL;")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_learning_progress_user
                           ON learning_progress(user_id,last_studied_at DESC);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_documents (
                id BIGSERIAL PRIMARY KEY, source_file VARCHAR(500) NOT NULL, subject VARCHAR(255) NOT NULL,
                content_type VARCHAR(30) NOT NULL DEFAULT 'Từ vựng',
                lesson VARCHAR(255), lesson_pages VARCHAR(255), topic VARCHAR(255), topic_pages VARCHAR(255),
                question_pages VARCHAR(255), answer_pages VARCHAR(255), namespace VARCHAR(255) NOT NULL DEFAULT '__default__',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            # Migration for databases created by previous server versions.
            cur.execute("ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS content_type VARCHAR(30) DEFAULT 'Từ vựng';")
            cur.execute("UPDATE knowledge_documents SET content_type='Từ vựng' WHERE content_type IS NULL OR TRIM(content_type)='';")
        conn.commit()
    finally:
        conn.close()

@app.on_event("startup")
def startup():
    global pc, index, gemini
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
    if GEMINI_API_KEY:
        gemini = genai.Client(api_key=GEMINI_API_KEY)
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


def _normalize_content_type(value):
    v = str(value or "").strip()
    aliases = {
        "vocabulary": "Từ vựng", "từ vựng": "Từ vựng",
        "grammar": "Ngữ pháp", "ngữ pháp": "Ngữ pháp",
        "exercise": "Bài tập", "bài tập": "Bài tập",
        "reading": "Truyện đọc", "truyện đọc": "Truyện đọc",
    }
    return aliases.get(v.lower(), v if v in {"Từ vựng","Ngữ pháp","Bài tập","Truyện đọc"} else "")

def record_learning_event(user_id, event):
    """Persist a normalized learning event. Non-scored content is tracked as progress only."""
    ctype = _normalize_content_type(event.get("content_type"))
    if not ctype:
        return None

    subject = str(event.get("subject") or "").strip() or None
    lesson = str(event.get("lesson") or "").strip() or None
    topic = str(event.get("topic") or "").strip() or None
    content_id = str(event.get("content_id") or event.get("item_key") or "").strip() or None
    status = str(event.get("status") or "studied").strip().lower()
    if status not in {"studied","in_progress","completed","review"}:
        status = "studied"

    # Only exercises are scored.
    score = event.get("score") if ctype == "Bài tập" else None
    try:
        score = int(score) if score is not None else None
    except Exception:
        score = None

    def intval(k):
        try:
            return int(event[k]) if event.get(k) is not None else None
        except Exception:
            return None

    current_position = intval("current_position")
    current_page = intval("current_page")
    attempt_inc = intval("attempt_increment") or 0
    correct_inc = intval("correct_increment") or 0
    wrong_inc = intval("wrong_increment") or 0

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # One active progress row per user/content. Legacy rows without
            # content_type are left intact; new tracking uses content_id.
            cur.execute("""
                SELECT id,attempt_count,correct_count,wrong_count,score
                FROM learning_progress
                WHERE user_id=%s
                  AND COALESCE(content_type,'')=%s
                  AND COALESCE(content_id,'')=COALESCE(%s,'')
                  AND COALESCE(subject,'')=COALESCE(%s,'')
                  AND COALESCE(lesson,'')=COALESCE(%s,'')
                  AND COALESCE(topic,'')=COALESCE(%s,'')
                ORDER BY id DESC LIMIT 1
            """, (user_id,ctype,content_id,subject,lesson,topic))
            old = cur.fetchone()

            if old:
                new_score = score if score is not None else old.get("score")
                cur.execute("""
                    UPDATE learning_progress SET
                      item_key=COALESCE(%s,item_key),
                      score=%s,
                      status=%s,
                      current_position=COALESCE(%s,current_position),
                      current_page=COALESCE(%s,current_page),
                      attempt_count=COALESCE(attempt_count,0)+%s,
                      correct_count=COALESCE(correct_count,0)+%s,
                      wrong_count=COALESCE(wrong_count,0)+%s,
                      last_studied_at=NOW(),
                      completed_at=CASE WHEN %s='completed' THEN NOW() ELSE completed_at END,
                      next_review_at=%s
                    WHERE id=%s
                    RETURNING *
                """, (content_id,new_score,status,current_position,current_page,
                      max(0,attempt_inc),max(0,correct_inc),max(0,wrong_inc),
                      status,event.get("next_review_at"),old["id"]))
            else:
                cur.execute("""
                    INSERT INTO learning_progress
                    (user_id,subject,lesson,topic,item_key,score,status,last_studied_at,
                     content_type,content_id,current_position,current_page,
                     attempt_count,correct_count,wrong_count,next_review_at,completed_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,
                           CASE WHEN %s='completed' THEN NOW() ELSE NULL END)
                    RETURNING *
                """, (user_id,subject,lesson,topic,content_id,score,status,
                      ctype,content_id,current_position,current_page,
                      max(0,attempt_inc),max(0,correct_inc),max(0,wrong_inc),
                      event.get("next_review_at"),status))
            row = dict(cur.fetchone())
        conn.commit()
        return row
    finally:
        conn.close()

def infer_learning_event(user_id, user_text, assistant_text, catalog, learning):
    """
    Ask Gemini for a tiny structured event describing what was actually studied.
    If the model cannot identify a concrete event, it returns track=false.
    This is intentionally separate from the teaching response so chat remains
    compatible with the existing API.
    """
    if not gemini:
        return None
    event_prompt = f"""Bạn là bộ phận ghi nhận tiến độ học của Doraemon.
Chỉ tạo JSON nếu lượt trò chuyện này thực sự có hoạt động học cụ thể.
Không tạo event chỉ vì user chào hỏi, hỏi hệ thống, hoặc chỉ nói chuyện.

QUY TẮC:
- content_type chỉ được: Từ vựng, Ngữ pháp, Bài tập, Truyện đọc.
- Truyện đọc: chỉ ghi nhận truyện đã bắt đầu/đang đọc/đã hoàn thành và current_page nếu có.
- Từ vựng: chỉ ghi nhận chủ đề/từ vựng đã học hoặc đang học và current_position nếu có.
- Ngữ pháp: ghi nhận bài/chủ đề đã học hoặc đang học.
- Bài tập: mới được ghi score/attempt/correct/wrong.
- Không tự bịa score.
- Không đánh dấu completed nếu hội thoại chưa cho thấy đã hoàn thành.
- Chọn subject/lesson/topic/content_id từ danh mục nếu có.
- Nếu không xác định được nội dung cụ thể, track=false.

Trả về DUY NHẤT JSON:
{{
 "track": true/false,
 "content_type": "...",
 "subject": "...",
 "lesson": "...",
 "topic": "...",
 "content_id": "...",
 "status": "in_progress|studied|completed|review",
 "current_position": null,
 "current_page": null,
 "score": null,
 "attempt_increment": 0,
 "correct_increment": 0,
 "wrong_increment": 0
}}

DANH MỤC:
{json.dumps(catalog,ensure_ascii=False,default=str)}

LỊCH SỬ:
{json.dumps(learning[-30:],ensure_ascii=False,default=str)}

USER:
{user_text}

DORAEMON:
{assistant_text}
"""
    try:
        r = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=event_prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        raw = (r.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
        event = json.loads(raw)
        if not isinstance(event, dict) or not event.get("track"):
            return None
        return event
    except Exception as e:
        print("Learning event inference skipped:", type(e).__name__, str(e))
        return None

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

    # Tự động ghi nhận tiến độ. Nếu bước này lỗi thì KHÔNG làm hỏng câu trả lời
    # của người học.
    tracked_event=None
    try:
        event=infer_learning_event(user["id"],data.text,reply,catalog,learning)
        if event:
            tracked_event=record_learning_event(user["id"],event)
    except Exception as e:
        print("Learning progress save skipped:", type(e).__name__, str(e))

    return {
        "reply":reply,
        "model":GEMINI_MODEL,
        "sources":source_meta[:10],
        "learning_history_count":len(learning),
        "learning_progress":tracked_event
    }

@app.post("/learning/progress")
def save_learning_progress(data: dict, authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization)
    if not data.get("content_type"):
        # Backward compatible manual calls may still only provide subject.
        data["content_type"]="Từ vựng"
    row=record_learning_event(user["id"],data)
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

@app.get("/health")
def health():
    return {"status":"ok","pinecone":index is not None,"gemini":gemini is not None,
            "database":bool(DATABASE_URL),"gemini_model":GEMINI_MODEL}
