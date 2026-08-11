import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from pinecone import Pinecone
from google import genai
from google.genai import types

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "doraemon")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_RENDER")
ADMIN_WS_TOKEN = os.getenv("ADMIN_WS_TOKEN")
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

app = FastAPI(title="Doraemon SaaS Server")
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
                id BIGSERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                lesson VARCHAR(255),
                topic VARCHAR(255),
                content_type VARCHAR(50),
                item_id VARCHAR(255),
                status VARCHAR(30) NOT NULL DEFAULT 'IN_PROGRESS',
                score DOUBLE PRECISION,
                total_questions INTEGER,
                correct_questions INTEGER,
                note TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_learning_progress_user_updated
                           ON learning_progress(user_id, updated_at DESC);""")
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
    message: str

class LearningProgressRequest(BaseModel):
    lesson: Optional[str] = None
    topic: Optional[str] = None
    content_type: Optional[str] = None
    item_id: Optional[str] = None
    status: str = "IN_PROGRESS"
    score: Optional[float] = None
    total_questions: Optional[int] = None
    correct_questions: Optional[int] = None
    note: Optional[str] = None

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
            await notify_admin(row)
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
    if not gemini: raise HTTPException(500, "Gemini chưa được khởi tạo.")
    r = gemini.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return r.embeddings[0].values

def pinecone_search(message, top_k=8, metadata_filter=None):
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    kwargs = {
        "vector": embed_text(message),
        "top_k": top_k,
        "include_metadata": True
    }
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    return index.query(**kwargs)


def normalize_matches(result):
    matches = []
    for m in result.matches:
        md = dict(m.metadata or {})
        text = md.get("text", md.get("content", ""))
        if text:
            matches.append({
                "id": getattr(m, "id", None),
                "score": float(m.score),
                "text": text,
                "metadata": md
            })
    return matches


@app.post("/search")
def search(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization)
    return {"matches": normalize_matches(pinecone_search(data.message, top_k=8))}


@app.get("/learning/lessons")
def learning_lessons(authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization)
    # Pinecone query là semantic discovery; upload service cần lưu metadata lesson.
    result = pinecone_search("danh sách các bài học trong giáo trình", top_k=100)
    lessons = {}
    for item in normalize_matches(result):
        md = item["metadata"]
        lesson = md.get("lesson")
        if lesson:
            key = str(lesson).strip()
            lessons.setdefault(key, {
                "lesson": key,
                "topic": md.get("topic"),
                "source": md.get("source", md.get("filename"))
            })
    return {"lessons": list(lessons.values())}


@app.get("/learning/topics")
def learning_topics(
    lesson: Optional[str] = None,
    authorization: Optional[str] = Header(default=None)
):
    require_active_user(authorization)
    metadata_filter = {"lesson": lesson} if lesson else None
    result = pinecone_search("các chủ đề học tập", top_k=100, metadata_filter=metadata_filter)
    topics = {}
    for item in normalize_matches(result):
        md = item["metadata"]
        topic = md.get("topic")
        if topic:
            key = str(topic).strip()
            topics.setdefault(key, {
                "topic": key,
                "lesson": md.get("lesson"),
                "source": md.get("source", md.get("filename"))
            })
    return {"topics": list(topics.values())}


@app.get("/learning/questions")
def learning_questions(
    lesson: Optional[str] = None,
    topic: Optional[str] = None,
    authorization: Optional[str] = Header(default=None)
):
    require_active_user(authorization)
    metadata_filter = {"content_type": "question"}
    if lesson:
        metadata_filter["lesson"] = lesson
    if topic:
        metadata_filter["topic"] = topic
    result = pinecone_search("câu hỏi bài tập", top_k=100, metadata_filter=metadata_filter)
    return {"questions": normalize_matches(result)}


@app.post("/learning/progress")
def save_learning_progress(
    data: LearningProgressRequest,
    authorization: Optional[str] = Header(default=None)
):
    user = require_active_user(authorization)
    status = data.status.upper().strip()
    allowed = {"IN_PROGRESS", "COMPLETED", "PASSED", "FAILED"}
    if status not in allowed:
        raise HTTPException(400, f"status phải là một trong: {', '.join(sorted(allowed))}")

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO learning_progress (
                    user_id, lesson, topic, content_type, item_id, status,
                    score, total_questions, correct_questions, note
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id,user_id,lesson,topic,content_type,item_id,status,
                          score,total_questions,correct_questions,note,
                          created_at,updated_at
            """, (
                user["id"], data.lesson, data.topic, data.content_type,
                data.item_id, status, data.score, data.total_questions,
                data.correct_questions, data.note
            ))
            row = dict(cur.fetchone())
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "progress": row}


@app.get("/learning/progress")
def get_learning_progress(
    lesson: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = 100,
    authorization: Optional[str] = Header(default=None)
):
    user = require_active_user(authorization)
    limit = max(1, min(limit, 500))
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """SELECT id,lesson,topic,content_type,item_id,status,score,
                              total_questions,correct_questions,note,
                              created_at,updated_at
                       FROM learning_progress WHERE user_id=%s"""
            params = [user["id"]]
            if lesson:
                query += " AND lesson=%s"
                params.append(lesson)
            if topic:
                query += " AND topic=%s"
                params.append(topic)
            query += " ORDER BY updated_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return {"progress": rows}


@app.get("/learning/summary")
def learning_summary(authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT COUNT(*) AS total_records,
                       COUNT(*) FILTER (WHERE status='COMPLETED') AS completed,
                       COUNT(*) FILTER (WHERE status='IN_PROGRESS') AS in_progress,
                       COUNT(*) FILTER (WHERE status='PASSED') AS passed,
                       COUNT(*) FILTER (WHERE status='FAILED') AS failed,
                       AVG(score) AS average_score,
                       COUNT(DISTINCT lesson) FILTER (WHERE lesson IS NOT NULL) AS lessons_studied,
                       COUNT(DISTINCT topic) FILTER (WHERE topic IS NOT NULL) AS topics_studied
                FROM learning_progress WHERE user_id=%s
            """, (user["id"],))
            summary = dict(cur.fetchone())
            cur.execute("""
                SELECT lesson,topic,content_type,item_id,status,score,updated_at
                FROM learning_progress WHERE user_id=%s
                ORDER BY updated_at DESC LIMIT 10
            """, (user["id"],))
            recent = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return {"summary": summary, "recent": recent}


@app.post("/api/proxy-chat")
def proxy_chat(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)
    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")

    result = pinecone_search(data.message, top_k=8)
    matches = normalize_matches(result)

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT lesson,topic,content_type,item_id,status,score
                FROM learning_progress
                WHERE user_id=%s
                ORDER BY updated_at DESC LIMIT 10
            """, (user["id"],))
            progress = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    contexts = [
        f"[metadata={item['metadata']}]\n{item['text']}"
        for item in matches
    ]
    progress_text = "\n".join(
        f"- lesson={p.get('lesson')}, topic={p.get('topic')}, "
        f"type={p.get('content_type')}, item={p.get('item_id')}, "
        f"status={p.get('status')}, score={p.get('score')}"
        for p in progress
    ) or "Chưa có lịch sử học tập."

    prompt = f"""Bạn là Doraemon, trợ lý học tiếng Nhật.

Mục tiêu:
- Giúp người dùng học theo giáo trình đã nạp vào hệ thống.
- Ưu tiên nội dung giáo trình trong CONTEXT.
- Metadata lesson/topic/content_type dùng để hiểu cấu trúc tài liệu.
- Khi người dùng muốn học bài, giúp chọn hoặc tiếp tục bài học.
- Khi muốn học chủ đề, ưu tiên topic phù hợp.
- Khi muốn làm bài, ưu tiên content_type='question'.
- Không bịa nội dung giáo trình.
- Nếu context không đủ, nói rõ tài liệu hiện tại chưa có thông tin đó.
- Có thể dùng lịch sử học tập để đề xuất nội dung tiếp theo, nhưng không coi lịch sử là nội dung giáo trình.

Lịch sử học tập gần đây:
{progress_text}

Câu hỏi:
{data.message}

CONTEXT GIÁO TRÌNH:
{"\n\n---\n\n".join(contexts)}
"""
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return {
        "reply": response.text or "",
        "model": GEMINI_MODEL,
        "learning": {
            "recent_progress_count": len(progress),
            "context_count": len(contexts)
        }
    }


@app.get("/health")
def health():
    return {"status":"ok","pinecone":index is not None,"gemini":gemini is not None,
            "database":bool(DATABASE_URL),"gemini_model":GEMINI_MODEL}
