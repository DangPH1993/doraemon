import os
import io
import uuid
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
import json

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, UploadFile, File
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

            # Các cột mới có thể được thêm vào bảng cũ với NULL. Đảm bảo
            # last_studied_at luôn có giá trị trước khi tạo index.
            cur.execute("UPDATE learning_progress SET last_studied_at=NOW() WHERE last_studied_at IS NULL;")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_learning_progress_user
                           ON learning_progress(user_id,last_studied_at DESC);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_documents (
                id BIGSERIAL PRIMARY KEY, source_file VARCHAR(500) NOT NULL, subject VARCHAR(255) NOT NULL,
                lesson VARCHAR(255), lesson_pages VARCHAR(255), topic VARCHAR(255), topic_pages VARCHAR(255),
                question_pages VARCHAR(255), answer_pages VARCHAR(255), namespace VARCHAR(255) NOT NULL DEFAULT '__default__',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
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

@app.post("/api/proxy-chat")
def proxy_chat(data: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user = require_active_user(authorization)
    if not gemini: raise HTTPException(500, "Gemini chưa được khởi tạo.")
    if not index: raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    if not data.text: raise HTTPException(400, "Tin nhắn không được để trống.")
    result = index.query(vector=embed_text(data.text), top_k=data.top_k, include_metadata=True,
                         namespace=data.knowledge_namespace or "__default__")
    contexts=[]; source_meta=[]
    for m in result.matches:
        md=m.metadata or {}; txt=md.get("text",md.get("content",""))
        if txt: contexts.append(txt); source_meta.append(md)
    learning=[]; catalog=[]
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT subject,lesson,topic,item_key,score,status,last_studied_at FROM learning_progress WHERE user_id=%s ORDER BY last_studied_at DESC LIMIT 100",(user["id"],))
            learning=[dict(x) for x in cur.fetchall()]
            cur.execute("SELECT subject,lesson,lesson_pages,topic,topic_pages,question_pages,answer_pages,source_file,namespace FROM knowledge_documents ORDER BY subject,lesson,topic,id")
            catalog=[dict(x) for x in cur.fetchall()]
    finally: conn.close()
    prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật cá nhân.

Khi người học hỏi chung như 'hôm nay học gì', hãy dựa vào DANH MỤC GIÁO TRÌNH để hỏi lần lượt: môn học -> bài học -> chủ đề -> luyện câu hỏi. Chỉ dùng tên môn/bài/chủ đề có trong danh mục. Nếu user muốn học mới, ưu tiên nội dung chưa có trong lịch sử; nếu muốn ôn, ưu tiên nội dung điểm thấp hoặc lâu chưa học. Metadata trang là nguồn tham chiếu: bài học, chủ đề, câu hỏi, đáp án. Không được bịa trang. Với câu hỏi, không tiết lộ đáp án trước khi user trả lời. Dạy theo tài liệu, có thể giải thích bằng tiếng Việt.

DANH MỤC GIÁO TRÌNH:
{json.dumps(catalog,ensure_ascii=False,default=str)}

LỊCH SỬ HỌC:
{json.dumps(learning,ensure_ascii=False,default=str)}

NỘI DUNG TÌM ĐƯỢC:
{chr(10).join(contexts)}

TIN NHẮN:
{data.text}"""
    response=gemini.models.generate_content(model=GEMINI_MODEL,contents=prompt,config=types.GenerateContentConfig(temperature=0.2))
    return {"reply":response.text or "","model":GEMINI_MODEL,"sources":source_meta[:10],"learning_history_count":len(learning)}

@app.post("/learning/progress")
def save_learning_progress(data: dict, authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization); subject=str(data.get("subject","")).strip()
    if not subject: raise HTTPException(400,"subject là bắt buộc.")
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("INSERT INTO learning_progress(subject,lesson,topic,item_key,score,status,user_id) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *",(subject,str(data.get("lesson","")).strip(),str(data.get("topic","")).strip(),str(data.get("item_key","")).strip(),data.get("score"),str(data.get("status","studied")),user["id"]))
            row=dict(cur.fetchone())
        conn.commit(); return {"success":True,"progress":row}
    finally: conn.close()

@app.get("/learning/summary")
def learning_summary(authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization); conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT subject,lesson,topic,item_key,score,status,last_studied_at FROM learning_progress WHERE user_id=%s ORDER BY last_studied_at DESC LIMIT 200",(user["id"],))
            return {"success":True,"user_id":user["id"],"learning_history":[dict(x) for x in cur.fetchall()]}
    finally: conn.close()

@app.get("/learning/catalog")
def learning_catalog(authorization: Optional[str] = Header(default=None)):
    require_active_user(authorization); conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT subject,lesson,lesson_pages,topic,topic_pages,question_pages,answer_pages,source_file,namespace FROM knowledge_documents ORDER BY subject,lesson,topic,id")
            return {"success":True,"documents":[dict(x) for x in cur.fetchall()]}
    finally: conn.close()

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
.small{font-size:13px;color:#666}
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
<div style="display:flex;gap:8px;flex-wrap:wrap">
<input id="subject" value="Tiếng Nhật" placeholder="Môn học *" required style="flex:1">
<input id="lesson" placeholder="Bài học" style="flex:1"><input id="lessonPages" placeholder="Trang bài học: 1-10" style="flex:1">
<input id="topic" placeholder="Chủ đề" style="flex:1"><input id="topicPages" placeholder="Trang chủ đề: 3-5" style="flex:1">
<input id="questionPages" placeholder="Trang câu hỏi: 8-10" style="flex:1"><input id="answerPages" placeholder="Trang đáp án: 20-21" style="flex:1">
<input id="chunkSize" type="number" value="1200" min="300" max="5000" style="width:120px">
<input id="overlap" type="number" value="200" min="0" max="4900" style="width:110px">
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
async function uploadKnowledge(event){
  event.preventDefault();
  const file=document.getElementById("pdfFile").files[0];
  if(!file)return;
  const status=document.getElementById("uploadStatus"), btn=document.getElementById("uploadBtn");
  btn.disabled=true;
  status.textContent="⏳ Đang xử lý PDF và upload Pinecone...";
  try{
    const fd=new FormData();
    fd.append("file",file);
    const params=new URLSearchParams({password:pw,subject:document.getElementById("subject").value.trim(),lesson:document.getElementById("lesson").value.trim(),lesson_pages:document.getElementById("lessonPages").value.trim(),topic:document.getElementById("topic").value.trim(),topic_pages:document.getElementById("topicPages").value.trim(),question_pages:document.getElementById("questionPages").value.trim(),answer_pages:document.getElementById("answerPages").value.trim(),chunk_size:document.getElementById("chunkSize").value||1200,overlap:document.getElementById("overlap").value||200});
    fd.append("overlap",document.getElementById("overlap").value||200);
    const r=await fetch("/admin/api/knowledge/upload?"+params.toString(),{method:"POST",body:fd});
    const t=await r.text(); let d={}; try{d=JSON.parse(t)}catch{d={detail:t}}
    if(!r.ok)throw Error(d.detail||("HTTP "+r.status));
    status.textContent=`✅ ${d.filename}: ${d.pages} trang · ${d.chunks} chunks · ${d.dimension} dimensions`;
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

@app.post("/admin/api/knowledge/upload")
async def admin_knowledge_upload(
    password: str, file: UploadFile = File(...),
    subject: str = "", lesson: str = "", lesson_pages: str = "",
    topic: str = "", topic_pages: str = "", question_pages: str = "", answer_pages: str = "",
    chunk_size: int = 1200, overlap: int = 200
):
    check_admin(password)
    if not subject.strip(): raise HTTPException(400, "Môn học là bắt buộc.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Vui lòng chọn file PDF.")
    if not gemini: raise HTTPException(500, "GEMINI_API_KEY chưa được cấu hình.")
    if not index: raise HTTPException(500, "Pinecone chưa được khởi tạo.")
    if chunk_size < 300 or chunk_size > 5000:
        raise HTTPException(400, "chunk_size phải từ 300 đến 5000.")
    if overlap < 0 or overlap >= chunk_size:
        raise HTTPException(400, "overlap phải >= 0 và nhỏ hơn chunk_size.")

    raw=await file.read()
    if len(raw)>50*1024*1024:
        raise HTTPException(400, "File quá lớn. Giới hạn 50 MB.")
    try:
        reader=PdfReader(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Không đọc được PDF: {e}")

    records=[]; total=0; namespace="__default__"
    source_file=os.path.basename(file.filename)
    try:
        for page_no,page in enumerate(reader.pages,1):
            chunks=kb_chunk_text(page.extract_text() or "", chunk_size, overlap)
            for chunk_no,chunk in enumerate(chunks):
                vector=embed_text(chunk)
                records.append({
                    "id":uuid.uuid4().hex,
                    "values":vector,
                    "metadata":{
                        "text":chunk,
                        "course":subject.strip(), "subject":subject.strip(),
                        "lesson":lesson.strip(), "lesson_pages":lesson_pages.strip(),
                        "topic":topic.strip(), "topic_pages":topic_pages.strip(),
                        "question_pages":question_pages.strip(), "answer_pages":answer_pages.strip(),
                        "source_file":source_file,
                        "page":page_no,
                        "chunk_index":chunk_no
                    }
                })
                total+=1
                if len(records)>=50:
                    index.upsert(vectors=records, namespace=namespace)
                    records=[]
        if records:
            index.upsert(vectors=records, namespace=namespace)
    except Exception as e:
        raise HTTPException(500, f"Lỗi embedding/Pinecone: {e}")

    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO knowledge_documents(source_file,subject,lesson,lesson_pages,topic,topic_pages,question_pages,answer_pages,namespace) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",(source_file,subject.strip(),lesson.strip(),lesson_pages.strip(),topic.strip(),topic_pages.strip(),question_pages.strip(),answer_pages.strip(),namespace))
        conn.commit()
    finally: conn.close()
    return {
        "success":True,"filename":source_file,"subject":subject.strip(),"lesson":lesson.strip(),"topic":topic.strip(),"pages":len(reader.pages),
        "chunks":total,"dimension":768,"index":PINECONE_INDEX,"namespace":namespace
    }

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

@app.get("/health")
def health():
    return {"status":"ok","pinecone":index is not None,"gemini":gemini is not None,
            "database":bool(DATABASE_URL),"gemini_model":GEMINI_MODEL}
