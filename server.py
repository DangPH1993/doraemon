# VERSION: v19_72 — v19_71 + OpenAI/GPT token usage logging
# VERSION: v19_66 — strict whole-message Japanese response language fix
# VERSION: v19_64 — DB-direct vocabulary factual follow-up + pronunciation flow
BASELINE_VERSION = "19.48-curriculum-step-delete-image-state"
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
import base64
import calendar
import hashlib
import tempfile
import gc
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from pinecone import Pinecone
from google import genai
from google.genai import types
from pypdf import PdfReader

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

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
# LLM provider for chat generation only. RAG embeddings / PDF vision ingestion remain
# on the existing Gemini pipeline so the current Pinecone index and knowledge base
# stay fully compatible.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_LOW = os.getenv("OPENAI_MODEL_LOW", "gpt-4.1-mini")
OPENAI_MODEL_MEDIUM = os.getenv("OPENAI_MODEL_MEDIUM", "gpt-5-mini")
OPENAI_REASONING_MEDIUM = os.getenv("OPENAI_REASONING_MEDIUM", "medium")

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_THINKING_LEVEL = "low"
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
print("[DORAEMON SERVER FINGERPRINT] 19.66-one-exchange-genai-context")
SERVER_VERSION = "2026-08-24-v19_66-one-exchange-genai-context"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pc = None
index = None
gemini = None
openai_client = None
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
def _study_confirmation_hard_gate(
    *,
    study_confirmed: bool = False,
    lesson: str | None = None,
    explicit_confirmation: bool = False,
) -> bool:
    """
    HARD GATE:
    No explicit user confirmation of a specific lesson => no embedding,
    Pinecone, RAG, or images. Active lesson/history must never bypass this gate.
    """
    return bool(study_confirmed and explicit_confirmation and lesson)


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
                image_key TEXT NOT NULL, image_hash VARCHAR(128), image_url TEXT, description TEXT,
                term TEXT, reading TEXT, meaning TEXT, associated_text TEXT,
                bbox TEXT, width INTEGER, height INTEGER, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS user_learning_state (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                welcome_seen BOOLEAN NOT NULL DEFAULT FALSE,
                reset_count INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_chatbox_id VARCHAR(128);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_assets (
                id BIGSERIAL PRIMARY KEY, source_file VARCHAR(500) NOT NULL, content_hash VARCHAR(128) NOT NULL,
                subject VARCHAR(255) NOT NULL, page_count INTEGER NOT NULL DEFAULT 0,
                status VARCHAR(20) NOT NULL DEFAULT 'READY', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(source_file, content_hash)
            );""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_vision_cache (
                id BIGSERIAL PRIMARY KEY, asset_id BIGINT REFERENCES knowledge_assets(id) ON DELETE CASCADE,
                source_file VARCHAR(500) NOT NULL, content_type VARCHAR(30), lesson VARCHAR(255), topic VARCHAR(255),
                page INTEGER, chunk_index INTEGER, image_key TEXT NOT NULL, image_url TEXT,
                vision_json JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(source_file, image_key)
            );""")
            cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_lesson_cache (
                id BIGSERIAL PRIMARY KEY, asset_id BIGINT REFERENCES knowledge_assets(id) ON DELETE CASCADE,
                source_file VARCHAR(500) NOT NULL, subject VARCHAR(255) NOT NULL, content_type VARCHAR(30) NOT NULL,
                lesson VARCHAR(255) NOT NULL, topic VARCHAR(255), status VARCHAR(20) NOT NULL DEFAULT 'READY',
                cache_json JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(source_file, content_type, lesson, topic)
            );""")
            # Safe migrations for any cache tables created by intermediate builds.
            for sql in [
                "ALTER TABLE knowledge_assets ADD COLUMN IF NOT EXISTS content_hash VARCHAR(128);",
                "ALTER TABLE knowledge_assets ADD COLUMN IF NOT EXISTS subject VARCHAR(255);",
                "ALTER TABLE knowledge_assets ADD COLUMN IF NOT EXISTS page_count INTEGER DEFAULT 0;",
                "ALTER TABLE knowledge_assets ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'READY';",
                "ALTER TABLE knowledge_assets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS asset_id BIGINT REFERENCES knowledge_assets(id) ON DELETE CASCADE;",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS content_type VARCHAR(30);",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS lesson VARCHAR(255);",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS topic VARCHAR(255);",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS page INTEGER;",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS chunk_index INTEGER;",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS image_hash VARCHAR(128);",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS image_url TEXT;",
                "ALTER TABLE knowledge_vision_cache ADD COLUMN IF NOT EXISTS vision_json JSONB DEFAULT '{}'::jsonb;",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS asset_id BIGINT REFERENCES knowledge_assets(id) ON DELETE CASCADE;",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS source_file VARCHAR(500) DEFAULT '';",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS subject VARCHAR(255);",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS content_type VARCHAR(30);",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS lesson VARCHAR(255);",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS topic VARCHAR(255);",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'READY';",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS cache_json JSONB DEFAULT '{}'::jsonb;",
                "ALTER TABLE knowledge_lesson_cache ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();",
            ]:
                cur.execute(sql)

            cur.execute("""CREATE TABLE IF NOT EXISTS study_plans (
                id BIGSERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                version INTEGER NOT NULL DEFAULT 1, status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',
                goal_name TEXT NOT NULL, content_type VARCHAR(30) NOT NULL DEFAULT 'Giáo trình',
                scope TEXT NOT NULL DEFAULT '', start_date DATE NOT NULL, target_date DATE,
                units_per_day NUMERIC(8,3), days_per_unit INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), confirmed_at TIMESTAMPTZ, superseded_at TIMESTAMPTZ
            );""")
            cur.execute("""ALTER TABLE study_plans ADD COLUMN IF NOT EXISTS parent_plan_id BIGINT REFERENCES study_plans(id) ON DELETE SET NULL;""")
            cur.execute("""CREATE TABLE IF NOT EXISTS study_plan_items (
                id BIGSERIAL PRIMARY KEY, study_plan_id BIGINT NOT NULL REFERENCES study_plans(id) ON DELETE CASCADE,
                plan_date DATE NOT NULL, unit_index INTEGER NOT NULL, lesson VARCHAR(255) NOT NULL,
                target TEXT NOT NULL DEFAULT '', status VARCHAR(20) NOT NULL DEFAULT 'pending', completed_at TIMESTAMPTZ
            );""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS learning_mode VARCHAR(20);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN NOT NULL DEFAULT FALSE;""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_active BOOLEAN NOT NULL DEFAULT FALSE;""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_content_type VARCHAR(30);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_course VARCHAR(255);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_lesson VARCHAR(255);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_topic VARCHAR(255);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_started_at TIMESTAMPTZ;""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_end_prompt_pending BOOLEAN NOT NULL DEFAULT FALSE;""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS pending_plan_content_type VARCHAR(30);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS pending_plan_scope VARCHAR(255);""")
            cur.execute("""ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS pending_plan_created_at TIMESTAMPTZ;""")

            cur.execute("""CREATE TABLE IF NOT EXISTS daily_question_usage (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                usage_date DATE NOT NULL,
                question_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(user_id, usage_date)
            );""")
            cur.execute("""CREATE TABLE IF NOT EXISTS payment_packages (
                months INTEGER PRIMARY KEY,
                plan_name VARCHAR(50) NOT NULL,
                price_vnd BIGINT NOT NULL DEFAULT 0,
                qr_key TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );""")
            cur.execute("""INSERT INTO payment_packages(months,plan_name,price_vnd,qr_key)
                           VALUES
                           (1,'1 tháng',0,NULL),
                           (3,'3 tháng',0,NULL),
                           (6,'6 tháng',0,NULL)
                           ON CONFLICT(months) DO NOTHING;""")

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
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS study_session_chatbox_id VARCHAR(128);",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_step INTEGER NOT NULL DEFAULT 0;",
                """DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='user_learning_state' AND column_name='curriculum_waiting'
                    ) THEN
                        ALTER TABLE user_learning_state ADD COLUMN curriculum_waiting VARCHAR(50) DEFAULT 'continue';
                    ELSIF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='user_learning_state' AND column_name='curriculum_waiting' AND data_type='boolean'
                    ) THEN
                        ALTER TABLE user_learning_state ALTER COLUMN curriculum_waiting TYPE VARCHAR(50)
                        USING CASE WHEN curriculum_waiting IS TRUE THEN 'continue' ELSE 'answer' END;
                    END IF;
                END $$;""",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_exercise_answered BOOLEAN NOT NULL DEFAULT FALSE;",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_global_exercise_question TEXT NOT NULL DEFAULT '';",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_global_exercise_evidence TEXT NOT NULL DEFAULT '';",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_summary_notes TEXT NOT NULL DEFAULT '';",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_intro_history TEXT NOT NULL DEFAULT '';",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_intro_b0b1_history TEXT NOT NULL DEFAULT '';",
                "ALTER TABLE user_learning_state ADD COLUMN IF NOT EXISTS curriculum_global_exercise_result TEXT NOT NULL DEFAULT '';",
            ]:
                cur.execute(sql)
            cur.execute("ALTER TABLE user_learning_state ALTER COLUMN curriculum_waiting TYPE VARCHAR(50)")
            cur.execute("ALTER TABLE user_learning_state ALTER COLUMN curriculum_waiting SET DEFAULT 'continue';")
            cur.execute("UPDATE user_learning_state SET curriculum_waiting='continue' WHERE curriculum_waiting IS NULL OR curriculum_waiting='';")
            cur.execute("UPDATE knowledge_vision_cache SET image_hash=md5(image_key) WHERE image_hash IS NULL AND image_key IS NOT NULL;")
            cur.execute("UPDATE learning_progress SET last_studied_at=NOW() WHERE last_studied_at IS NULL;")
            cur.execute("UPDATE learning_progress SET subject='' WHERE subject IS NULL;")
            cur.execute("UPDATE learning_progress SET content_type='Từ vựng' WHERE content_type IS NULL OR TRIM(content_type)='';")
            cur.execute("UPDATE knowledge_documents SET content_type='Từ vựng' WHERE content_type IS NULL OR TRIM(content_type)='';")
            # Backfill cache source_file from its asset when migrating intermediate schemas that
            # created knowledge_lesson_cache without source_file. This lets legacy cache rows
            # participate in the new runtime lookup without requiring PDF re-upload.
            try:
                cur.execute("""
                    UPDATE knowledge_lesson_cache kc
                    SET source_file = ka.source_file
                    FROM knowledge_assets ka
                    WHERE kc.asset_id = ka.id
                      AND (kc.source_file IS NULL OR TRIM(kc.source_file) = '')
                """)
            except Exception as exc:
                print("[KNOWLEDGE CACHE MIGRATION] source_file backfill skipped:", type(exc).__name__, str(exc))
            cur.execute("UPDATE knowledge_lesson_cache SET source_file='' WHERE source_file IS NULL;")
            cur.execute("UPDATE knowledge_lesson_cache SET status='READY' WHERE status IS NULL OR TRIM(status)='';")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_lesson_cache_scope ON knowledge_lesson_cache(content_type,lesson,topic,status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_vision_cache_scope ON knowledge_vision_cache(source_file,lesson,topic,page);")
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
    global pc, index, gemini, openai_client, b2
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
    if GEMINI_API_KEY:
        gemini = genai.Client(api_key=GEMINI_API_KEY)
    if OPENAI_API_KEY and OpenAI is not None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
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
        init_curriculum_db()
        print("PostgreSQL: OK")
    else:
        print("WARNING: DATABASE_URL chưa được cấu hình.")
    print("LLM provider:", LLM_PROVIDER)
    print("OpenAI models:", OPENAI_MODEL_LOW, "/", OPENAI_MODEL_MEDIUM, "reasoning:", OPENAI_REASONING_MEDIUM)
    print("Gemini model:", GEMINI_MODEL, "thinking_level:", GEMINI_THINKING_LEVEL)

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
    # New chatbox markers are optional so old clients remain compatible.
    chatbox_new: bool = False
    chatbox_id: str | None = None
    image_base64: str | None = None
    use_knowledge_base: bool = True
    knowledge_namespace: str = "default"
    top_k: int = 8
    proactive: bool = False
    action: str | None = None

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
    # All package expiry and daily Free quota boundaries use Vietnam time (GMT+7).
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))


def _as_vn(dt):
    """Return a timezone-aware datetime in Vietnam time."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))


def _add_calendar_months(dt, months):
    """Add calendar months, not a fixed 30-day period, preserving local wall-clock time.

    Example: 17/08 11:18 + 1 month = 17/09 11:18.
    End-of-month dates are clamped to the last valid day of the target month.
    """
    local = _as_vn(dt)
    month_index = local.month - 1 + int(months)
    year = local.year + month_index // 12
    month = month_index % 12 + 1
    day = min(local.day, calendar.monthrange(year, month)[1])
    return local.replace(year=year, month=month, day=day)


def _vn_display(dt):
    local = _as_vn(dt)
    return local.strftime("%d/%m/%Y %H:%M GMT+7") if local else None

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
    active_paid = plan != "Free" and str(sub.get("status") or "").upper() == "ACTIVE" and expires_at and expires_at > _now_local()
    if active_paid:
        return {
            "id": sub.get("id"), "plan": plan, "started_at": sub.get("started_at"),
            "expires_at": expires_at, "expires_at_vn": _vn_display(expires_at), "status": "ACTIVE",
            "daily_limit": None, "used_today": used, "remaining_today": None, "unlimited": True
        }

    return {
        "id": sub.get("id"), "plan": "Free", "started_at": sub.get("started_at"),
        "expires_at": None, "expires_at_vn": None, "status": "ACTIVE",
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


@app.get("/payments/packages")
def payment_packages(authorization: Optional[str] = Header(default=None)):
    """Return admin-configured purchase options for the logged-in user."""
    user = current_user(bearer(authorization))
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT months,plan_name,price_vnd,qr_key,updated_at FROM payment_packages WHERE months IN (1,3,6) ORDER BY months")
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    by_months = {int(r["months"]): r for r in rows}
    out = []
    for months in (1,3,6):
        r = by_months.get(months, {"months":months,"plan_name":f"{months} tháng","price_vnd":0,"qr_key":None,"updated_at":None})
        out.append({
            "months": months,
            "plan_name": r.get("plan_name") or f"{months} tháng",
            "price_vnd": int(r.get("price_vnd") or 0),
            "price_display": f"{int(r.get('price_vnd') or 0):,}".replace(",", ".") + " đ" if int(r.get("price_vnd") or 0) > 0 else "Liên hệ Admin",
            "qr_url": b2_url(r.get("qr_key")) if r.get("qr_key") else None,
            "payment_content": f"{user['phone']}_mua gói {r.get('plan_name') or f'{months} tháng'}"
        })
    return {"timezone":"Asia/Ho_Chi_Minh","packages":out}


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
    """Load both legacy Knowledge Base rows and PUBLISHED curriculum rows.

    PostgreSQL is the routing source of truth: a newly published Curriculum
    lesson must become visible to the same catalog router immediately, even
    when it was created through AI Curriculum Studio rather than the legacy
    knowledge_documents uploader.
    """
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
                    UNION ALL
                    SELECT subject,content_type,lesson,NULL::VARCHAR AS lesson_pages,
                           NULL::VARCHAR AS topic,NULL::VARCHAR AS topic_pages,
                           NULL::VARCHAR AS question_pages,NULL::VARCHAR AS answer_pages,
                           source_file,'__default__'::VARCHAR AS namespace
                    FROM curriculum_lessons
                    WHERE status='PUBLISHED'
                    ORDER BY subject,content_type,lesson,topic,source_file
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

def _log_gemini_usage(response, operation="unknown", request_id=None):
    """Log Gemini usage metadata in a stable, grep-friendly format.

    Gemini generation responses expose usage_metadata with token counters.
    We log the counters without logging prompt contents, so Render logs can
    be used to audit token/cost spikes without leaking the full prompt.
    """
    try:
        usage = getattr(response, "usage_metadata", None)
        prefix = f" request={request_id}" if request_id else ""
        if usage is None:
            print(f"[GEMINI TOKENS]{prefix} operation={operation!r} usage_metadata=NONE")
            return

        def _num(name):
            value = getattr(usage, name, None)
            try:
                return int(value) if value is not None else 0
            except Exception:
                return 0

        prompt_tokens = _num("prompt_token_count")
        candidates_tokens = _num("candidates_token_count")
        thoughts_tokens = _num("thoughts_token_count")
        cached_tokens = _num("cached_content_token_count")
        tool_tokens = _num("tool_use_prompt_token_count")
        total_tokens = _num("total_token_count")

        # Some SDK/model combinations may omit total_token_count. Keep the
        # logged value useful rather than silently showing zero.
        if total_tokens == 0:
            total_tokens = prompt_tokens + candidates_tokens + thoughts_tokens

        print(
            f"[GEMINI TOKENS] operation={operation!r}{prefix} "
            f"model={GEMINI_MODEL!r} "
            f"input={prompt_tokens} output={candidates_tokens} "
            f"thoughts={thoughts_tokens} cached={cached_tokens} "
            f"tool_prompt={tool_tokens} total={total_tokens}"
        )
    except Exception as exc:
        print("[GEMINI TOKENS] logging failed:", type(exc).__name__, str(exc))




def _log_openai_usage(response, operation="chat_generation", request_id=None):
    """Log OpenAI Responses API token usage without logging prompt content.

    Handles SDK objects as well as dict-like usage payloads and keeps nested
    cached/reasoning counters when the API exposes them.
    """
    try:
        usage = getattr(response, "usage", None)
        prefix = f" request={request_id}" if request_id else ""
        if usage is None:
            print(f"[OPENAI TOKENS]{prefix} operation={operation!r} usage=NONE")
            return

        def _get(obj, name, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        def _num(value):
            try:
                return int(value) if value is not None else 0
            except Exception:
                return 0

        input_tokens = _num(_get(usage, "input_tokens"))
        output_tokens = _num(_get(usage, "output_tokens"))
        total_tokens = _num(_get(usage, "total_tokens"))

        input_details = _get(usage, "input_tokens_details")
        output_details = _get(usage, "output_tokens_details")
        cached_tokens = _num(_get(input_details, "cached_tokens"))
        reasoning_tokens = _num(_get(output_details, "reasoning_tokens"))

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        print(
            f"[OPENAI TOKENS] operation={operation!r}{prefix} "
            f"model={getattr(response, 'model', None) or 'unknown'!r} "
            f"input={input_tokens} output={output_tokens} "
            f"reasoning={reasoning_tokens} cached={cached_tokens} "
            f"total={total_tokens}"
        )
    except Exception as exc:
        # Token logging must never break a successful model response.
        print("[OPENAI TOKENS] logging failed:", type(exc).__name__, str(exc))

def embed_text(text):
    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    r = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    # Embedding responses may expose usage_metadata depending on the SDK/API
    # version. Log it when available; never fail an embedding because logging
    # is unsupported.
    _log_gemini_usage(r, operation="embedding")
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
        _sync_active_plan_completion(user_id, row)
        return row
    finally:
        conn.close()




def _sync_active_plan_completion(user_id, row):
    if not row or str(row.get('status') or '').lower() != 'completed':
        return
    lesson=str(row.get('lesson') or '').strip()
    if not lesson:
        return
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE study_plan_items i SET status='completed', completed_at=NOW()
                FROM study_plans p WHERE i.study_plan_id=p.id AND p.user_id=%s AND p.status='ACTIVE'
                AND lower(i.lesson)=lower(%s) AND lower(coalesce(p.content_type,''))=lower(coalesce(%s,''))
                AND i.status<>'completed'""",(user_id,lesson,row.get('content_type') or ''))
        conn.commit()
    finally: conn.close()

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



def _encode_lesson_confirm_scope(scope):
    payload = {
        "course": scope.get("course"),
        "content_type": scope.get("content_type"),
        "lesson": scope.get("lesson"),
        "topic": scope.get("topic"),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_lesson_confirm_scope(value):
    try:
        raw = str(value or "")
        raw += "=" * (-len(raw) % 4)
        data = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
        return {
            "course": str(data.get("course") or "").strip() or None,
            "content_type": _normalize_content_type(data.get("content_type")) if data.get("content_type") else None,
            "lesson": str(data.get("lesson") or "").strip() or None,
            "topic": str(data.get("topic") or "").strip() or None,
        }
    except Exception as exc:
        print("[LESSON CONFIRM] decode failed:", type(exc).__name__, str(exc))
        return None


def _is_specific_lesson_request(text: str) -> bool:
    q = str(text or "").strip().casefold()
    if not q:
        return False
    markers = (
        "muốn học bài", "muon hoc bai", "muốn dạy bài", "muon day bai",
        "dạy mình bài", "day minh bai", "học bài ", "hoc bai ",
        "học giáo trình", "hoc giao trinh", "học ngữ pháp", "hoc ngu phap",
        "học bài tập", "hoc bai tap", "học từ vựng", "hoc tu vung",
        "học truyện", "hoc truyen", "học kanji", "hoc kanji",
    )
    return any(m in q for m in markers) and bool(re.search(r"\bbài\s*\S+|\btopic\b|\bkanji\b|\bbộ thủ\b", q))


def _lesson_suggestions(catalog, content_type=None, limit=5):
    rows = []
    seen = set()
    for item in catalog or []:
        ct = _normalize_content_type(item.get("content_type"))
        if content_type and ct != content_type:
            continue
        lesson = str(item.get("lesson") or "").strip()
        topic = str(item.get("topic") or "").strip()
        if not lesson:
            continue
        key = (ct.casefold(), lesson.casefold(), topic.casefold())
        if key in seen:
            continue
        seen.add(key)
        label = lesson + (f" – {topic}" if topic else "")
        rows.append((ct, lesson, topic, label))
        if len(rows) >= limit:
            break
    return rows


def _select_active_scope(query_text, text_matches, catalog):
    """
    Determine the active learning scope in strict order:
    explicit content type -> explicitly named lesson/topic -> supporting course.

    IMPORTANT:
    A content type mention such as "mình muốn học giáo trình" must NEVER
    inherit an arbitrary lesson from the first catalog row. Lesson/topic are
    only populated when the learner actually names them. Otherwise the request
    stays at content-type level and can be handled as a routing/selection turn.
    """
    q = _clean_scope_value(query_text)

    course = None
    for item in catalog or []:
        c = str(item.get("course") or item.get("course_name") or "").strip()
        if c and _clean_scope_value(c) in q:
            course = c
            break

    # Content type is explicit and authoritative.
    explicit_patterns = [
        ("Giáo trình", ["giáo trình", "học theo giáo trình", "học giáo trình", "theo giáo trình", "trong giáo trình"]),
        ("Truyện đọc", ["truyện đọc", "đọc truyện", "câu chuyện", "học truyện"]),
        ("Bài tập", ["bài tập", "làm bài", "bài quiz", "quiz"]),
        ("Ngữ pháp", ["ngữ pháp", "học ngữ pháp", "ôn ngữ pháp", "grammar"]),
        ("Từ vựng", ["từ vựng", "học từ vựng", "từ mới", "học từ mới", "vocabulary"]),
    ]
    explicit_type = None
    for typ, keys in explicit_patterns:
        if any(k in q for k in keys):
            explicit_type = typ
            break

    # Always initialize lesson/topic before any alias/catalog matching.
    # Short conversational turns such as "hôm nay hơi mệt nhưng sẽ cố học" are
    # valid thread messages but may contain no lesson name. Initializing these
    # variables prevents thread-scope extraction from crashing with
    # UnboundLocalError.
    lesson = None
    topic = None

    # Find a lesson/topic ONLY when its actual name appears in the query.
    named_candidates = []
    for item in catalog or []:
        item_course = str(item.get("course") or item.get("course_name") or "").strip()
        item_type = _normalize_content_type(item.get("content_type"))
        item_lesson = str(item.get("lesson") or "").strip()
        item_topic = str(item.get("topic") or "").strip()

        if explicit_type and item_type != explicit_type:
            continue
        if course and item_course and item_course != course:
            continue

        lesson_n = _clean_scope_value(item_lesson)
        topic_n = _clean_scope_value(item_topic)
        lesson_hit = bool(lesson_n and lesson_n in q)
        topic_hit = bool(topic_n and topic_n in q)

        if not lesson_hit and not topic_hit:
            continue

        score = 0
        if topic_hit:
            score += 1000 + len(topic_n) * 10
        if lesson_hit:
            score += 500 + len(lesson_n) * 5
        if item_course and course:
            score += 50

        named_candidates.append((score, item))

    # Numeric lesson alias support: users naturally say "Bài 3", while
    # imported catalogs can contain identifiers such as "Bài 3v4".
    # Resolve the numeric lesson against the catalog ONLY when the user
    # explicitly names a lesson number, and keep the selected content type.
    numeric_match = re.search(r"\bbài\s*(\d+)\b", q, flags=re.IGNORECASE)
    if numeric_match and not named_candidates:
        lesson_no = numeric_match.group(1)
        numeric_candidates = []
        for item in catalog or []:
            item_type = _normalize_content_type(item.get("content_type"))
            item_lesson = str(item.get("lesson") or "").strip()
            if explicit_type and item_type != explicit_type:
                continue
            m_item = re.search(r"\bbài\s*(\d+)\b", _clean_scope_value(item_lesson), flags=re.IGNORECASE)
            if m_item and m_item.group(1) == lesson_no:
                numeric_candidates.append(item)
        if numeric_candidates:
            # Prefer the cleanest canonical-looking lesson name, then the
            # first catalog item for deterministic behavior.
            numeric_candidates.sort(key=lambda item: (
                len(str(item.get("lesson") or "")),
                str(item.get("lesson") or "").casefold(),
            ))
            best = numeric_candidates[0]
            lesson = str(best.get("lesson") or "").strip() or None
            topic = None
            if not explicit_type:
                explicit_type = _normalize_content_type(best.get("content_type"))

    if named_candidates:
        named_candidates.sort(key=lambda x: x[0], reverse=True)
        best_item = named_candidates[0][1]
        lesson = str(best_item.get("lesson") or "").strip() or None
        topic = str(best_item.get("topic") or "").strip() or None
        if not explicit_type:
            explicit_type = _normalize_content_type(best_item.get("content_type"))

    # Kanji/Bộ thủ are vocabulary lessons, never standalone content types.
    if any(k in q for k in ["kanji", "học kanji"]):
        explicit_type = "Từ vựng"
        if not lesson:
            lesson = "Kanji"
    elif any(k in q for k in ["bộ thủ", "học bộ thủ", "radical"]):
        explicit_type = "Từ vựng"
        if not lesson:
            lesson = "Bộ thủ"

    # Only use top RAG metadata when there is NO explicit content-type signal.
    # This prevents "mình muốn học giáo trình" from inheriting an unrelated
    # grammar/exercise lesson merely because of catalog ordering or similarity.
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


def _is_casual_conversation_request(text: str) -> bool:
    """Detect short casual/emotional chat that does not need study retrieval.

    This is intentionally conservative: explicit study words are excluded, and
    short emotional/social utterances such as "mệt quá hic" take a lightweight
    conversational path instead of embedding/Pinecone/image retrieval.
    """
    q = str(text or "").strip().casefold()
    if not q:
        return False
    study_markers = (
        "học", "bài", "giáo trình", "ngữ pháp", "từ vựng", "bộ thủ", "kanji",
        "bài tập", "truyện đọc", "lộ trình", "ôn", "luyện", "giải thích",
        "đáp án", "câu hỏi", "sai", "đúng",
    )
    if any(m in q for m in study_markers):
        return False

    casual_phrases = (
        "mệt quá", "mệt thật", "mệt ghê", "mệt quá hic", "hic", "huhu",
        "chán quá", "chán thật", "buồn quá", "buồn thật", "nản quá",
        "đuối quá", "kiệt sức", "khó chịu quá", "bực quá", "stress quá",
        "haha", "hihi", "hehe", "haiz", "thở dài", "ôi mệt", "mệt ghê",
        "hôm nay mệt", "hôm nay chán", "hôm nay buồn",
    )
    if any(p in q for p in casual_phrases):
        return True

    # Very short social utterances are safe to handle conversationally, but do
    # not swallow likely lesson follow-ups such as "vậy thì sao?".
    if len(q) <= 18 and q in {"hic", "huhu", "haiz", "haha", "hihi", "hehe", "ôi", "wow"}:
        return True
    return False


def _is_general_non_learning_request(text: str) -> bool:
    """Detect ordinary/non-learning questions that must not fall into study RAG."""
    q = str(text or "").strip().casefold()
    if not q:
        return False
    # Explicit study intent always wins.
    study_markers = (
        "học", "bài", "giáo trình", "ngữ pháp", "từ vựng", "bộ thủ", "kanji",
        "bài tập", "truyện đọc", "lộ trình", "ôn", "luyện", "giải thích",
    )
    if any(m in q for m in study_markers):
        return False
    weather_markers = (
        "thời tiết", "thoi tiet", "nhiệt độ", "nhiet do", "mưa không", "mưa không",
        "trời mưa", "trời nắng", "dự báo thời tiết", "du bao thoi tiet",
    )
    time_markers = ("mấy giờ", "may gio", "bây giờ là mấy giờ", "gio hien tai")
    date_markers = ("hôm nay ngày mấy", "hom nay ngay may", "hôm nay là ngày", "ngày hôm nay")
    if any(m in q for m in weather_markers + time_markers + date_markers):
        return True
    return False


def _is_exercise_suggestion_only_request(text: str) -> bool:
    """A learning recommendation for exercises: no lesson-image attachment."""
    q = str(text or "").strip().casefold()
    if not q:
        return False
    markers = (
        "gợi ý bài tập", "goi y bai tap", "đề xuất bài tập", "de xuat bai tap",
        "bài tập nào", "bai tap nao", "cho mình bài tập", "cho minh bai tap",
        "gợi ý một bài tập", "goi y mot bai tap",
    )
    return any(m in q for m in markers)

def _is_ambiguous_study_request(text):
    """Return True when the user asks to study but gives no target/mode.

    Examples: "tôi muốn học", "mình muốn học", "tôi muốn học nhé".
    These requests must NOT fall back to the top RAG hit or durable learning
    state, because that can silently open an arbitrary lesson such as Grammar 1.
    The assistant should ask the learner to choose: follow the roadmap, continue
    the unfinished lesson, or name a specific lesson/topic.
    """
    low = str(text or "").strip().casefold()
    if not low:
        return False

    # A concrete target/mode makes the request explicit and therefore not ambiguous.
    concrete_markers = (
        "học bài ", "học phần ", "học lesson ", "bài ", "phần ",
        "ngữ pháp", "từ vựng", "từ mới", "kanji", "bộ thủ",
        "giáo trình", "truyện đọc", "bài tập", "quiz",
        "theo lộ trình", "lộ trình", "tiếp tục", "học tiếp",
        "bài đang dở", "đang học", "ôn lại", "review",
    )
    if any(m in low for m in concrete_markers):
        return False

    generic_patterns = (
        "tôi muốn học", "mình muốn học", "minh muon hoc",
        "tôi muốn học nhé", "mình muốn học nhé",
        "muốn học", "muon hoc",
    )
    return any(low == p or low.startswith(p + " ") or low.startswith(p + ",")
               or low.startswith(p + ".") or low.startswith(p + "!")
               for p in generic_patterns)


def _is_correction_followup(text):
    """Detect a short user message that is correcting/challenging the previous answer.

    This is intentionally conservative: it is for phrases such as
    "chiều thứ 6 Ken có lịch rồi mà", "cậu nói sai", "không đúng", etc.
    A correction must stay in the current lesson/conversation instead of being
    re-routed as a new study request.
    """
    low = str(text or "").strip().casefold()
    if not low:
        return False
    phrases = (
        "không đúng", "sai rồi", "cậu sai", "doraemon sai", "bạn sai",
        "nhầm rồi", "cậu nhầm", "doraemon nhầm", "không phải",
        "đã có lịch", "có lịch rồi", "có lịch mà", "đang có lịch",
        "lịch rồi mà", "đã có lịch mà", "nhưng ken", "ken có lịch",
        "chiều thứ 6", "chiều thứ sáu", "thứ 6 ken", "thứ sáu ken",
        "không khớp", "không phải như vậy", "đâu có"
    )
    return any(p in low for p in phrases)


def _extract_thread_scope(recent_history, catalog):
    """
    Recover the active lesson/content scope from the OPEN chat thread.

    Priority is intentional:
      1. explicit scope stated by the student in recent turns
      2. explicit scope stated by Doraemon in recent turns
      3. no scope

    This is only conversational context. It is NOT durable learning state and
    must not leak into a new chatbox because the client sends a fresh history.
    """
    if not recent_history:
        return None

    def as_scope(item):
        return {
            "course": str(
                item.get("course") or item.get("course_name") or ""
            ).strip() or None,
            "content_type": _normalize_content_type(item.get("content_type")),
            "lesson": str(item.get("lesson") or "").strip() or None,
            "topic": str(item.get("topic") or "").strip() or None,
        }

    # Student statements are strongest evidence of what the current thread is
    # about. Search newest -> oldest so a later explicit lesson switch wins.
    for preferred_role in ("user", "model"):
        for h in reversed(recent_history):
            if h.get("role") != preferred_role:
                continue
            found = _explicit_lesson_topic(h.get("text") or "", catalog)
            if found:
                scope = as_scope(found)
                if any(scope.values()):
                    return scope

            # Fall back to explicit content-type/lesson/topic wording in the
            # same message when the catalog helper cannot find an exact item.
            candidate = _select_active_scope(h.get("text") or "", [], catalog)
            if any(candidate.values()):
                return candidate

    return None


def _catalog_next_lesson(catalog, content_type, current_lesson):
    """Return the next lesson by numeric lesson number for the active type."""
    if not current_lesson:
        return None
    ct = _normalize_content_type(content_type) if content_type else None
    m_cur = re.search(r"\bbài\s*(\d+)\b", _clean_scope_value(current_lesson), flags=re.IGNORECASE)
    if not m_cur:
        return None
    cur_no = int(m_cur.group(1))
    candidates = []
    for item in catalog or []:
        if ct and _normalize_content_type(item.get("content_type")) != ct:
            continue
        lesson = str(item.get("lesson") or "").strip()
        m = re.search(r"\bbài\s*(\d+)\b", _clean_scope_value(lesson), flags=re.IGNORECASE)
        if not m:
            continue
        no = int(m.group(1))
        if no == cur_no + 1:
            candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        len(str(item.get("lesson") or "")),
        str(item.get("lesson") or "").casefold(),
    ))
    item = candidates[0]
    return {
        "course": str(item.get("course") or item.get("course_name") or "").strip() or None,
        "content_type": _normalize_content_type(item.get("content_type")),
        "lesson": str(item.get("lesson") or "").strip() or None,
        "topic": str(item.get("topic") or "").strip() or None,
    }


def _is_explicit_thread_switch(text):
    """
    True only when the current user message clearly asks to move to another
    lesson/content scope. Ordinary follow-ups/corrections must stay in the
    current chat thread.
    """
    low = str(text or "").strip().casefold()
    if not low:
        return False
    switch_phrases = (
        "chuyển sang", "đổi sang", "sang bài", "sang phần", "sang lesson",
        "học bài mới", "học bài khác", "học phần khác", "đổi bài",
        "muốn học bài", "muốn học phần", "mình muốn học bài",
        "mình muốn học phần", "bây giờ học bài", "tiếp theo học bài",
        "học bài tiếp", "học bài tiếp theo", "học tiếp bài", "bài tiếp theo",
    )
    return any(p in low for p in switch_phrases)


def infer_learning_event(user_id, user_text, reply, catalog, learning, source_meta=None, active_scope=None):
    """Infer only learning progress, never a score. Exercises are scored via /learning/progress."""
    text = (user_text or "").strip()
    low = text.lower()
    source_meta = source_meta or []

    # A correction of the previous assistant answer is not a new learning event.
    # Do not let a keyword in the correction sentence overwrite the active lesson.
    if _is_correction_followup(text):
        return None
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

    # Lesson-scope visuals are a deliberate exception to strict chunk locking:
    # they illustrate the whole lesson/page (e.g. a doctor examining Ken), so
    # they must not be forced onto either table chunk. They still require exact
    # source_file + page + lesson identity.
    if str(md.get("image_scope") or "").strip().lower() == "lesson":
        img_lesson = _normalize_chunk_text_for_match(md.get("lesson"))
        chunk_lesson = _normalize_chunk_text_for_match(chunk_md.get("lesson"))
        return bool(img_lesson and chunk_lesson and img_lesson == chunk_lesson)

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
        raw_indices = md.get("chunk_indices")
        if raw_indices not in (None, ""):
            try:
                parsed = json.loads(raw_indices) if isinstance(raw_indices, str) else raw_indices
                if isinstance(parsed, list):
                    normalized = []
                    for item in parsed:
                        try:
                            normalized.append(int(item))
                        except Exception:
                            normalized.append(str(item).strip())
                    if chunk_index in normalized or str(chunk_index) in normalized:
                        return True
            except Exception:
                pass
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
            # Preserve the exact text-chunk scope on direct-image payloads.
            # Without these fields, the later lesson/thread image guard sees
            # lesson/topic as None and incorrectly rejects valid images even
            # when image_key is attached directly to the selected text chunk.
            "content_type": str(md.get("content_type") or "").strip(),
            "subject": str(md.get("subject") or md.get("course") or "").strip(),
            "course": str(md.get("course") or "").strip(),
            "lesson": str(md.get("lesson") or "").strip(),
            "topic": str(md.get("topic") or "").strip(),
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
    lesson_jobs = {}

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
            lesson = str(md.get("lesson") or "").strip()
            # Lesson-scope visuals are for the general/normal lesson context.
            # Do not add them to a chunk that already has an exact chunk/table
            # image; that would make a table chunk look as if its lesson image
            # belonged to the table itself.
            if lesson and not direct_keys:
                lesson_jobs.setdefault((sf, page, lesson), []).append((order, chunk))

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

    # Lesson-scope visuals are retrieved separately but only inside the exact
    # source_file + page + lesson scope of a selected text chunk. They are not
    # semantic image search and are not allowed to leak across lessons.
    def fetch_lesson_scope(item):
        (sf, page, lesson), refs = item
        order, chunk = refs[0]
        filt = {
            "record_type": {"$eq": "image"},
            "source_file": {"$eq": sf},
            "page": {"$eq": int(page) if str(page).isdigit() else page},
            "lesson": {"$eq": lesson},
            "image_scope": {"$eq": "lesson"},
        }
        try:
            res = index.query(vector=query_vector, top_k=20, include_metadata=True, namespace=namespace, filter=filt)
            found=[]
            for m in res.matches:
                md=m.metadata or {}
                if str(md.get("image_scope") or "").strip().lower() != "lesson":
                    continue
                if not _image_belongs_to_text_chunk(md, chunk["metadata"], chunk["text"]):
                    continue
                found.extend(_image_payload_from_metadata(md, getattr(m, "score", 0), order, chunk["text"]))
            print(f"[IMAGE lesson-scope] source={sf!r} page={page!r} lesson={lesson!r} candidates={len(res.matches)} accepted={len(found)}")
            return found
        except Exception as exc:
            print("[IMAGE lesson-scope] failed:", sf, page, lesson, type(exc).__name__, str(exc))
            return []

    if lesson_jobs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(lesson_jobs))) as executor:
            for found in executor.map(fetch_lesson_scope, lesson_jobs.items()):
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


def _cache_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _cache_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cache_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _upsert_upload_knowledge_cache(source_file, source_hash, subject, page_count, text_records, image_records, *, defer_curriculum_plan=False):
    """Persist upload-time knowledge with deterministic one-section-per-text-chunk ordering."""
    def _norm_topic(value):
        t = str(value or "").strip()
        return re.sub(r"\s+", " ", t) or None

    def _group_key(md):
        return (
            _normalize_content_type(md.get("content_type")),
            _canonical_lesson_key(md.get("lesson")),
            _norm_topic(md.get("topic")),
        )

    grouped = {}
    for rec in text_records:
        md = rec.get("metadata") or {}
        key = _group_key(md)
        if not key[1]:
            continue
        grouped.setdefault(key, {"sections": [], "images": [], "source_file": source_file, "subject": subject, "lesson_display": str(md.get("lesson") or "").strip(), "topic_display": str(md.get("topic") or "").strip() or None})
        grouped[key]["sections"].append({
            "chunk_index": int(md.get("chunk_index") or 0),
            "page_chunk_index": int(md.get("page_chunk_index") or 0),
            "page": md.get("page"),
            "content_unit_id": md.get("content_unit_id"),
            "text": str(rec.get("text") or ""),
            "image_keys": list(rec.get("image_keys") or []),
            "metadata": {
                "lesson_pages": md.get("lesson_pages"), "topic_pages": md.get("topic_pages"),
                "question_pages": md.get("question_pages"), "answer_pages": md.get("answer_pages"),
            },
        })
    for img in image_records:
        md = img.get("metadata") or {}
        key = _group_key(md)
        if not key[1]:
            continue
        grouped.setdefault(key, {"sections": [], "images": [], "source_file": source_file, "subject": subject, "lesson_display": str(md.get("lesson") or "").strip(), "topic_display": str(md.get("topic") or "").strip() or None})
        vision = {k:v for k,v in md.items() if k not in {"image_key","image_url"}}
        grouped[key]["images"].append({
            "image_key": md.get("image_key"), "image_url": md.get("image_url"),
            "page": md.get("page"), "chunk_index": md.get("chunk_index"),
            "vision": _cache_jsonable(vision),
        })

    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM knowledge_assets WHERE source_file=%s AND content_hash=%s ORDER BY id DESC LIMIT 1", (source_file, source_hash))
            existing_asset = cur.fetchone()
            if existing_asset:
                asset_id = existing_asset["id"]
                cur.execute("UPDATE knowledge_assets SET subject=%s,page_count=%s,status='READY',updated_at=NOW() WHERE id=%s", (subject, int(page_count), asset_id))
            else:
                cur.execute("""INSERT INTO knowledge_assets(source_file,content_hash,subject,page_count,status,updated_at)
                               VALUES(%s,%s,%s,%s,'READY',NOW()) RETURNING id""", (source_file, source_hash, subject, int(page_count)))
                asset_id = cur.fetchone()["id"]
            cur.execute("DELETE FROM knowledge_vision_cache WHERE source_file=%s", (source_file,))
            cur.execute("DELETE FROM knowledge_lesson_cache WHERE source_file=%s", (source_file,))
            # image_hash is globally unique. The same visual asset may be referenced
            # by more than one cached lesson/topic package, so de-duplicate it
            # within the upload transaction and tolerate an existing global row.
            seen_image_hashes = set()
            for key, payload in grouped.items():
                ct, lesson_key, topic = key
                lesson = payload.get("lesson_display") or lesson_key
                topic_display = payload.get("topic_display") or topic
                payload["sections"].sort(key=lambda x: (
                    int(x.get("chunk_index") if x.get("chunk_index") not in (None, "") else 10**9),
                    int(x.get("page") or 0),
                    int(x.get("page_chunk_index") or 0),
                    str(x.get("content_unit_id") or ""),
                ))
                for new_idx, section in enumerate(payload["sections"]):
                    section["chunk_index"] = new_idx
                payload["images"].sort(key=lambda x: (
                    int(x.get("page") or 0),
                    int(x.get("chunk_index") or 0),
                    str(x.get("image_key") or ""),
                ))
                if not payload["sections"]:
                    raise RuntimeError(f"Knowledge Cache không có text sections cho lesson={lesson_key!r}")
                print(
                    f"[KNOWLEDGE CACHE CHUNK AUDIT] source={source_file!r} lesson={lesson!r} "
                    f"topic={topic!r} sections={len(payload['sections'])} "
                    f"chunk_indexes={[x.get('chunk_index') for x in payload['sections']]}"
                )
                # Tiny immutable overview for the runtime prompt; full source remains in sections.
                overview = " ".join(x["text"] for x in payload["sections"][:2]).strip()[:2400]
                package = {
                    "version": 1,
                    "source_file": source_file,
                    "content_hash": source_hash,
                    "subject": subject,
                    "content_type": ct,
                    "lesson": lesson,
                    "topic": topic_display,
                    "overview": overview,
                    "sections": payload["sections"],
                    "images": payload["images"],
                }
                cur.execute("""INSERT INTO knowledge_lesson_cache(\n                    asset_id,source_file,subject,content_type,lesson,topic,status,cache_json,updated_at\n                ) VALUES(%s,%s,%s,%s,%s,%s,'READY',%s::jsonb,NOW())""",
                    (asset_id, source_file, subject, ct, lesson, topic_display, json.dumps(_cache_jsonable(package), ensure_ascii=False)))
                for img in payload["images"]:
                    image_key = str(img.get("image_key") or "").strip()
                    image_hash = str(img.get("image_hash") or "").strip() or None
                    if image_hash is None and image_key:
                        import hashlib
                        image_hash = hashlib.sha256(image_key.encode("utf-8")).hexdigest()
                    if image_hash and image_hash in seen_image_hashes:
                        print(f"[KNOWLEDGE CACHE IMAGE DEDUPE] source={source_file!r} image_hash={image_hash[:12]}...")
                        continue
                    if image_hash:
                        seen_image_hashes.add(image_hash)
                    __IMAGE_HASH__ = image_hash
                    cur.execute("""INSERT INTO knowledge_vision_cache(\n                        asset_id,source_file,content_type,lesson,topic,page,chunk_index,image_key,image_hash,image_url,vision_json\n                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)\n                    ON CONFLICT (image_hash) DO NOTHING""",
                        (asset_id,source_file,ct,lesson,topic_display,img.get("page"),img.get("chunk_index"),img.get("image_key"),__IMAGE_HASH__,img.get("image_url"),json.dumps(_cache_jsonable(img.get("vision") or {}), ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()
    print(f"[KNOWLEDGE CACHE READY] source={source_file!r} lessons={len(grouped)} images={len(image_records)} hash={source_hash[:12]}...")
    for _gk, _pl in grouped.items():
        _idxs = [int(x.get("chunk_index") or 0) for x in (_pl.get("sections") or [])]
        print(f"[KNOWLEDGE CACHE CHUNK AUDIT GLOBAL] lesson={_pl.get('lesson_display')!r} indexes={_idxs}")
    return len(grouped)


def _canonical_lesson_key(value: str) -> str:
    """Normalize lesson names so UI phrases like "bài visionv1" match cached "visionv1"."""
    s = str(value or "").strip().casefold()
    s = re.sub(r"^\s*bài\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _published_curriculum_step_text(content):
    content = content if isinstance(content, dict) else {}
    parts=[]
    main=content.get("content")
    if isinstance(main,str) and main.strip(): parts.append(main.strip())
    elif main not in (None,"",[],{}): parts.append(json.dumps(main,ensure_ascii=False,indent=2))
    items=content.get("items")
    if isinstance(items,list) and items:
        lines=[]
        for i,item in enumerate(items,1):
            if isinstance(item,dict):
                title=str(item.get("title") or item.get("question") or item.get("word") or item.get("pattern") or "").strip()
                body=str(item.get("content") or item.get("answer") or item.get("meaning") or item.get("example") or "").strip()
                if title and body: lines.append(f"{i}. {title}\n{body}")
                elif title: lines.append(f"{i}. {title}")
                elif body: lines.append(f"{i}. {body}")
            elif str(item).strip(): lines.append(f"{i}. {str(item).strip()}")
        if lines: parts.append("\n".join(lines))
    return "\n\n".join(parts).strip()


def _published_curriculum_images(content, pages=None):
    content=content if isinstance(content,dict) else {}
    inventory={}
    for page in pages or []:
        for im in page.get("images") or []:
            key=str(im.get("image_key") or "").strip()
            if key: inventory[key]=im
    out=[]; seen=set()
    for item in content.get("images") or []:
        if not isinstance(item,dict): continue
        key=str(item.get("image_key") or item.get("key") or "").strip()
        if not key or key in seen: continue
        base=inventory.get(key,{})
        vision=item.get("vision") or base.get("vision") or {}
        out.append({
            "key":key,
            "url":str(item.get("image_url") or base.get("image_url") or b2_url(key) or "").strip(),
            "page":item.get("page") or base.get("page"),
            "caption":str(item.get("caption") or vision.get("caption") or vision.get("description") or vision.get("explanation") or "").strip()
        })
        seen.add(key)
    return out


def _published_curriculum_runtime_payload(lesson_row, step_rows):
    raw_source=lesson_row.get("raw_source_json") or {}
    pages=raw_source.get("pages") if isinstance(raw_source,dict) else []
    pages=pages if isinstance(pages,list) else []
    sections=[]; images=[]
    for order,row in enumerate(step_rows):
        content=row.get("content_json") or {}
        if not isinstance(content,dict): content={}
        step_imgs=_published_curriculum_images(content,pages)
        keys=[]
        for im in step_imgs:
            keys.append(im["key"])
            images.append({"image_key":im["key"],"image_url":im["url"],"page":im["page"],"chunk_index":order,"content_unit_id":f"curriculum:{row.get('step_code')}","vision":{"caption":im.get("caption","")}})
        refs=content.get("source_refs") if isinstance(content,dict) else []
        page=None
        if isinstance(refs,list) and refs and isinstance(refs[0],dict): page=refs[0].get("page")
        text=_published_curriculum_step_text(content) or str(row.get("title") or "").strip()
        sections.append({"chunk_index":order,"page":page or order+1,"content_unit_id":f"curriculum:{row.get('step_code')}","step_code":str(row.get("step_code") or ""),"step_title":str(row.get("title") or ""),"step_type":str(row.get("step_type") or "lesson"),"text":text,"content":content,"image_keys":keys})
    return {"version":int(lesson_row.get("version") or 1),"source_file":lesson_row.get("source_file"),"content_hash":None,"subject":lesson_row.get("subject"),"content_type":lesson_row.get("content_type"),"lesson":lesson_row.get("lesson"),"topic":None,"overview":" ".join(x["text"] for x in sections[:2])[:2400],"sections":sections,"images":images,"published_curriculum":True,"lesson_id":int(lesson_row.get("id"))}


def _published_curriculum_step(cache,index):
    sections=list((cache or {}).get("sections") or [])
    if not sections: return None
    index=max(0,min(int(index),len(sections)-1))
    sec=sections[index]; by_key={str(x.get("image_key")):x for x in (cache or {}).get("images") or [] if x.get("image_key")}
    imgs=[]
    for key in sec.get("image_keys") or []:
        im=by_key.get(str(key))
        if im: imgs.append({"key":str(key),"url":im.get("image_url") or b2_url(str(key)),"page":im.get("page"),"caption":str((im.get("vision") or {}).get("caption") or "")})
    return {
        "index":index,
        "code":sec.get("step_code") or f"B{index}",
        "title":sec.get("step_title") or "",
        "text":sec.get("text") or "",
        "content":sec.get("content") if isinstance(sec.get("content"),dict) else {},
        "images":imgs,
        "is_final":str(sec.get("step_code") or "").upper()=="FINAL" or index==len(sections)-1
    }


def _published_curriculum_vocabulary_text(step):
    """Render vocabulary fields exactly as stored in curriculum_steps.content_json.
    Never ask Gemini to reconstruct spelling/reading/meaning at runtime.
    """
    content=step.get("content") if isinstance(step,dict) else {}
    content=content if isinstance(content,dict) else {}
    items=content.get("items") if isinstance(content.get("items"),list) else []
    if not items:
        return str(step.get("text") or "").strip()

    def pick(item, *keys):
        for key in keys:
            value=item.get(key) if isinstance(item,dict) else None
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    lines=[]
    for i,item in enumerate(items,1):
        if not isinstance(item,dict):
            if str(item).strip(): lines.append(f"{i}. {str(item).strip()}")
            continue
        writing=pick(item,"writing","word","term","kanji","title","text")
        reading=pick(item,"reading","hiragana","kana","pronunciation","yomikata")
        meaning=pick(item,"meaning","definition","translation","vietnamese_meaning")
        example=pick(item,"example","content")
        if writing or reading or meaning:
            row=[f"{i}. {writing}" if writing else f"{i}."]
            if reading:
                row.append(f"   📖 Cách đọc: {reading}")
                row.append(f"   🔊 Phát âm: {reading}")
            if meaning: row.append(f"   🇻🇳 Nghĩa: {meaning}")
            if example: row.append(f"   Ví dụ: {example}")
            lines.append("\n".join(row))
        else:
            fallback=pick(item,"question","pattern","answer")
            if fallback: lines.append(f"{i}. {fallback}")
    intro=str(content.get("content") or "").strip()
    # If the content is only a generic intro, keep it; the structured rows are the
    # authoritative spelling/reading/meaning payload.
    return "\n\n".join([x for x in (intro,"\n".join(lines)) if x]).strip() or str(step.get("text") or "").strip()



def _vocab_direct_answer(step, question_text):
    """Cheap DB-only answers for factual vocabulary lookups.

    Returns a response when the learner only asks for spelling/reading/pronunciation/
    meaning of a vocabulary item already present in the published DB. Returns None
    for comparison, explanation, examples, or anything that needs GenAI.
    """
    if not isinstance(step, dict):
        return None
    q = str(question_text or "").strip()
    if not q:
        return None
    q_fold = q.casefold()
    direct_markers = (
        "phát âm", "phat am", "đọc như nào", "đọc thế nào", "đọc sao", "đọc là gì",
        "cách đọc", "cach doc", "đọc như thế nào", "nghĩa là gì", "nghĩa gì",
        "nghĩa tiếng việt", "tiếng việt là gì", "dịch nghĩa", "viết như nào",
        "viết thế nào", "chữ gì", "cách viết", "意味", "読み方", "発音", "どう読む",
        "何の意味", "ベトナム語"
    )
    if not any(m in q_fold for m in direct_markers):
        return None
    # Explicit explanatory/comparison requests must remain GenAI turns.
    complex_markers = ("khác nhau", "so sánh", "tại sao", "vì sao", "giải thích", "ví dụ", "đặt câu", "phân biệt")
    if any(m in q_fold for m in complex_markers):
        return None

    content = step.get("content") if isinstance(step.get("content"), dict) else {}
    items = content.get("items") if isinstance(content.get("items"), list) else []
    if not items:
        return None

    def pick(item, *keys):
        for key in keys:
            value = item.get(key) if isinstance(item, dict) else None
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    # Match the Japanese token/reading appearing in the user's question.
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        writing = pick(item, "writing", "word", "term", "kanji", "title", "text")
        reading = pick(item, "reading", "hiragana", "kana", "pronunciation", "yomikata")
        meaning = pick(item, "meaning", "definition", "translation", "vietnamese_meaning")
        pronunciation = pick(item, "pronunciation", "reading", "hiragana", "kana", "yomikata")
        if not (writing or reading or meaning):
            continue
        candidates.append((writing, reading, pronunciation, meaning))

    # Prefer an exact token; otherwise only use a unique contained Japanese token.
    matched = None
    for row in candidates:
        writing, reading, pronunciation, meaning = row
        if writing and writing.casefold() in q_fold:
            matched = row
            break
        if reading and reading.casefold() in q_fold:
            matched = row
            break
    if matched is None:
        return None

    writing, reading, pronunciation, meaning = matched
    asks_meaning = any(x in q_fold for x in ("nghĩa", "意味", "ベトナム語", "tiếng việt", "dịch"))
    asks_reading = any(x in q_fold for x in ("đọc", "cách đọc", "読み方", "どう読む"))
    asks_pron = any(x in q_fold for x in ("phát âm", "phat am", "発音"))
    asks_writing = any(x in q_fold for x in ("viết", "cách viết", "chữ", "書き方"))

    lines = []
    if writing:
        lines.append(f"📝 **Chữ:** {writing}")
    if reading and (asks_reading or asks_pron or not (asks_meaning or asks_writing)):
        lines.append(f"📖 **Cách đọc:** {reading}")
    if pronunciation and asks_pron:
        lines.append(f"🔊 **Phát âm:** {pronunciation}")
    if meaning and asks_meaning:
        lines.append(f"🇻🇳 **Nghĩa:** {meaning}")
    if not lines:
        return None
    return "\n".join(lines)


def _vocab_direct_answer_from_cache(cache, current_step, question_text):
    """Search all published vocabulary steps, not only the currently displayed B0/B1 step."""
    sections=list((cache or {}).get("sections") or [])
    order=[]
    if current_step is not None:
        try: order.append(int(current_step))
        except Exception: pass
    order.extend(i for i in range(len(sections)) if i not in order)
    for idx in order:
        try:
            sec=_published_curriculum_step(cache, idx)
        except Exception:
            continue
        ans=_vocab_direct_answer(sec, question_text)
        if ans:
            return ans
    return None

def _published_curriculum_answer_step(cache):
    sections=list((cache or {}).get("sections") or [])
    for idx,sec in enumerate(sections):
        if str(sec.get("step_code") or "").upper() in {"B2","ANSWER"}:
            return _published_curriculum_step(cache,idx)
    return None


def _published_curriculum_non_giao_trinh_blocks(step, cache, content_type, *, answered=False):
    """Deterministic DB-first UI for Từ vựng/Ngữ pháp/Bài tập/Truyện đọc.

    All visible teaching/answer text comes from curriculum_steps.content_json.
    The only generated UI text is navigation chrome (labels/prompts).
    """
    if not step: return []
    ct=str(content_type or "").strip()
    blocks=[]
    title=f"**{step['code']} · {step['title']}**" if step.get("title") else f"**{step['code']}**"
    text=_published_curriculum_vocabulary_text(step) if ct == "Từ vựng" else str(step.get("text") or "").strip()
    if text:
        blocks.append({"type":"text","text":(title+"\n\n"+text).strip()})
    for im in step.get("images") or []:
        if im.get("url"):
            blocks.append({"type":"image","key":im.get("key"),"url":im.get("url"),"page":im.get("page"),"caption":im.get("caption","")})

    sections=list((cache or {}).get("sections") or [])
    is_exercise = ct == "Bài tập"
    is_answer_step = str(step.get("code") or "").upper() in {"B2","ANSWER"}
    is_question_step = (ct == "Bài tập" and str(step.get("code") or "").upper() == "B0") or (ct == "Từ vựng" and str(step.get("code") or "").upper() == "B2")

    if answered and is_exercise:
        answer_step=_published_curriculum_answer_step(cache)
        if answer_step and answer_step.get("index") != step.get("index"):
            answer_text=str(answer_step.get("text") or "").strip()
            if answer_text:
                blocks.append({"type":"text","text":"📘 **Đáp án trong DB:**\n\n"+answer_text})
            for im in answer_step.get("images") or []:
                if im.get("url"):
                    blocks.append({"type":"image","key":im.get("key"),"url":im.get("url"),"page":im.get("page"),"caption":im.get("caption","")})
            step=answer_step
            is_answer_step=True

    if is_question_step and not answered:
        blocks.append({"type":"text","text":"✍️ Cậu hãy trả lời câu hỏi trên nhé. Nếu chưa biết, cứ nói **mình không biết**; Doraemon sẽ hiển thị đáp án đã lưu trong DB."})

    if step.get("is_final"):
        blocks.extend(_curriculum_final_blocks())
    else:
        blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
        blocks.extend(_curriculum_continue_blocks(int(step.get("index") or 0)))
    return blocks


def _published_curriculum_db_only_turn(query_text, action, study_session):
    a=str(action or "").strip().casefold()
    q=str(query_text or "").strip()
    if a.startswith("curriculum_next") or a=="lesson_confirm_yes": return True
    return bool(study_session and q and _is_continue_confirmation(q))


def _published_curriculum_blocks(step, cache):
    if not step: return []
    blocks=[]
    title=f"**{step['code']} · {step['title']}**" if step.get("title") else f"**{step['code']}**"
    text=(title+"\n\n"+step.get("text","")).strip()
    if text: blocks.append({"type":"text","text":text})
    for im in step.get("images") or []:
        if im.get("url"): blocks.append({"type":"image","key":im.get("key"),"url":im.get("url"),"page":im.get("page"),"caption":im.get("caption","")})
    if step.get("is_final"):
        blocks.extend(_curriculum_final_blocks())
    else:
        blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
        blocks.extend(_curriculum_continue_blocks(step["index"]))
    return blocks


def _load_runtime_lesson_cache(content_type, lesson, topic=None, *, request_id=None):
    """Load runtime lesson cache and merge compatible legacy rows deterministically."""
    if not lesson:
        return None
    ct = str(content_type or "").strip()
    ls = str(lesson or "").strip()
    tp = str(topic or "").strip()
    canonical_ls = _canonical_lesson_key(ls)
    canonical_tp = re.sub(r"\s+", " ", tp).strip().casefold()
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id,source_file,subject,content_type,lesson,status,version,raw_source_json
                FROM curriculum_lessons
                WHERE status='PUBLISHED'
                  AND lower(trim(content_type))=lower(trim(%s))
                  AND (lower(trim(lesson))=lower(trim(%s)) OR
                       regexp_replace(lower(trim(lesson)), '^bài\s+', '', 'g')=regexp_replace(lower(trim(%s)), '^bài\s+', '', 'g'))
                ORDER BY version DESC,id DESC
                LIMIT 1
            """,(ct,ls,ls))
            curriculum_row=cur.fetchone()
            if curriculum_row:
                cur.execute("SELECT id,step_code,step_order,title,step_type,content_json FROM curriculum_steps WHERE lesson_id=%s ORDER BY step_order,id",(curriculum_row['id'],))
                step_rows=cur.fetchall() or []
                if step_rows:
                    payload=_published_curriculum_runtime_payload(curriculum_row,step_rows)
                    print(f"[CURRICULUM DB RUNTIME HIT] request={request_id} lesson={ls!r} lesson_id={curriculum_row['id']} steps={len(step_rows)}")
                    return payload

            cur.execute(
                """SELECT id,source_file,subject,content_type,lesson,topic,status,updated_at,cache_json
                   FROM knowledge_lesson_cache
                   WHERE status='READY'
                     AND lower(trim(content_type))=lower(trim(%s))
                   ORDER BY updated_at DESC,id DESC""",
                (ct,),
            )
            all_rows = cur.fetchall() or []
            lesson_rows = [r for r in all_rows if _canonical_lesson_key(r.get('lesson')) == canonical_ls]
            if canonical_tp:
                compatible = [
                    r for r in lesson_rows
                    if re.sub(r"\s+", " ", str(r.get('topic') or '').strip()).casefold() == canonical_tp
                ]
            else:
                compatible = lesson_rows

            print(
                f"[KNOWLEDGE CACHE LOOKUP] request={request_id} "
                f"requested_content_type={ct!r} requested_lesson={ls!r} requested_topic={tp!r} "
                f"canonical_lesson={canonical_ls!r} canonical_topic={canonical_tp!r} "
                f"lesson_rows={len(lesson_rows)} compatible_rows={len(compatible)} "
                f"match={'1' if compatible else '0'}"
            )
            if not compatible:
                return None

            payloads = []
            base = dict(compatible[0])
            for row in compatible:
                payload = row.get('cache_json')
                if isinstance(payload, dict) and payload.get('sections'):
                    payloads.append(payload)
            if not payloads:
                return None

            merged_sections = []
            merged_images = []
            seen_section_keys = set()
            seen_image_keys = set()
            for payload in payloads:
                for sec in payload.get('sections') or []:
                    sec = dict(sec)
                    key = (
                        str(sec.get('content_unit_id') or ''),
                        int(sec.get('page') or 0),
                        str(sec.get('text') or ''),
                    )
                    if key in seen_section_keys:
                        continue
                    seen_section_keys.add(key)
                    merged_sections.append(sec)
                for img in payload.get('images') or []:
                    img = dict(img)
                    key = str(img.get('image_key') or '').strip()
                    if key and key not in seen_image_keys:
                        seen_image_keys.add(key)
                        merged_images.append(img)

            merged_sections.sort(key=lambda x: (
                int(x.get('page') or 0),
                int(x.get('chunk_index') or 0),
                str(x.get('content_unit_id') or ''),
                str(x.get('text') or ''),
            ))
            for idx, sec in enumerate(merged_sections):
                sec['chunk_index'] = idx
            merged_images.sort(key=lambda x: (
                int(x.get('page') or 0),
                int(x.get('chunk_index') or 0),
                str(x.get('image_key') or ''),
            ))

            merged = {
                'version': 1,
                'source_file': base.get('source_file'),
                'content_hash': (payloads[0] or {}).get('content_hash'),
                'subject': base.get('subject') or (payloads[0] or {}).get('subject'),
                'content_type': base.get('content_type') or (payloads[0] or {}).get('content_type'),
                'lesson': base.get('lesson') or canonical_ls,
                'topic': base.get('topic') or tp or None,
                'overview': ' '.join(str(s.get('text') or '') for s in merged_sections[:2]).strip()[:2400],
                'sections': merged_sections,
                'images': merged_images,
            }
            print(
                f"[KNOWLEDGE CACHE PAYLOAD] request={request_id} cache_id={base.get('id')} "
                f"payload_ready=1 sections={len(merged_sections)} images={len(merged_images)} "
                f"stored_lesson={base.get('lesson')!r} stored_topic={base.get('topic')!r}"
            )
            return merged
    except Exception as exc:
        print("[KNOWLEDGE CACHE] load failed:", type(exc).__name__, str(exc))
        return None
    finally:
        conn.close()

def _select_runtime_cache_sections(cache, query_text, *, max_sections=2, initial=False):
    sections = list((cache or {}).get("sections") or [])
    if not sections:
        return []
    if initial:
        return sections[:max_sections]
    q = _clean_scope_value(query_text)
    tokens = [t for t in re.findall(r"[\wÀ-ỹ一-龥ぁ-んァ-ンー]+", q) if len(t) >= 2]
    scored=[]
    for idx, sec in enumerate(sections):
        text=_clean_scope_value(sec.get("text"))
        score=0
        for tok in tokens:
            if tok in text:
                score += 3 if len(tok) >= 4 else 1
        # Keep adjacent structure stable so a selected section brings its immediate context.
        if score:
            score += max(0, 2 - idx*0.05)
        scored.append((score, idx, sec))
    scored.sort(key=lambda x:(x[0], -x[1]), reverse=True)
    chosen=[x[2] for x in scored[:max_sections] if x[0] > 0]
    if not chosen:
        chosen=sections[:max_sections]
    chosen.sort(key=lambda x:(int(x.get("page") or 0), int(x.get("chunk_index") or 0)))
    return chosen


def _curriculum_chunk_images(cache, selected_section):
    """Return ONLY images whose provenance belongs to the exact curriculum chunk.

    Curriculum rule: one teaching step = one Knowledge Cache chunk. Do not leak
    lesson-wide images from the page or sibling chunks into that step.
    """
    if not isinstance(selected_section, dict):
        return []
    images = list((cache or {}).get("images") or [])
    target_chunk = selected_section.get("chunk_index")
    target_unit = str(selected_section.get("content_unit_id") or "").strip()
    target_keys = {str(k).strip() for k in (selected_section.get("image_keys") or []) if str(k).strip()}
    out=[]
    seen=set()
    for item in images:
        key=str(item.get("image_key") or "").strip()
        if not key or key in seen:
            continue
        item_chunk=item.get("chunk_index")
        item_unit=str(item.get("content_unit_id") or "").strip()
        exact_chunk = (target_chunk not in (None, "") and item_chunk not in (None, "") and str(item_chunk)==str(target_chunk))
        exact_unit = bool(target_unit and item_unit and item_unit==target_unit)
        key_linked = key in target_keys
        # Curriculum section.image_keys is authoritative provenance created at
        # upload/chunking time. If a chunk explicitly carries an image key, that
        # image belongs to the chunk even when the legacy knowledge_vision_cache
        # row has no chunk_index/content_unit_id (older caches often look like this).
        # Exact chunk/unit provenance remains preferred, but explicit key linkage
        # must be honored so cached lesson images are not silently dropped.
        if not (exact_chunk or exact_unit or key_linked):
            continue
        vision=item.get("vision") or {}
        out.append({
            "key": key, "url": item.get("image_url") or b2_url(key),
            "term": vision.get("term", ""), "reading": vision.get("reading", ""),
            "meaning": vision.get("meaning", ""), "page": item.get("page"),
            "_chunk_order": 0,
            "content_type": (cache or {}).get("content_type"),
            "lesson": (cache or {}).get("lesson"), "topic": (cache or {}).get("topic"),
            "_exact_chunk": True,
            "_key_linked": key_linked,
        })
        seen.add(key)
    return out


def _runtime_cache_images(cache, selected_sections):
    images_by_key={str(x.get("image_key")):x for x in (cache or {}).get("images",[]) if x.get("image_key")}
    out=[]
    seen=set()
    for order, sec in enumerate(selected_sections):
        for key in sec.get("image_keys") or []:
            key=str(key).strip()
            if not key or key in seen or key not in images_by_key:
                continue
            item=images_by_key[key]
            vision=item.get("vision") or {}
            out.append({
                "key": key, "url": item.get("image_url") or b2_url(key),
                "term": vision.get("term", ""), "reading": vision.get("reading", ""),
                "meaning": vision.get("meaning", ""), "page": item.get("page"),
                "_chunk_order": order,
                "content_type": (cache or {}).get("content_type"),
                "lesson": (cache or {}).get("lesson"), "topic": (cache or {}).get("topic"),
            })
            seen.add(key)
    return out


_GREETING_EXACT = {
    "chào", "chào bạn", "chào cậu", "chào doraemon",
    "xin chào", "xin chào bạn", "xin chào doraemon",
    "hello", "hello doraemon", "hi", "hi doraemon",
    "hey", "hey doraemon", "alo", "alo doraemon",
    "doraemon ơi", "doraemon ơi chào",
}


def _get_study_session(user_id, chatbox_id=None):
    """Return the persisted ACTIVE study session, scoped to the current chatbox."""
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT study_session_active,study_session_content_type,study_session_course,
                       study_session_lesson,study_session_topic,study_session_started_at,
                       study_end_prompt_pending,study_session_chatbox_id,
                       curriculum_step,curriculum_waiting,curriculum_exercise_answered,curriculum_intro_history,curriculum_intro_b0b1_history,curriculum_global_exercise_result
                FROM user_learning_state WHERE user_id=%s
            """, (user_id,))
            row = cur.fetchone()
            if not row or not row.get("study_session_active"):
                return None
            stored_chatbox = str(row.get("study_session_chatbox_id") or "").strip() or None
            requested_chatbox = str(chatbox_id or "").strip() or None
            if requested_chatbox and stored_chatbox and stored_chatbox != requested_chatbox:
                return None
            return {
                "active": True,
                "content_type": _normalize_content_type(row.get("study_session_content_type")) or None,
                "course": str(row.get("study_session_course") or "").strip() or None,
                "lesson": str(row.get("study_session_lesson") or "").strip() or None,
                "topic": str(row.get("study_session_topic") or "").strip() or None,
                "started_at": row.get("study_session_started_at"),
                "end_prompt_pending": bool(row.get("study_end_prompt_pending")),
                "chatbox_id": stored_chatbox,
                "curriculum_step": int(row.get("curriculum_step") or 0),
                "curriculum_waiting": str(row.get("curriculum_waiting") or "continue"),
                "curriculum_exercise_answered": bool(row.get("curriculum_exercise_answered")),
                "curriculum_global_exercise_question": str(row.get("curriculum_global_exercise_question") or ""),
                "curriculum_global_exercise_evidence": str(row.get("curriculum_global_exercise_evidence") or ""),
                "curriculum_summary_notes": str(row.get("curriculum_summary_notes") or ""),
                "curriculum_intro_history": str(row.get("curriculum_intro_history") or ""),
                "curriculum_intro_b0b1_history": str(row.get("curriculum_intro_b0b1_history") or ""),
                "curriculum_global_exercise_result": str(row.get("curriculum_global_exercise_result") or ""),
            }
    finally:
        conn.close()


def _start_study_session(user_id, scope, chatbox_id=None):
    """Persist an explicitly confirmed lesson as the only active scope for this chatbox."""
    scope = scope or {}
    lesson = str(scope.get("lesson") or "").strip()
    if not lesson:
        raise ValueError("Study session requires an exact lesson.")
    content_type = _normalize_content_type(scope.get("content_type")) or None
    course = str(scope.get("course") or scope.get("course_name") or "").strip() or None
    topic = str(scope.get("topic") or "").strip() or None
    chatbox = str(chatbox_id or "").strip() or None
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_learning_state(
                    user_id,welcome_seen,reset_count,learning_mode,onboarding_completed,
                    study_session_active,study_session_content_type,study_session_course,
                    study_session_lesson,study_session_topic,study_session_chatbox_id,study_session_started_at,
                    study_end_prompt_pending,curriculum_step,curriculum_waiting,curriculum_exercise_answered,curriculum_global_exercise_question,curriculum_global_exercise_evidence,curriculum_summary_notes,curriculum_intro_history,curriculum_intro_b0b1_history,curriculum_global_exercise_result,updated_at
                ) VALUES(%s,TRUE,0,NULL,TRUE,TRUE,%s,%s,%s,%s,%s,NOW(),FALSE,0,'continue',FALSE,'','','','','','',NOW())
                ON CONFLICT(user_id) DO UPDATE SET
                    study_session_active=TRUE,
                    study_session_content_type=%s,
                    study_session_course=%s,
                    study_session_lesson=%s,
                    study_session_topic=%s,
                    study_session_chatbox_id=%s,
                    study_session_started_at=NOW(),
                    study_end_prompt_pending=FALSE,
                    curriculum_step=0,
                    curriculum_waiting='continue',
                    curriculum_exercise_answered=FALSE,
                    curriculum_global_exercise_question='',
                    curriculum_global_exercise_evidence='',
                    curriculum_summary_notes='',
                    curriculum_intro_history='',
                    curriculum_intro_b0b1_history='',
                    curriculum_global_exercise_result='',
                    updated_at=NOW()
            """, (
                user_id,content_type,course,lesson,topic,chatbox,
                content_type,course,lesson,topic,chatbox
            ))
        conn.commit()
    finally:
        conn.close()


def _finish_study_session(user_id):
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE user_learning_state
                           SET study_session_active=FALSE, study_end_prompt_pending=FALSE,
                               study_session_chatbox_id=NULL, updated_at=NOW()
                           WHERE user_id=%s""", (user_id,))
        conn.commit()
    finally:
        conn.close()


def _set_study_end_prompt_pending(user_id, pending=True):
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE user_learning_state
                SET study_end_prompt_pending=%s, updated_at=NOW()
                WHERE user_id=%s
            """, (bool(pending), user_id))
        conn.commit()
    finally:
        conn.close()


def _finish_study_session(user_id):
    """Close the active study session; future turns are non-RAG until a new YES."""
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE user_learning_state
                SET study_session_active=FALSE,
                    study_session_content_type=NULL,
                    study_session_course=NULL,
                    study_session_lesson=NULL,
                    study_session_topic=NULL,
                    study_session_started_at=NULL,
                    study_end_prompt_pending=FALSE,
                    curriculum_step=0,
                    curriculum_waiting='continue',
                    curriculum_exercise_answered=FALSE,
                    curriculum_global_exercise_question='',
                    curriculum_global_exercise_evidence='',
                    curriculum_summary_notes='',
                    curriculum_intro_history='',
                    curriculum_intro_b0b1_history='',
                    curriculum_global_exercise_result='',
                    updated_at=NOW()
                WHERE user_id=%s
            """, (user_id,))
        conn.commit()
    finally:
        conn.close()


def _active_session_scope(session):
    if not session or not session.get("active") or not session.get("lesson"):
        return None
    return {
        "course": session.get("course"),
        "content_type": session.get("content_type"),
        "lesson": session.get("lesson"),
        "topic": session.get("topic"),
    }


def _set_curriculum_compact_state(user_id, *, global_question=None, global_evidence=None, summary_notes=None):
    conn = db()
    cur = conn.cursor()
    try:
        sets = []
        vals = []
        if global_question is not None:
            sets.append("curriculum_global_exercise_question=%s")
            vals.append(str(global_question)[:1200])
        if global_evidence is not None:
            sets.append("curriculum_global_exercise_evidence=%s")
            vals.append(str(global_evidence)[:2400])
        if summary_notes is not None:
            sets.append("curriculum_summary_notes=%s")
            vals.append(str(summary_notes)[:6000])
        if sets:
            sets.append("updated_at=NOW()")
            cur.execute("UPDATE user_learning_state SET " + ", ".join(sets) + " WHERE user_id=%s", tuple(vals + [user_id]))
            conn.commit()
    finally:
        conn.close()

def _append_curriculum_intro_history(user_id, text):
    """Store only Doraemon's opening/section teaching replies for later global exercise/summary context."""
    text = re.sub(r"\[\[(?:CHUNK_EXERCISE|NO_CHUNK_EXERCISE|WHOLE_LESSON_EXERCISE)(?:_[^\]]+)?\]\]", "", str(text or "")).strip()
    if not text:
        return
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT curriculum_intro_history FROM user_learning_state WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            prev = str((row[0] if row else "") or "").strip()
            combined = (prev + "\n\n" + text).strip()
            # Keep only the teaching history, not exercise answers or full lesson source.
            combined = combined[-12000:]
            cur.execute("UPDATE user_learning_state SET curriculum_intro_history=%s, updated_at=NOW() WHERE user_id=%s", (combined, user_id))
            conn.commit()
    finally:
        conn.close()

def _append_curriculum_intro_b0b1_history(user_id, step, text):
    """Store only B0 + B1 teaching replies for global exercise detection and final summary."""
    if int(step) not in (0, 1):
        return
    text = re.sub(r"\[\[(?:CHUNK_EXERCISE|NO_CHUNK_EXERCISE|WHOLE_LESSON_EXERCISE)(?:_[^\]]+)?\]\]", "", str(text or "")).strip()
    if not text:
        return
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT curriculum_intro_b0b1_history FROM user_learning_state WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            prev = str((row[0] if row else "") or "").strip()
            combined = (prev + "\n\n" + text).strip()[-5000:]
            cur.execute("UPDATE user_learning_state SET curriculum_intro_b0b1_history=%s, updated_at=NOW() WHERE user_id=%s", (combined, user_id))
            conn.commit()
    finally:
        conn.close()

def _set_curriculum_global_exercise_result(user_id, text):
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE user_learning_state SET curriculum_global_exercise_result=%s, updated_at=NOW() WHERE user_id=%s", (str(text or "")[:4000], user_id))
            conn.commit()
    finally:
        conn.close()

def _set_curriculum_flow(user_id, *, step=None, waiting=None, exercise_answered=None):
    """Persist the lightweight Giáo trình step state for the current study session."""
    sets=[]; vals=[]
    if step is not None:
        sets.append("curriculum_step=%s"); vals.append(int(step))
    if waiting is not None:
        sets.append("curriculum_waiting=%s"); vals.append(str(waiting))
    if exercise_answered is not None:
        sets.append("curriculum_exercise_answered=%s"); vals.append(bool(exercise_answered))
    if not sets:
        return
    sets.append("updated_at=NOW()")
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE user_learning_state SET {', '.join(sets)} WHERE user_id=%s", tuple(vals+[user_id]))
        conn.commit()
    finally:
        conn.close()


def _curriculum_sections(cache):
    """Return deterministic teaching sections with fresh contiguous indexes.

    Some older uploads retained page-local/duplicate chunk indexes. Curriculum
    order must follow the stored section list, not trust those legacy indexes.
    """
    raw = list((cache or {}).get("sections") or [])
    indexed = list(enumerate(raw))
    indexed.sort(key=lambda pair: (
        int(pair[1].get("page") or 0),
        int(pair[1].get("chunk_index") or 0),
        str(pair[1].get("content_unit_id") or ""),
        pair[0],
    ))
    out = []
    for new_idx, (_, section) in enumerate(indexed):
        sec = dict(section)
        sec["chunk_index"] = new_idx
        out.append(sec)
    return out


def _curriculum_step_map(cache):
    """Fixed Giáo trình flow with a separate whole-lesson exercise checkpoint.

    Flow:
      0 = introduction only
      1..N = exactly one Knowledge Cache chunk per teaching step
      N+1 = whole-lesson exercise checkpoint (only if a real source exercise exists)
      N+2 = final summary

    Runtime LLM decides whether an exercise belongs to the current chunk. A question
    that needs multiple chunks is deferred to the whole-lesson checkpoint.
    """
    sections = _curriculum_sections(cache)
    global_exercise_step = 1 + len(sections)
    summary_step = 2 + len(sections)
    return {
        "sections": sections,
        "exercise": None,
        "embedded_exercise_steps": set(),
        "post_chunk_exercise": True,
        "global_exercise_step": global_exercise_step,
        "exercise_step": global_exercise_step,
        "summary_step": summary_step,
    }


def _is_continue_confirmation(text):
    q=str(text or "").strip().casefold()
    return q in {"có","co","ok","okay","oke","ừ","ừm","vâng","được","được rồi","tiếp","tiếp đi","tiếp nhé","tiếp tục","rồi","hiểu rồi","sang phần tiếp theo","tổng kết đi","mình sẵn sàng"}


def _curriculum_continue_blocks(step, label="Tiếp tục"):
    return [{"type":"choice","id":"curriculum_next","options":[{"label":label,"action":f"curriculum_next:{int(step)}"}]}]


def _curriculum_final_blocks():
    return [{"type":"choice","id":"curriculum_final","options":[
        {"label":"Có — mình nắm rồi","action":"curriculum_finish_yes"},
        {"label":"Chưa — mình muốn ôn lại","action":"curriculum_finish_no"},
    ]}]


def _is_study_followup(text):
    """Cheap deterministic classifier used ONLY while a study session is active.

    It is intentionally much broader than the old casual-message detector: any
    likely lesson continuation stays in RAG, while clear social/off-topic text
    exits to the end-of-lesson confirmation without invoking Gemini embedding or
    Pinecone.
    """
    q = str(text or "").strip().casefold()
    if not q:
        return False
    study_markers = (
        "học", "bài", "lesson", "từ vựng", "từ mới", "ngữ pháp", "kanji",
        "bộ thủ", "bài tập", "quiz", "truyện", "giáo trình", "giải thích",
        "giải", "đáp án", "câu", "ví dụ", "nghĩa", "đọc", "viết", "phát âm",
        "tại sao", "vì sao", "như thế nào", "thế nào", "tiếp theo", "tiếp",
        "làm lại", "nói lại", "chậm hơn", "không hiểu", "chưa hiểu", "khó",
        "sai", "đúng không", "ôn", "luyện", "review", "nhắc lại", "phần này",
        "chỗ này", "cái này", "đoạn này", "câu này", "từ này", "mẫu câu",
    )
    if any(m in q for m in study_markers):
        return True
    if q in {"ok", "ừ", "ừm", "vâng", "được", "được rồi", "tiếp đi", "tiếp nhé", "rồi", "hiểu rồi"}:
        return True
    return False



def _exercise_simple_direct_answer(query_text, step, cache=None, current_step=None):
    """Deterministic no-LLM handling for casual/simple exercise-session turns."""
    q=str(query_text or '').strip().casefold()
    if not q or not isinstance(step, dict):
        return None

    # Off-topic/casual chat while the learner is inside an exercise session.
    casual_markers=(
        'hôm nay nóng', 'hôm nay lạnh', 'thời tiết', 'trời nóng', 'trời lạnh',
        'haha', 'hihi', 'hehe', 'ẹc', 'ặc', 'ôi', 'mệt quá', 'chán quá',
        'khó thế', 'khó thật', 'khó quá', 'khó ghê', 'nản quá', 'bó tay',
        'mình chịu', 'chịu rồi', 'không làm được', 'khó quá mình chịu',
    )
    if any(x in q for x in casual_markers):
        # On an answer/review step, a learner saying they cannot do it should
        # reveal the official DB answer instead of spending a GenAI turn.
        code=str(step.get('code') or '').upper()
        if code in {'B0','B1'} and any(x in q for x in ('khó','chịu','không làm được','bó tay')):
            ans=_published_curriculum_answer_step(cache or {}) if cache else None
            at=str((ans or {}).get('text') or '').strip()
            if at:
                return ('answer_db', 'Được nhé 😄 Không sao. Đây là **đáp án chính thức trong DB**:\n\n'+at)
        return ('casual', '😄 Ừ, bài này hơi khó thật. Cậu cứ bình tĩnh nhé. Muốn tiếp tục bài tập thì bấm **Tiếp theo**.')

    # Common progress/status questions are deterministic.
    if any(x in q for x in ('hết chưa', 'xong chưa', 'còn câu nào', 'còn bài nào')):
        sections=list((cache or {}).get('sections') or [])
        if sections and current_step is not None:
            remaining=max(0, len(sections)-int(current_step)-1)
            if remaining:
                return ('progress', f'📌 Chưa hết nhé, còn khoảng **{remaining} bước** trong bài tập này. Cậu bấm **Tiếp theo** để sang phần tiếp.')
            return ('progress', '✅ Gần hết rồi. Cậu đang ở phần cuối của bài tập này.')
        return ('progress', '📌 Mình đang ở trong bài tập hiện tại. Cậu bấm **Tiếp theo** để xem phần tiếp theo nhé.')

    # Asking what exercise material is available can be answered from the lesson catalog,
    # without asking Gemini to invent a list.
    if any(x in q for x in ('cậu có bài nào', 'có bài nào', 'có bài tập nào', 'bài tập nào có')):
        try:
            lessons=_unique_lessons_for_scope('Bài tập', '')
        except Exception:
            lessons=[]
        if lessons:
            shown=lessons[:10]
            suffix='...' if len(lessons)>10 else ''
            return ('catalog', '📚 Hiện có các bài tập: **'+', '.join(shown)+suffix+'**')
        return ('catalog', '📚 Mình chưa tìm thấy danh sách bài tập khác trong DB.')

    return None

def _is_off_topic_during_study(text):
    q = str(text or "").strip().casefold()
    if not q:
        return False
    emotional = (
        "mệt", "chán", "buồn", "nản", "stress", "bực", "khó chịu", "kiệt sức",
        "quá mệt mỏi", "mệt mỏi với", "cảm ơn", "thanks", "thank you", "haha",
        "hihi", "hehe", "huhu", "hic", "nghỉ", "dừng lại", "thôi",
        "hôm nay nóng", "hôm nay lạnh", "thời tiết", "trời nóng", "trời lạnh",
        "ẹc", "ặc", "ôi",
    )
    if any(x in q for x in emotional):
        return True
    return _is_general_non_learning_request(text)


def _study_end_choice_blocks(scope, prefix=""):
    lesson = str((scope or {}).get("lesson") or "bài này").strip()
    text = prefix + (f"\n\nMình kết thúc bài học **{lesson}** hôm nay nhé? 😊" if prefix else f"Mình kết thúc bài học **{lesson}** hôm nay nhé? 😊")
    return [
        {"type":"text","text":text},
        {"type":"choice","id":"study_end",
         "options":[
             {"label":"Có","display_label":f"Có — kết thúc {lesson}","action":"study_end_yes"},
             {"label":"Không","display_label":f"Không — học tiếp {lesson}","action":"study_end_no"}
         ]}
    ]


def _get_learning_profile(user_id):
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT learning_mode,onboarding_completed FROM user_learning_state WHERE user_id=%s""", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else {"learning_mode": None, "onboarding_completed": False}
    finally:
        conn.close()


def _set_learning_profile(user_id, mode, completed=True):
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO user_learning_state(user_id,welcome_seen,reset_count,learning_mode,onboarding_completed,updated_at)\n                VALUES(%s,TRUE,0,%s,%s,NOW())\n                ON CONFLICT(user_id) DO UPDATE SET learning_mode=%s,onboarding_completed=%s,updated_at=NOW()""",
                (user_id, mode, completed, mode, completed))
        conn.commit()
    finally:
        conn.close()


def _plan_content_type_from_text(text):
    q = str(text or '').casefold()
    # Prefer the explicit study subtype over the generic word "giáo trình".
    # This prevents requests such as "tạo lộ trình học ngữ pháp" from being
    # interpreted as a curriculum plan simply because a curriculum source is
    # mentioned elsewhere in the sentence/context.
    patterns = [
        ("Ngữ pháp", ("ngữ pháp", "ngu phap", "grammar")),
        ("Bài tập", ("bài tập", "bai tap", "luyện tập", "luyen tap", "exercise", "quiz")),
        ("Từ vựng", ("từ vựng", "tu vung", "từ mới", "tu moi", "vocabulary", "bộ thủ", "bo thu", "kanji")),
        ("Truyện đọc", ("truyện đọc", "truyen doc", "đọc truyện", "doc truyen", "reading", "story")),
        ("Giáo trình", ("giáo trình", "giao trinh", "học theo giáo trình", "theo giáo trình")),
    ]
    for content_type, keys in patterns:
        if any(k in q for k in keys):
            return content_type
    return None


def _unique_lessons_for_scope(content_type='Giáo trình', scope_query=''):
    """Return lessons strictly from the requested plan content type.

    Plan generation must never fall back from a requested type (e.g. Ngữ pháp)
    to generic curriculum lessons. The catalog cache is used only as a source,
    but matching is strict and case-insensitive.
    """
    wanted = str(content_type or 'Giáo trình').strip()
    catalog = _load_catalog_cached()
    rows=[]; seen=set(); q=_clean_scope_value(scope_query)
    for item in catalog:
        raw_type = str(item.get('content_type') or '').strip()
        normalized = _normalize_content_type(raw_type)
        if normalized.casefold() != wanted.casefold():
            continue
        lesson=str(item.get('lesson') or '').strip()
        if not lesson:
            continue
        if q and q not in _clean_scope_value(str(item.get('subject') or '')) and q not in _clean_scope_value(lesson):
            continue
        key=lesson.casefold()
        if key in seen:
            continue
        seen.add(key); rows.append(lesson)
    print(f"[STUDY PLAN LESSONS] content_type={wanted!r} scope={scope_query!r} lessons={len(rows)}")
    return rows


def _parse_plan_request(text):
    q=str(text or '').strip()
    ql=q.casefold()
    target_date=None
    m=re.search(r'(?:đến|tới|trước|trong)\s*(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?', ql)
    if m:
        day=int(m.group(1)); month=int(m.group(2)); year=int(m.group(3) or _now_local().year)
        try: target_date=date(year,month,day)
        except: target_date=None
    months=None; days=None
    m=re.search(r'(\d+)\s*tháng', ql)
    if m: months=int(m.group(1))
    m=re.search(r'(?:trong|for)\s*(\d+)\s*(?:ngày|day)', ql)
    if m: days=int(m.group(1))
    if days is None:
        m=re.search(r'(\d+)\s*(?:ngày|day)', ql)
        if m and 'mỗi ngày' not in ql: days=int(m.group(1))
    units_per_day=None; days_per_unit=None; unit_label=None
    m=re.search(r'(?:mỗi ngày|mỗi hôm)\s*(?:học\s*)?(\d+(?:\.\d+)?)\s*(bài|lesson|từ|từ vựng|mục|câu)', ql)
    if m:
        units_per_day=float(m.group(1)); unit_label=m.group(2)
    m=re.search(r'(\d+)\s*ngày\s*(?:một|1)\s*bài', ql)
    if m: days_per_unit=int(m.group(1))
    if target_date is None:
        if months: target_date=_now_local().date()+timedelta(days=round(months*30)-1)
        elif days: target_date=_now_local().date()+timedelta(days=max(0, days-1))
    scope='N5' if 'n5' in ql else ''
    if 'bộ thủ' in ql or 'bo thu' in ql:
        scope='Bộ thủ'
    elif 'kanji' in ql:
        scope='Kanji'
    detected_type = _plan_content_type_from_text(ql)
    # Do not silently default a plan request to Giáo trình when the current
    # message contains only a duration/speed. The caller may already have a
    # pending plan subtype (e.g. Ngữ pháp) or can infer it from the preceding
    # user message. A silent default here was the root cause of requests such
    # as "tạo lộ trình học ngữ pháp" -> "mình muốn học trong 5 ngày"
    # falling back to Giáo trình.
    content_type = detected_type
    goal=q.strip()
    return {"goal":goal,"target_date":target_date,"units_per_day":units_per_day,"days_per_unit":days_per_unit,"scope":scope,"content_type":content_type,"unit_label":unit_label}



def _is_lightweight_casual_message(message: str) -> bool:
    """Hard-stop obvious casual/emotional chatter before lesson RAG."""
    s = (message or "").strip().casefold()
    if not s:
        return True
    phrases = (
        "mệt quá", "mệt lắm", "mệt rồi", "mệt quá hic", "mệt lắm rồi",
        "mệt lắm rồi đấy", "mệt lắm rồi nhé", "mệt lắm rồi đấy nhé",
        "buồn quá", "chán quá", "stress quá", "huhu", "hic", "haha",
        "cảm ơn", "thanks", "thank you", "nghỉ tí", "nghỉ một chút",
        "mình mệt", "tôi mệt",
    )
    return any(x in s for x in phrases)


def _normalize_chat_history(chat_history, max_messages=20):
    """Normalize the current chatbox history into at most max_messages messages.

    The client supplies the history of the CURRENT open chatbox. This is the
    authoritative conversational context for routing multi-turn flows such as
    Study Plan creation. max_messages=20 means up to 10 user/model exchanges.
    """
    recent = []
    for item in (chat_history or [])[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        parts = item.get("parts")
        msg_text = ""
        if isinstance(parts, list):
            pieces = []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    pieces.append(str(part.get("text")))
            msg_text = " ".join(pieces).strip()
        elif item.get("text"):
            msg_text = str(item.get("text")).strip()
        elif item.get("content"):
            msg_text = str(item.get("content")).strip()
        if role in {"user", "model", "assistant"} and msg_text:
            recent.append({
                "role": "model" if role == "assistant" else role,
                "text": msg_text[-1200:],
            })
    return recent


def _last_chat_exchange(recent_history):
    """Return only the latest user+assistant/model exchange for lightweight GenAI turns.

    This is deliberately separate from the larger routing/history window used by
    Study Plan and lesson routing. Vocabulary/Exercise GenAI teacher turns only
    need one preceding exchange, plus the current user message.
    """
    hist=list(recent_history or [])
    if not hist:
        return []
    latest_model_idx=None
    for i in range(len(hist)-1, -1, -1):
        if str(hist[i].get("role") or "").strip().lower() == "model":
            latest_model_idx=i
            break
    if latest_model_idx is not None:
        user_idx=None
        for i in range(latest_model_idx-1, -1, -1):
            if str(hist[i].get("role") or "").strip().lower() == "user":
                user_idx=i
                break
        if user_idx is not None:
            return [hist[user_idx], hist[latest_model_idx]]
    # Fallback: one latest message only. Never return a larger history window.
    return [hist[-1]]


def _infer_plan_content_type_from_history(recent_history):
    """Infer a pending plan subtype from the most recent explicit user request.

    This is only a fallback for a follow-up target message such as
    "mình muốn học trong 5 ngày". It never uses assistant text, catalog ranking,
    RAG, or active learning state to invent a plan subtype.
    """
    for item in reversed(recent_history or []):
        if str(item.get('role') or '').strip().lower() != 'user':
            continue
        txt = str(item.get('text') or '').strip()
        if not txt:
            continue
        detected = _plan_content_type_from_text(txt)
        if detected:
            return detected
    return None


def _set_pending_plan_request(user_id, content_type=None, scope=None):
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE user_learning_state SET pending_plan_content_type=%s,pending_plan_scope=%s,pending_plan_created_at=NOW(),updated_at=NOW() WHERE user_id=%s""",(content_type,scope,user_id))
        conn.commit()
    finally:
        conn.close()

def _get_pending_plan_request(user_id):
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT pending_plan_content_type,pending_plan_scope,pending_plan_created_at FROM user_learning_state WHERE user_id=%s",(user_id,))
            r=cur.fetchone()
            return dict(r) if r else None
    finally:
        conn.close()

def _clear_pending_plan_request(user_id):
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE user_learning_state SET pending_plan_content_type=NULL,pending_plan_scope=NULL,pending_plan_created_at=NULL,updated_at=NOW() WHERE user_id=%s""",(user_id,))
        conn.commit()
    finally:
        conn.close()

def _build_plan_preview(user_id, req, existing_plan_id=None):
    requested_type = req.get('content_type') or 'Giáo trình'
    requested_type = _normalize_content_type(requested_type) if requested_type in CONTENT_TYPES else requested_type
    req['content_type'] = requested_type
    print(f"[STUDY PLAN LESSONS] content_type={requested_type!r} scope={req.get('scope') or ''!r} source={req.get('goal') or ''!r}")
    lessons=list(req.get('_lessons_override') or []) or _unique_lessons_for_scope(requested_type, req.get('scope') or '')
    if not lessons:
        # fall back to all curriculum lessons from the selected type
        lessons=_unique_lessons_for_scope(requested_type, '')
    if not lessons:
        return None, f"Doraemon chưa tìm thấy danh sách bài phù hợp cho loại nội dung **{requested_type}** để lập lộ trình."
    start=_now_local().date()
    target=req.get('target_date')
    if not target:
        units_per_day=req.get('units_per_day') or 1.0
        days_per_unit=req.get('days_per_unit') or 0
        if days_per_unit:
            target=start+timedelta(days=max(1, int(days_per_unit*len(lessons))-1))
        else:
            target=start+timedelta(days=max(0, int((len(lessons)/units_per_day))-1))
    total_days=max(1,(target-start).days+1)
    units_per_day=req.get('units_per_day')
    days_per_unit=req.get('days_per_unit')
    if not units_per_day and not days_per_unit:
        units_per_day=round(len(lessons)/total_days,3)
        if units_per_day < 1: days_per_unit=max(1, round(total_days/len(lessons)))
    # Generate plan items. For vocabulary/"Bộ thủ" style goals expressed as a daily count,
    # create one daily target item rather than treating the whole lesson as one unit.
    items=[]
    if req.get('units_per_day') and req.get('unit_label') and req.get('content_type') == 'Từ vựng' and req.get('target_date'):
        day=start
        idx=1
        label=req.get('scope') or 'Từ vựng'
        while day <= target:
            items.append((day, idx, label, f"{req.get('units_per_day'):g} {req.get('unit_label')}/ngày"))
            idx += 1
            day += timedelta(days=1)
    else:
        items=[]
    if not items:
        if days_per_unit:
            for i,lesson in enumerate(lessons,1):
                d=start+timedelta(days=(i-1)*int(days_per_unit))
                if d>target: break
                items.append((d,i,lesson,''))
        else:
            upd=max(float(units_per_day or 1),0.01)
            day=start; idx=0
            while idx<len(lessons) and day<=target:
                quota=max(1,int(round(upd))) if upd>=1 else 1
                for _ in range(quota):
                    if idx>=len(lessons): break
                    items.append((day,idx+1,lessons[idx],'')); idx+=1
                day+=timedelta(days=1)
        if len(items)<len(lessons) and not days_per_unit:
            d=target
            for lesson in lessons[len(items):]:
                items.append((d,len(items)+1,lesson,''))
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(version),0)+1 FROM study_plans WHERE user_id=%s",(user_id,))
            version=int(cur.fetchone()[0])
            cur.execute("""INSERT INTO study_plans(user_id,version,status,goal_name,content_type,scope,start_date,target_date,units_per_day,days_per_unit)\n                VALUES(%s,%s,'DRAFT',%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (user_id,version,req.get('goal') or 'Lộ trình học',req.get('content_type') or 'Giáo trình',req.get('scope') or '',start,target,units_per_day,days_per_unit))
            plan_id=int(cur.fetchone()[0])
            for d,i,lesson,target_text in items:
                cur.execute("INSERT INTO study_plan_items(study_plan_id,plan_date,unit_index,lesson,target,status) VALUES(%s,%s,%s,%s,%s,'pending')",
                            (plan_id,d,i,lesson,target_text or 'Hoàn thành bài học và xác nhận với Doraemon'))
        conn.commit()
    finally: conn.close()
    preview="🤖 Doraemon đã tính thử lộ trình mới:\n\n"
    preview += f"🎯 {req.get('goal') or 'Lộ trình học'}\n📅 Từ {start.strftime('%d/%m/%Y')} đến {target.strftime('%d/%m/%Y')}\n📚 {len(items)} bài\n"
    preview += "\n".join(f"• {d.strftime('%d/%m')}: {lesson}{(" – "+target_text) if target_text else ""}" for d,i,lesson,target_text in items[:12])
    if len(items)>12: preview += f"\n• ... và {len(items)-12} bài tiếp theo"
    preview += "\n\nCậu có muốn áp dụng lộ trình này không? Hãy trả lời 'Có' để xác nhận hoặc 'Không' để giữ nguyên lộ trình hiện tại."
    return plan_id, preview


def _finalize_completed_active_plans(user_id):
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE study_plans p SET status='COMPLETED',superseded_at=NOW()
                           WHERE p.user_id=%s AND p.status='ACTIVE'
                             AND NOT EXISTS (SELECT 1 FROM study_plan_items i WHERE i.study_plan_id=p.id AND i.status<>'completed')""",(user_id,))
        conn.commit()
    finally: conn.close()

def _active_plans(user_id, include_completed=False):
    _finalize_completed_active_plans(user_id)
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            status_clause="status='ACTIVE'" if not include_completed else "status IN ('ACTIVE','COMPLETED')"
            cur.execute(f"SELECT * FROM study_plans WHERE user_id=%s AND {status_clause} ORDER BY start_date, id",(user_id,))
            plans=[]
            for row in cur.fetchall():
                p=dict(row)
                cur.execute("SELECT id,plan_date,unit_index,lesson,target,status,completed_at FROM study_plan_items WHERE study_plan_id=%s ORDER BY unit_index",(p['id'],))
                p['items']=[dict(x) for x in cur.fetchall()]
                if include_completed or any(str(x.get('status') or '').lower()!='completed' for x in p['items']):
                    plans.append(p)
            return plans
    finally: conn.close()

def _active_plan(user_id, plan_id=None):
    plans=_active_plans(user_id)
    if plan_id is not None:
        try:
            pid=int(plan_id)
        except Exception:
            pid=None
        if pid is not None:
            for p in plans:
                if int(p.get('id'))==pid:
                    return p
    return plans[0] if plans else None

def _latest_draft(user_id):
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM study_plans WHERE user_id=%s AND status='DRAFT' ORDER BY id DESC LIMIT 1",(user_id,))
            row=cur.fetchone(); return dict(row) if row else None
    finally: conn.close()

def _confirm_latest_draft(user_id):
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM study_plans WHERE user_id=%s AND status='DRAFT' ORDER BY id DESC LIMIT 1",(user_id,))
            row=cur.fetchone()
            if not row:
                return None
            pid=int(row['id'])
            parent_id=row.get('parent_plan_id')
            if parent_id:
                cur.execute("UPDATE study_plans SET status='SUPERSEDED',superseded_at=NOW() WHERE id=%s AND status='ACTIVE'",(int(parent_id),))
            cur.execute("UPDATE study_plans SET status='ACTIVE',confirmed_at=NOW() WHERE id=%s RETURNING *",(pid,))
            plan=cur.fetchone()
        conn.commit()
        return dict(plan) if plan else None
    except Exception:
        conn.rollback(); raise
    finally: conn.close()


def _study_plan_brief_for_auto_chat(user_id):
    plans=_active_plans(user_id)
    if not plans: return ""
    today=_now_local().date()
    summaries=[]
    for plan in plans[:5]:
        items=plan.get('items') or []
        todays=[x for x in items if x['plan_date']==today and str(x.get('status') or '').lower()!='completed']
        completed=[x for x in items if str(x.get('status')).lower()=='completed' and x['plan_date']<=today]
        overdue=[x for x in items if x['plan_date']<today and str(x.get('status')).lower()!='completed']
        if todays:
            if not overdue and not any(str(x.get('status') or '').lower()!='completed' for x in todays): state='ON_TRACK'
            elif overdue: state='BEHIND'
            else: state='NO_ACTIVITY'
        else:
            state='BEHIND' if overdue else 'ON_TRACK'
        target=', '.join(x['lesson'] for x in todays[:4]) or 'không có bài mới hôm nay'
        summaries.append(f"[{plan.get('content_type')}] {plan.get('goal_name')}; status={state}; mục tiêu hôm nay={target}; đã hoàn thành={len(completed)}; quá hạn={len(overdue)}")
    return (f"Hôm nay={today.isoformat()}; " + " | ".join(summaries) + " Hãy viết 1 câu ngắn, thân thiện bằng tiếng Việt để động viên/nhắc/chúc mừng phù hợp; không được tạo bài học mới.")


def _is_pure_greeting(text: str) -> bool:
    """True for a standalone greeting (including common variants), not a greeting + request."""
    low = str(text or "").strip().casefold()
    if not low:
        return False
    normalized = re.sub(r"[\W_]+", " ", low, flags=re.UNICODE).strip()
    if normalized in _GREETING_EXACT:
        return True
    patterns = [
        r"^(?:xin\s+)?chào(?:\s+(?:cậu|bạn|doraemon|nhé|nha))?$",
        r"^(?:hi|hello|hey|alo)(?:\s+(?:cậu|bạn|doraemon|nhé|nha))?$",
        r"^chào\s+buổi\s+(?:sáng|trưa|chiều|tối)$",
        r"^(?:xin\s+)?chao(?:\s+(?:cau|ban|doraemon|nhe|nha))?$",
        r"^chao\s+buoi\s+(?:sang|trua|chieu|toi)$",
    ]
    return any(re.fullmatch(pattern, normalized, flags=re.UNICODE) for pattern in patterns)


def _build_plan_choice_blocks(user_id: int, include_header: bool = True):
    """Build all active Study Plans with one inline Yes/No choice per plan."""
    plans = _active_plans(user_id)
    today = _now_local().date()
    blocks = []
    if include_header:
        blocks.append({"type":"text","text":"🎯 Doraemon đang theo dõi các lộ trình học của cậu.\n\n"})
    for plan in plans:
        items = plan.get("items") or []
        todays = [x for x in items if x.get("plan_date") == today and str(x.get("status") or "").lower() != "completed"]
        completed_items = [x for x in items if str(x.get("status") or "").lower() == "completed" and x.get("plan_date") <= today]
        overdue_items = [x for x in items if x.get("plan_date") < today and str(x.get("status") or "").lower() != "completed"]
        next_item = next((x for x in items if str(x.get("status") or "").lower() != "completed"), None)
        target_today = ", ".join(str(x.get("lesson") or "").strip() for x in todays if x.get("lesson"))
        if not target_today and next_item:
            target_today = str(next_item.get("lesson") or "").strip()
        target_today = target_today or "chưa xác định"
        if todays and not overdue_items:
            plan_state = "đang đúng tiến độ ✅"
        elif overdue_items:
            plan_state = f"đang chậm {len(overdue_items)} bài so với lộ trình ⏰"
        else:
            plan_state = "chưa có mục tiêu mới cho hôm nay"
        msg = (
            f"🎯 Lộ trình: {plan.get('goal_name') or 'Lộ trình học'}\n"
            f"🗂 Nội dung: {plan.get('content_type') or 'Giáo trình'}\n"
            f"📅 Hôm nay: {today.strftime('%d/%m/%Y')}\n"
            f"📚 Mục tiêu hôm nay: {target_today}\n"
            f"✅ Đã hoàn thành: {len(completed_items)} bài\n"
            f"📌 Tình trạng: {plan_state}\n\n"
            "Hôm nay cậu có muốn học tiếp theo lộ trình này không? 😊"
        )
        blocks.append({"type":"text","text":msg})
        plan_id = int(plan['id'])
        plan_title = str(plan.get('goal_name') or 'Lộ trình học').strip()
        blocks.append({"type":"choice","id":f"plan_today_{plan_id}","options":[
            {"label":"Có","display_label":f"Có — {plan_title}","action":f"plan_today_yes:{plan_id}"},
            {"label":"Không","display_label":f"Không — {plan_title}","action":f"plan_today_no:{plan_id}"}
        ]})
    return plans, blocks


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
    profile = _get_learning_profile(user["id"])
    active_plan = _active_plan(user["id"]) if profile.get("learning_mode") == "planned" else None
    if not profile.get("onboarding_completed") and not profile.get("learning_mode"):
        message = (f"Chào {nickname}! 👋 Tớ là Doraemon. Trước khi bắt đầu, cậu muốn tớ "
                   "lập lộ trình học theo mục tiêu cho cậu, hay cậu muốn học tự do?\n\n"
                   "Hãy chọn một trong hai lựa chọn bên dưới nhé.")
        blocks = [
            {"type":"text","text":message},
            {"type":"choice","id":"plan_choice","options":[
                {"label":"Học theo lộ trình","action":"onboarding_planned"},
                {"label":"Học tự do","action":"onboarding_free"}
            ]}
        ]
        return {"success":True,"mode":"plan_choice","message":message,"content_blocks":blocks,"learning_history":unfinished_rows}

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

    # Planned users with one or more unfinished plans are asked per plan whether
    # they want to follow that plan today. Fully completed plans are hidden.
    if profile.get("learning_mode") == "planned":
        active_plans, plan_blocks = _build_plan_choice_blocks(user["id"], include_header=False)
        if active_plans:
            header=(f"Chào {nickname}! 👋 Mừng cậu quay lại với Doraemon. 🤖\n\n{curriculum}\n\n")
            blocks=[{"type":"text","text":header.rstrip()}] + plan_blocks
            if unfinished_rows:
                seen_old=set(); parts_old=[]
                for row in unfinished_rows:
                    key=(row.get('content_type'),row.get('lesson'),row.get('topic'))
                    if key in seen_old: continue
                    seen_old.add(key)
                    label=str(row.get('content_type') or 'Nội dung')
                    detail=' '.join(str(x).strip() for x in (row.get('lesson'),row.get('topic')) if x and str(x).strip())
                    state=str(row.get('status') or '').strip().lower()
                    state_text='đang học dở' if state in {'in_progress','active'} else 'cần ôn'
                    parts_old.append(f"• {label}: {detail or 'nội dung'} – {state_text}")
                    if len(parts_old)>=8: break
                if parts_old:
                    blocks.append({"type":"text","text":"📖 Những phần cậu đang học dở/cần ôn từ các phiên học trước:\n" + "\n".join(parts_old)})
            return {"success":True,"mode":"planned_returning","message":header.rstrip(),"content_blocks":blocks,"learning_history":unfinished_rows,"study_plans":active_plans,"study_plan":active_plans[0]}

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



def _chat_model_for_content(content_type: Optional[str], provider: Optional[str] = None):
    """Resolve the chat model without changing routing/RAG decisions."""
    provider = (provider or LLM_PROVIDER).strip().lower()
    if provider == "openai":
        if content_type in {"Bài tập", "Giáo trình"}:
            return OPENAI_MODEL_MEDIUM
        return OPENAI_MODEL_LOW
    return GEMINI_MODEL



def _is_ultra_simple_casual(text: str) -> bool:
    """Detect very low-risk social utterances that do not need an LLM call."""
    s = re.sub(r"\s+", " ", (text or "").strip().casefold())
    if not s or len(s) > 40:
        return False
    exact = {
        "hehe", "haha", "hihi", "huhu", "ừ", "uh", "ok", "okay", "oke",
        "vui ghê", "vui quá", "cười xỉu", "mệt quá", "mệt ghê", "chán quá",
        "hic", "hix", "ôi", "ôi trời", "trời ơi", "hay ghê", "được đó",
        "cảm ơn", "cám ơn", "thanks", "thank you", "nice", "yay",
    }
    if s in exact:
        return True
    return bool(re.fullmatch(r"(?:ha|he|hi|hihi|haha|hehe|huhu){1,6}", s))

def _local_casual_reply(text: str) -> str:
    """Tiny deterministic replies for ultra-simple casual chat; zero LLM tokens."""
    s = re.sub(r"\s+", " ", (text or "").strip().casefold())
    if s in {"cảm ơn", "cám ơn", "thanks", "thank you"}:
        return "Không có gì nhé 😄"
    if s in {"mệt quá", "mệt ghê", "chán quá", "hic", "hix"}:
        return "Ừm, nghỉ một chút nha 🤖💙"
    if s in {"vui ghê", "vui quá", "hay ghê", "được đó", "yay", "nice", "cười xỉu"}:
        return "Hehe 😄"
    if s in {"ừ", "uh", "ok", "okay", "oke"}:
        return "Ừ nha 😄"
    if s in {"hehe", "haha", "hihi"} or re.fullmatch(r"(?:ha|he|hi|hihi|haha|hehe|huhu){1,6}", s):
        return "Hehe 😄"
    if s in {"ôi", "ôi trời", "trời ơi"}:
        return "Hehe, Doraemon đây 🤖"
    return "Ừ nha 😄"

def _is_japanese_only_message(text: str) -> bool:
    """Return True only when the whole user message is Japanese text.

    Japanese script alone is not enough: a mixed message such as
    "いろ phát âm như nào ?" must stay Vietnamese. We allow Japanese
    Hiragana/Katakana/Kanji, punctuation, whitespace, and digits, but reject
    Latin alphabet / Vietnamese text. Explicit language requests are handled
    separately by _preferred_response_language().
    """
    s = str(text or "").strip()
    if not s:
        return False
    if not re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", s):
        return False
    # Any Latin letter means the message is mixed/non-Japanese for automatic
    # detection. This intentionally makes the rule strict.
    if re.search(r"[A-Za-z]", s):
        return False
    # Vietnamese and other Latin-based accented letters are also rejected.
    # Keep Unicode punctuation, digits, spaces, and Japanese scripts allowed.
    if re.search(r"[À-ỹà-ỹ]", s, flags=re.IGNORECASE):
        return False
    # Reject other obvious non-Japanese alphabetic scripts.
    if re.search(r"[\u0370-\u03ff\u0400-\u04ff\u0530-\u058f\u0600-\u06ff\u0900-\u097f]", s):
        return False
    return True


def _preferred_response_language(user_text: str) -> str:
    """Choose response language from the latest user message.

    Automatic Japanese mode is strict: the entire message must be Japanese.
    A single Japanese word embedded in Vietnamese must never switch the reply
    language to Japanese.
    """
    text = str(user_text or "").strip()
    low = text.casefold()
    if any(x in low for x in ("bằng tiếng việt", "tiếng việt nhé", "trả lời bằng tiếng việt", "vietnamese")):
        return "vi"
    if any(x in low for x in ("bằng tiếng anh", "tiếng anh nhé", "trả lời bằng tiếng anh", "english")):
        return "en"
    if any(x in low for x in ("日本語で", "日本語で答えて", "日本語で話して", "日本語で返して")):
        return "ja"
    if _is_japanese_only_message(text):
        return "ja"
    return "vi"


def _with_global_language_directive(prompt: str, user_text: str) -> str:
    lang = _preferred_response_language(user_text)
    if lang == "ja":
        directive = (
            "GLOBAL RESPONSE-LANGUAGE DIRECTIVE (HIGHEST PRIORITY):\n"
            "The user's latest message is Japanese. Your ENTIRE response MUST be in Japanese. "
            "Do not answer in Vietnamese and do not translate into Vietnamese unless the user explicitly asks for Vietnamese. "
            "Preserve Japanese examples and terminology naturally.\n\n"
        )
    elif lang == "en":
        directive = (
            "GLOBAL RESPONSE-LANGUAGE DIRECTIVE (HIGHEST PRIORITY):\n"
            "The user explicitly requested English. Your ENTIRE response MUST be in English unless they ask for another language.\n\n"
        )
    else:
        directive = (
            "GLOBAL RESPONSE-LANGUAGE DIRECTIVE (HIGHEST PRIORITY):\n"
            "The latest user message is Vietnamese or has no clear Japanese signal. Reply in Vietnamese unless another language is explicitly requested.\n\n"
        )
    return directive + prompt


def _generate_chat_reply(prompt: str, *, content_type: Optional[str], request_id: str, gen_started: float, user_text: str = ""):
    """
    Provider-neutral chat adapter.
    Gemini remains the legacy/default provider. OpenAI is a drop-in alternative
    for the same final prompt so RAG, Study Plan, chat history, routing and
    content_blocks stay unchanged.
    """
    prompt = _with_global_language_directive(prompt, user_text)
    # Runtime audit: this adapter receives a text-only prompt. If image_parts=0,
    # no binary/image payload is being sent to Gemini from the chat generation path.
    print(f"[GEMINI INPUT AUDIT] request={request_id} prompt_chars={len(prompt)} image_parts=0")
    provider = LLM_PROVIDER
    thinking_level = (
        "medium" if content_type == "Bài tập"
        else ("minimal" if content_type is None else "low")
    )

    if provider == "openai":
        if openai_client is None:
            raise HTTPException(
                500,
                "LLM_PROVIDER=openai nhưng OPENAI_API_KEY chưa được cấu hình "
                "hoặc package openai chưa được cài."
            )
        model = _chat_model_for_content(content_type, "openai")
        print(
            f"[CHAT THINKING] request={request_id} provider='openai' "
            f"content_type={content_type!r} model={model!r} "
            f"reasoning={OPENAI_REASONING_MEDIUM if content_type == 'Bài tập' else 'none'!r}"
        )
        kwargs = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        }
        if model.startswith("gpt-5"):
            kwargs["reasoning"] = {
                "effort": OPENAI_REASONING_MEDIUM if content_type == "Bài tập" else "none"
            }
        response = openai_client.responses.create(**kwargs)
        _log_openai_usage(response, operation="chat_generation", request_id=request_id)
        reply = getattr(response, "output_text", "") or ""
        elapsed = time.perf_counter() - gen_started
        print(
            f"[CHAT OPENAI] request={request_id} model={model!r} "
            f"elapsed={elapsed:.3f}s reply_chars={len(reply)}"
        )
        return reply, model, elapsed

    if provider != "gemini":
        raise HTTPException(
            500,
            f"LLM_PROVIDER không hợp lệ: {provider!r}. "
            "Chọn 'gemini' hoặc 'openai'."
        )

    if gemini is None:
        raise HTTPException(500, "LLM_PROVIDER=gemini nhưng GEMINI_API_KEY chưa được cấu hình.")

    print(
        f"[CHAT THINKING] request={request_id} provider='gemini' "
        f"content_type={content_type!r} level={thinking_level!r}"
    )
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        ),
    )
    _log_gemini_usage(response, operation="chat_generation", request_id=request_id)
    reply = response.text or ""
    elapsed = time.perf_counter() - gen_started
    print(
        f"[CHAT GEMINI] request={request_id} model={GEMINI_MODEL!r} "
        f"elapsed={elapsed:.3f}s reply_chars={len(reply)}"
    )
    return reply, GEMINI_MODEL, elapsed


@app.post("/api/proxy-chat")
def proxy_chat(
    data: ChatRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    request_id = uuid.uuid4().hex[:12]
    perf_total = time.perf_counter()
    print(f"[CHAT START] request={request_id} message={data.text[:120]!r}")
    user = require_active_user(authorization)
    perf_auth = time.perf_counter()

    if not gemini:
        raise HTTPException(500, "Gemini chưa được khởi tạo.")
    if not data.text and not data.action:
        raise HTTPException(400, "Tin nhắn không được để trống.")

    # Paid packages are unlimited. Free is limited to 5 accepted questions/day.
    # Standalone greetings are onboarding actions and do not consume a question.
    if not _is_pure_greeting(data.text) and not data.proactive and not data.action:
        enforce_question_limit(user["id"])

    # A standalone greeting is a session/onboarding action, NOT a knowledge
    # question. Do not send "Chào" through embedding/Pinecone/Gemini, because
    # generic RAG similarity can accidentally make Doraemon start teaching a
    # random lesson (for example Bài tập) immediately after saying hello.
    profile=_get_learning_profile(user["id"])
    low0=data.text.casefold().strip()
    if data.chatbox_new and data.chatbox_id:
        old_session = _get_study_session(user["id"], None)
        if old_session and old_session.get("chatbox_id") and old_session.get("chatbox_id") != str(data.chatbox_id).strip():
            print(f"[CHATBOX RESET] new_chatbox=1 old_chatbox={old_session.get('chatbox_id')!r} -> closing previous lesson={old_session.get('lesson')!r}")
            _finish_study_session(user["id"])
        elif old_session and not old_session.get("chatbox_id"):
            print(f"[CHATBOX RESET] new_chatbox=1 legacy session -> closing previous lesson={old_session.get('lesson')!r}")
            _finish_study_session(user["id"])
    study_session = _get_study_session(user["id"], data.chatbox_id)

    # Safe default before routing is computed. Some confirmation/session branches
    # are evaluated earlier than the final hard-gate calculation below. Keeping
    # this initialized here prevents UnboundLocalError and, importantly, defaults
    # to CLOSED rather than accidentally enabling study retrieval.
    study_retrieval_allowed = False

    # End-of-lesson confirmation is a pure state transition: no Gemini,
    # embedding, Pinecone, RAG, images, or question quota.
    if ui_action_raw := (str(data.action or "").strip() or None):
        ui_action_early = ui_action_raw.split(":",1)[0].casefold()
        if ui_action_early == "study_end_yes":
            lesson_label = (study_session or {}).get("lesson") or "bài học này"
            _finish_study_session(user["id"])
            msg = f"✅ Được nhé! Doraemon đã kết thúc bài học **{lesson_label}** hôm nay. Khi nào cậu muốn học bài mới, hãy nói với tớ nhé! 🤖"
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
        if ui_action_early == "study_end_no":
            if study_session:
                _set_study_end_prompt_pending(user["id"], False)
                study_session["end_prompt_pending"] = False
                scope = _active_session_scope(study_session)
                lesson_label = scope.get("lesson") or "bài này"
                msg = f"Được nhé! 🤖 Mình vẫn giữ bài **{lesson_label}** đang mở. Cậu cứ hỏi tiếp phần đang học."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

    # The current open chatbox supplies the authoritative conversational context.
    # Keep the latest 20 messages (= up to 10 user/model exchanges) available
    # BEFORE any Study Plan routing so a follow-up like "học trong 5 ngày" keeps
    # the subtype from the earlier message in the SAME chatbox.
    # A new chatbox may use at most 10 recent user/model exchanges for its
    # initial context. Once the boxchat is open, the frontend continues sending
    # the current thread history; the server only keeps the same bounded window.
    # `chatbox_new` is optional for backward compatibility.
    plan_recent_history = _normalize_chat_history(data.chat_history, max_messages=20)
    if data.chatbox_new:
        plan_recent_history = plan_recent_history[-20:]
        print(f"[CHATBOX CONTEXT] new_chatbox=1 history_messages={len(plan_recent_history)} initial_only=10_exchanges")
    elif plan_recent_history:
        print(f"[CHATBOX CONTEXT] existing_chatbox history_messages={len(plan_recent_history)} bounded=10_exchanges")
    if plan_recent_history:
        print(f"[CHAT HISTORY PLAN] messages={len(plan_recent_history)}")

    # Structured UI actions from the Study Plan confirmation buttons.
    # These actions bypass text intent parsing and RAG completely.
    ui_action_raw = (str(data.action or "").strip() or None)
    ui_action = (ui_action_raw.split(":",1)[0].casefold() if ui_action_raw else None)
    action_plan_id = (ui_action_raw.split(":",1)[1] if ui_action_raw and ":" in ui_action_raw else None)
    if ui_action:
        print(f"[STUDY PLAN ACTION] user={user['id']} action={ui_action} plan_id={action_plan_id or '-'}")

    # Fixed Giáo trình curriculum flow actions. Only Giáo trình uses this state machine;
    # other content types keep the v19.8 flow unchanged.
    if ui_action in {"curriculum_finish_yes", "curriculum_finish_no"}:
        if ui_action == "curriculum_finish_yes" and study_session and study_session.get("content_type") == "Giáo trình":
            lesson_label=(study_session or {}).get("lesson") or "bài học này"
            _finish_study_session(user["id"])
            msg=f"✅ Tuyệt vời! Doraemon đã ghi nhận cậu đã hoàn thành **{lesson_label}**. Hẹn gặp cậu ở bài tiếp theo nhé! 🤖"
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
        if ui_action == "curriculum_finish_no" and study_session and study_session.get("content_type") == "Giáo trình":
            _set_curriculum_flow(user["id"], waiting="review")
            msg="Được nhé! 🤖 Cậu muốn Doraemon ôn lại phần nào của bài? Cậu có thể nói tên phần, từ, ngữ pháp hoặc câu hỏi cậu chưa chắc."
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

    if ui_action == "curriculum_next" and study_session and study_session.get("content_type") == "Giáo trình":
        try:
            expected=int(action_plan_id or -1)
        except Exception:
            expected=-1
        current=int(study_session.get("curriculum_step") or 0)
        if expected == current:
            step=current+1
            # The exact next-state classification is resolved after cache lookup below.
            _set_curriculum_flow(user["id"], step=step, waiting="continue", exercise_answered=False)
            study_session["curriculum_step"]=step
            study_session["curriculum_waiting"]="continue"
            study_session["curriculum_exercise_answered"]=False
            print(f"[CURRICULUM FLOW] advance user={user['id']} from={current} to={step}")

    if ui_action == "curriculum_next" and (not study_session or study_session.get("content_type") != "Giáo trình"):
        pass

    # Lesson confirmation actions are explicit intent confirmation.
    # They bypass routing/RAG only when YES; NO simply cancels the pending lesson open.
    lesson_confirmed_scope = None
    if ui_action in {"lesson_confirm_yes", "lesson_confirm_no"} and action_plan_id:
        decoded = _decode_lesson_confirm_scope(action_plan_id)
        if decoded and decoded.get("lesson"):
            if ui_action == "lesson_confirm_no":
                lesson_label = decoded.get("lesson") or "bài này"
                msg = f"Được nhé! 🤖 Doraemon chưa mở **{lesson_label}**. Cậu có thể nói bài khác mà cậu muốn học."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            lesson_confirmed_scope = decoded
            _start_study_session(user["id"], lesson_confirmed_scope, data.chatbox_id)
            study_session = dict(_get_study_session(user["id"], data.chatbox_id) or {})
            print(f"[LESSON CONFIRM] user={user['id']} confirmed scope={lesson_confirmed_scope}; study_session=ACTIVE")

    plan_start_action = None
    if ui_action:
        if ui_action == "onboarding_planned":
            _set_learning_profile(user["id"], "planned", True)
            msg = ("Tuyệt! 🤖 Cậu muốn Doraemon lập lộ trình cho mình.\n\n"
                   "Hãy cho Doraemon biết mục tiêu, ví dụ: 'Mình muốn học hết giáo trình N5 trong 1 tháng', "
                   "'Mỗi ngày 1 bài' hoặc '2 ngày học 1 bài'. Doraemon sẽ tính lộ trình để cậu xem và xác nhận trước khi áp dụng.")
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],
                    "content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

        elif ui_action == "onboarding_free":
            _set_learning_profile(user["id"], "free", True)
            msg = ("Được nhé! 🤖 Từ giờ cậu cứ học tự do theo nhu cầu.\n\n"
                   "Khi nào muốn có lộ trình, cậu chỉ cần nói với Doraemon nhé.")
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],
                    "content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

        elif ui_action == "plan_today_yes":
            # The selected plan is identified by action suffix: plan_today_yes:<plan_id>.
            # This avoids ambiguity when several plans show identical Có/Không buttons.
            profile = _get_learning_profile(user["id"])
            active_plan = _active_plan(user["id"], action_plan_id) if profile.get("learning_mode") == "planned" else None
            planned_start_item = next((x for x in (active_plan or {}).get("items", [])
                                       if str(x.get("status") or "").lower() != "completed"), None)
            if not planned_start_item:
                msg = "🎉 Lộ trình này đã hoàn thành hoặc không còn bài chưa học hôm nay."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None,"study_plan":active_plan}
            forced_content_type = _normalize_content_type((active_plan or {}).get("content_type") or "Giáo trình")
            forced_lesson = str(planned_start_item.get("lesson") or "").strip()
            print(f"[STUDY PLAN] today_yes user={user['id']} plan={active_plan.get('id') if active_plan else None} lesson={forced_lesson!r} content_type={forced_content_type!r}")
            plan_start_action = {"content_type": forced_content_type, "lesson": forced_lesson, "plan": active_plan}
            _start_study_session(user["id"], {"content_type":forced_content_type,"lesson":forced_lesson,"topic":None,"course":None}, data.chatbox_id)
            study_session = dict(_get_study_session(user["id"], data.chatbox_id) or {})

        elif ui_action == "plan_today_no":
            # The selected plan is identified by action suffix: plan_today_no:<plan_id>.
            msg = "Được nhé! 🤖 Hôm nay cậu có thể học tự do theo nhu cầu. Lộ trình này vẫn được giữ nguyên."
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

        elif ui_action == "plan_apply_draft":
            draft = _latest_draft(user["id"])
            if not draft:
                msg = "🤖 Doraemon không còn thấy lộ trình chờ xác nhận. Cậu hãy yêu cầu Doraemon lập lại lộ trình nhé."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            try:
                plan = _confirm_latest_draft(user["id"])
            except Exception as exc:
                print(f"[STUDY PLAN] UI confirm draft error user={user['id']}: {exc}")
                msg = "🤖 Doraemon chưa áp dụng được lộ trình lúc này. Cậu thử lại nhé."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            if not plan:
                msg = "🤖 Doraemon không thể áp dụng lộ trình lúc này. Cậu hãy yêu cầu Doraemon lập lại lộ trình nhé."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            _clear_pending_plan_request(user["id"])
            msg = ("🎯 Lộ trình đã được áp dụng rồi nhé! 🤖\n\n"
                   "Doraemon sẽ theo dõi tiến độ của cậu theo lộ trình này từ hôm nay.\n\n"
                   "Cậu có muốn học luôn bài đầu tiên theo lộ trình không? 😊")
            blocks = [{"type":"text","text":msg},{"type":"choice","id":"plan_start",
                     "options":[{"label":"Có","action":f"plan_start:{int(plan.get('id'))}"},{"label":"Không","action":f"plan_start_cancel:{int(plan.get('id'))}"}]}]
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":blocks,"learning_progress":None,"study_plan":plan}

        if ui_action == "plan_cancel_draft":
            draft = _latest_draft(user["id"])
            if draft:
                conn=db()
                try:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE study_plans SET status='CANCELLED' WHERE id=%s",(draft['id'],))
                    conn.commit()
                finally:
                    conn.close()
            _clear_pending_plan_request(user["id"])
            msg="Được nhé! 🤖 Doraemon giữ nguyên lộ trình hiện tại."
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

        if ui_action == "plan_start":
            profile = _get_learning_profile(user["id"])
            active_plan = _active_plan(user["id"], action_plan_id) if profile.get("learning_mode") == "planned" else None
            planned_start_item = next((x for x in (active_plan or {}).get("items",[]) if str(x.get("status")).lower() != "completed"), None)
            if not planned_start_item:
                msg="🎉 Cậu đã hoàn thành toàn bộ các bài hiện có trong lộ trình rồi nhé!"
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None,"study_plan":active_plan}
            # Build a strict scope for the regular RAG pipeline below.
            forced_content_type = _normalize_content_type((active_plan or {}).get("content_type") or "Giáo trình")
            forced_lesson = str(planned_start_item.get("lesson") or "").strip()
            msg = f"🤖 Được nhé! Mình cùng học **{forced_lesson}** theo lộ trình nào.\n"
            # Do not return here: continue into the existing RAG path, but force the exact plan scope.
            plan_start_action = {"content_type": forced_content_type, "lesson": forced_lesson, "plan": active_plan}
            _start_study_session(user["id"], {"content_type":forced_content_type,"lesson":forced_lesson,"topic":None,"course":None}, data.chatbox_id)
            study_session = dict(_get_study_session(user["id"], data.chatbox_id) or {})
        elif ui_action == "plan_start_cancel":
            msg="Được nhé! 🤖 Khi nào cậu muốn bắt đầu bài đầu tiên theo lộ trình, chỉ cần nói với Doraemon."
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
    # Learning-mode onboarding and explicit mode switch.
    plan_intent = any(k in low0 for k in [
        "học theo lộ trình",
        "học lộ trình",
        "muốn lộ trình",
        "cần lộ trình",
        "có lộ trình",
        "lộ trình học",
        "theo lộ trình",
        "tạo lộ trình",
        "lập lộ trình",
        "thêm lộ trình",
        "tạo một lộ trình",
        "lập một lộ trình"
    ])
    # Explicitly creating a NEW plan must take precedence over the default
    # "continue an existing plan" behavior. This allows a user who already
    # has Grammar plan to say e.g. "mình muốn tạo lộ trình học Bộ thủ" and
    # start a separate Vocabulary plan instead of being routed to Grammar.
    new_plan_intent = any(k in low0 for k in [
        "tạo lộ trình", "lập lộ trình", "thêm lộ trình",
        "tạo một lộ trình", "lập một lộ trình", "tạo thêm lộ trình",
        "lập thêm lộ trình"
    ])
    free_intent = any(k in low0 for k in ["học tự do","tự do","không cần lộ trình","không cần lịch trình"])

    # Brand-new learner: choose a mode once.
    if not profile.get("learning_mode") and not _is_pure_greeting(data.text):
        if free_intent:
            _set_learning_profile(user["id"],"free",True)
            return {"reply":"Được nhé! 🤖 Từ giờ cậu cứ học tự do theo nhu cầu. Khi nào muốn có lộ trình, hãy nói với Doraemon là cậu muốn học theo lộ trình nhé.","model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":"Được nhé! 🤖 Từ giờ cậu cứ học tự do theo nhu cầu. Khi nào muốn có lộ trình, hãy nói với Doraemon là cậu muốn học theo lộ trình nhé."}],"learning_progress":None}
        if plan_intent:
            _set_learning_profile(user["id"],"planned",True)
            return {"reply":"Tuyệt! 🤖 Cậu cho Doraemon biết mục tiêu nhé. Ví dụ: 'Mình muốn học hết giáo trình N5 trong 1 tháng' hoặc 'Mỗi ngày 1 bài'. Doraemon sẽ tính lộ trình cho cậu.","model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":"Tuyệt! 🤖 Cậu cho Doraemon biết mục tiêu nhé. Ví dụ: 'Mình muốn học hết giáo trình N5 trong 1 tháng' hoặc 'Mỗi ngày 1 bài'. Doraemon sẽ tính lộ trình cho cậu."}],"learning_progress":None}

    # A user who previously chose Free can explicitly switch to Planned later.
    # Any explicit plan-intent phrase (including shorthand such as "học lộ trình")
    # must be handled before RAG, otherwise it can be mistaken for a generic
    # learning request and Pinecone/Gemini may start teaching an unrelated lesson.
    if profile.get("learning_mode") == "free" and plan_intent and not _is_pure_greeting(data.text):
        _set_learning_profile(user["id"],"planned",True)
        return {"reply":"Được nhé! 🤖 Cậu muốn chuyển sang học theo lộ trình. Hãy cho Doraemon biết mục tiêu, ví dụ: 'Mình muốn học hết giáo trình N5 trong 1 tháng', 'Mỗi ngày 1 bài' hoặc '2 ngày học 1 bài'. Doraemon sẽ tính lộ trình mới để cậu xem và xác nhận trước khi áp dụng.","model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":"Được nhé! 🤖 Cậu muốn chuyển sang học theo lộ trình. Hãy cho Doraemon biết mục tiêu, ví dụ: 'Mình muốn học hết giáo trình N5 trong 1 tháng', 'Mỗi ngày 1 bài' hoặc '2 ngày học 1 bài'. Doraemon sẽ tính lộ trình mới để cậu xem và xác nhận trước khi áp dụng."}],"learning_progress":None}

    # A structured plan-start button action has already selected the exact item;
    # skip the text-based confirmation rules and enter the forced RAG path below.
    planned_start_item = None
    active_plan = None
    if plan_start_action:
        active_plan = plan_start_action["plan"]
        planned_start_item = next((x for x in active_plan.get("items",[]) if str(x.get("status")).lower() != "completed" and str(x.get("lesson") or "").strip() == plan_start_action["lesson"]), None)

    # If a previous message requested a NEW plan and asked the user for the target,
    # consume the next concrete target message here before any RAG. This prevents the
    # target sentence (e.g. "mỗi ngày 10 từ bộ thủ") from being mistaken for a lesson.
    pending = _get_pending_plan_request(user["id"]) if profile.get("learning_mode") == "planned" else None
    if pending and pending.get('pending_plan_created_at') and not data.action:
        pending_db_type = pending.get('pending_plan_content_type')
        history_type = _infer_plan_content_type_from_history(plan_recent_history)
        current_type = _plan_content_type_from_text(data.text)
        pending_type = current_type or history_type or pending_db_type or 'Giáo trình'
        pending_scope = pending.get('pending_plan_scope') or ''
        probe = _parse_plan_request(data.text)
        print(
            f"[STUDY PLAN CONTEXT] current_type={current_type!r} "
            f"history_type={history_type!r} db_pending_type={pending_db_type!r} "
            f"resolved_type={pending_type!r}"
        )
        has_target = bool(probe.get('target_date') or probe.get('units_per_day') or probe.get('days_per_unit'))
        if has_target:
            if not pending_type:
                pending_type = probe.get('content_type') or 'Giáo trình'
            probe['content_type'] = pending_type
            probe['scope'] = pending_scope or probe.get('scope') or ''
            print(f"[STUDY PLAN] pending lock content_type={pending_type!r} scope={probe['scope']!r} parsed_type={probe.get('content_type')!r} goal={data.text!r}")
            # A daily count without a finite horizon needs one more piece of information.
            if probe.get('units_per_day') and not probe.get('target_date') and not probe.get('days_per_unit'):
                msg=(f"🤖 Doraemon đã ghi nhận mục tiêu {probe.get('units_per_day'):g} {probe.get('unit_label') or 'mục'}/ngày cho {pending_scope or pending_type}. "
                     "Để tính lộ trình cụ thể, cậu cho Doraemon biết muốn duy trì mục tiêu này trong bao nhiêu ngày nhé.")
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            _clear_pending_plan_request(user["id"])
            pid,preview=_build_plan_preview(user["id"],probe)
            blocks=[{"type":"text","text":preview},{"type":"choice","id":"plan_draft","options":[{"label":"Có","action":"plan_apply_draft"},{"label":"Không","action":"plan_cancel_draft"}]}]
            return {"reply":preview,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":blocks,"learning_progress":None,"study_plan_draft_id":pid}
    # Planned-user confirmations and plan creation/update requests happen before RAG.
    # A short affirmative response after the "học luôn bài đầu tiên" prompt is a
    # structured command to start the FIRST ACTIVE plan item; it must never fall
    # through to generic RAG (which can select an unrelated lesson such as Ngữ pháp Bài 1).
    if profile.get("learning_mode")=="planned" and not plan_start_action:
        active_plan=_active_plan(user["id"])
        draft=_latest_draft(user["id"])
        # IMPORTANT: once a user already has an ACTIVE Study Plan, any explicit
        # plan-intent phrase such as "mình muốn học theo lộ trình" or
        # "mình muốn học lộ trình" means "start/continue my plan" rather than
        # a generic knowledge question. Handle this before RAG so it can never
        # drift to an unrelated lesson such as Ngữ pháp Bài 1.
        if active_plan and plan_intent and not draft:
            # Explicit NEW-plan requests are handled by the dedicated creation flow.
            # A generic request to "học theo lộ trình" should show ALL current plans
            # and let the user choose which one to follow today. Never send this to RAG.
            if new_plan_intent:
                pass
            else:
                probe = _parse_plan_request(data.text)
                has_concrete_target = bool(probe.get('target_date') or probe.get('units_per_day') or probe.get('days_per_unit'))
                if has_concrete_target:
                    pass
                else:
                    plans_all, plan_blocks = _build_plan_choice_blocks(user["id"], include_header=False)
                    if plans_all:
                        msg="🎯 Cậu đang có các lộ trình học đang theo dõi. Hôm nay cậu muốn tiếp tục lộ trình nào?"
                        blocks=[{"type":"text","text":msg}] + plan_blocks
                        return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":blocks,"learning_progress":None,"study_plans":plans_all}

        completion_words=("đã học xong","học xong rồi","mình học xong","hoàn thành bài này","xong bài này","đã hoàn thành")
        if any(w in low0 for w in completion_words):
            # Match the completion to the most recent in-progress learning record
            # so completing Grammar Bài 1 cannot complete a separate Vocabulary plan.
            latest_lp=None
            try:
                conn0=db()
                with conn0.cursor(cursor_factory=RealDictCursor) as cur0:
                    cur0.execute("SELECT content_type,lesson FROM learning_progress WHERE user_id=%s AND LOWER(COALESCE(status,'')) IN ('in_progress','active','review') ORDER BY last_studied_at DESC LIMIT 1",(user['id'],))
                    latest_lp=cur0.fetchone()
            finally:
                try: conn0.close()
                except Exception: pass
            candidates=_active_plans(user["id"])
            chosen=active_plan
            if latest_lp:
                for cand in candidates:
                    if _normalize_content_type(cand.get('content_type') or '') == _normalize_content_type(latest_lp.get('content_type') or ''):
                        if any(str(x.get('lesson') or '').casefold()==str(latest_lp.get('lesson') or '').casefold() and str(x.get('status') or '').lower()!='completed' for x in cand.get('items') or []):
                            chosen=cand; break
            if chosen:
                upcoming=[x for x in chosen.get('items',[]) if str(x.get('status')).lower()!='completed']
            else: upcoming=[]
            if upcoming:
                item=upcoming[0]
                record_learning_event(user["id"], {"content_type":chosen.get('content_type') or 'Giáo trình',"subject":"Tiếng Nhật","lesson":item.get('lesson'),"topic":"","item_key":item.get('lesson'),"status":"completed","completed":True})
                conn=db()
                try:
                    with conn.cursor() as cur: cur.execute("UPDATE study_plan_items SET status='completed',completed_at=NOW() WHERE id=%s",(item['id'],))
                    conn.commit()
                finally: conn.close()
                next_item=next((x for x in chosen.get('items',[]) if int(x.get('unit_index') or 0)>int(item.get('unit_index') or 0) and str(x.get('status')).lower()!='completed'),None)
                nxt=f" Bài tiếp theo theo lộ trình là {next_item.get('lesson')}." if next_item else " Cậu đã hoàn thành toàn bộ lộ trình hiện tại rồi! 🎉"
                msg=f"🎉 Tuyệt vời! Doraemon đã đánh dấu '{item.get('lesson')}' là hoàn thành.{nxt}"
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":{"status":"completed","lesson":item.get('lesson')}}
        # Plan-start confirmation is now handled exclusively by the UI buttons.
        # Draft confirmation is UI-only in v14.1; do not interpret free text as a confirmation command.
        if False and draft and low0 in {"có","ok","đồng ý","xác nhận","áp dụng","được","ừ","ừ được"}:
            try:
                plan=_confirm_latest_draft(user["id"])
            except Exception as exc:
                print(f"[STUDY PLAN] confirm draft error user={user['id']}: {exc}")
                msg="🤖 Doraemon chưa áp dụng được lộ trình lúc này. Cậu thử bấm xác nhận thêm một lần nữa nhé."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            if not plan:
                msg="🤖 Doraemon không còn thấy bản lộ trình chờ xác nhận. Cậu hãy yêu cầu Doraemon lập lại lộ trình nhé."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            msg=("🎯 Lộ trình đã được áp dụng rồi nhé! 🤖\n\n"
                 "Doraemon sẽ theo dõi tiến độ của cậu theo lộ trình này từ hôm nay. "
                 "Cậu có muốn học luôn bài đầu tiên theo lộ trình không? 😊")
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None,"study_plan":plan}
        if draft and low0 in {"không","không nhé","giữ nguyên","chưa","hủy"}:
            conn=db();
            try:
                with conn.cursor() as cur: cur.execute("UPDATE study_plans SET status='CANCELLED' WHERE id=%s",(draft['id'],))
                conn.commit()
            finally: conn.close()
            return {"reply":"Được nhé, Doraemon giữ nguyên lộ trình hiện tại.","model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":"Được nhé, Doraemon giữ nguyên lộ trình hiện tại."}],"learning_progress":None}
        if any(k in low0 for k in ["sửa lộ trình","đổi lộ trình","thay đổi lộ trình","sửa lại lộ trình","đổi mục tiêu"]):
            req=_parse_plan_request(data.text)
            if not req.get('target_date') and not req.get('units_per_day') and not req.get('days_per_unit'):
                msg="🤖 Cậu muốn sửa lộ trình như thế nào? Hãy nói rõ ngày mục tiêu mới hoặc tốc độ học, ví dụ: 'đổi mục tiêu sang 15/09' hoặc 'từ ngày 01/09, mỗi 2 ngày học 1 bài'."
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            oldplan=None
            requested_type=_parse_plan_request(data.text).get('content_type')
            for candidate in _active_plans(user["id"]):
                if _normalize_content_type(candidate.get('content_type') or '') == _normalize_content_type(requested_type or candidate.get('content_type') or ''):
                    oldplan=candidate
                    break
            if oldplan is None:
                oldplan=_active_plan(user["id"])
            if oldplan:
                # Rebuild from the remaining lessons, preserving completed history.
                done={x['lesson'] for x in oldplan['items'] if str(x.get('status')).lower()=='completed'}
                lessons=[x for x in _unique_lessons_for_scope(oldplan.get('content_type') or 'Giáo trình', oldplan.get('scope') or '') if x not in done]
                req['content_type']=oldplan.get('content_type') or 'Giáo trình'; req['scope']=oldplan.get('scope') or ''; req['goal']=data.text
                req['_lessons_override']=lessons; req['target_date']=req.get('target_date') or oldplan.get('target_date')
                # Simplified rebuild: create a draft using remaining lessons by temporarily using helper's catalog selection.
                # For consistency, if override exists we build the same plan shape manually below.
                req['override_done']=done; req['parent_plan_id']=oldplan.get('id')
            pid,preview=_build_plan_preview(user["id"],req)
            return {"reply":preview,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":preview},{"type":"choice","id":"plan_draft","options":[{"label":"Có","action":"plan_apply_draft"},{"label":"Không","action":"plan_cancel_draft"}]}],"learning_progress":None,"study_plan_draft_id":pid}
        # Plan creation. Existing active plans do not block a new independent plan.
        # Explicit NEW-plan requests always create a separate draft, even when
        # another plan of a different content type is already active.
        if new_plan_intent and not _is_pure_greeting(data.text):
            req_probe = _parse_plan_request(data.text)
            has_target = bool(req_probe.get('target_date') or req_probe.get('units_per_day') or req_probe.get('days_per_unit'))
            if not has_target:
                requested_type = _plan_content_type_from_text(data.text) or _infer_plan_content_type_from_history(plan_recent_history) or req_probe.get('content_type') or 'Giáo trình'
                scope_hint = req_probe.get('scope') or ''
                _set_pending_plan_request(user["id"], requested_type, scope_hint)
                msg=("Được nhé! 🤖 Doraemon sẽ tạo một lộ trình mới riêng cho cậu.\n\n"
                     f"Cậu muốn đặt mục tiêu cho **{requested_type}** như thế nào? Ví dụ: \"Mỗi ngày 1 bài\", \"2 ngày học 1 bài\", \"mỗi ngày 10 từ bộ thủ trong 7 ngày\" hoặc \"hoàn thành trong 10 ngày\".\n\n"
                     "Lộ trình hiện tại vẫn được giữ nguyên. Sau khi Doraemon tính xong, cậu sẽ xem và xác nhận trước khi áp dụng.")
                return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            if profile.get("learning_mode") != "planned":
                _set_learning_profile(user["id"], "planned", True)
            req_probe['content_type'] = _plan_content_type_from_text(data.text) or req_probe.get('content_type') or _infer_plan_content_type_from_history(recent_history) or 'Giáo trình'
            print(f"[STUDY PLAN] new-plan target content_type={req_probe['content_type']!r} source=current/history")
            pid,preview=_build_plan_preview(user["id"],req_probe)
            return {"reply":preview,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":preview},{"type":"choice","id":"plan_draft","options":[{"label":"Có","action":"plan_apply_draft"},{"label":"Không","action":"plan_cancel_draft"}]}],"learning_progress":None,"study_plan_draft_id":pid}

        # Plan creation. Existing active plans do not block a new independent plan.
        # Only explicit plan intent or concrete plan parameters trigger this branch.
        if not draft:
            if plan_intent and not _is_pure_greeting(data.text):
                # If it is only a generic mode request, ask for the target instead of RAG.
                req_probe=_parse_plan_request(data.text)
                has_target = bool(req_probe.get('target_date') or req_probe.get('units_per_day') or req_probe.get('days_per_unit'))
                if not has_target:
                    if profile.get("learning_mode") != "planned":
                        _set_learning_profile(user["id"], "planned", True)
                    req_type = _plan_content_type_from_text(data.text) or _infer_plan_content_type_from_history(plan_recent_history) or req_probe.get('content_type') or 'Giáo trình'
                    print(f"[STUDY PLAN] create request lock content_type={req_type!r} source=current/history")
                    _set_pending_plan_request(user["id"], req_type, req_probe.get('scope') or '')
                    msg=("Tuyệt! 🤖 Doraemon sẽ lập lộ trình cho cậu.\n\n"
                         "Cậu cho Doraemon biết mục tiêu cụ thể nhé. Ví dụ: "
                         "'Mình muốn học hết giáo trình N5 trong 1 tháng', "
                         "'Mỗi ngày 1 bài', '2 ngày học 1 bài', 'mỗi ngày 10 từ bộ thủ trong 7 ngày' hoặc 'mình muốn có lộ trình ngữ pháp 1 bài/ngày'. "
                         "Doraemon sẽ tính lộ trình để cậu xem và xác nhận trước khi áp dụng.")
                    return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}
            req=_parse_plan_request(data.text)
            if req.get('target_date') or req.get('units_per_day') or req.get('days_per_unit'):
                if profile.get("learning_mode") != "planned":
                    _set_learning_profile(user["id"], "planned", True)
                req['content_type'] = _plan_content_type_from_text(data.text) or _infer_plan_content_type_from_history(plan_recent_history) or req.get('content_type') or 'Giáo trình'
                print(f"[STUDY PLAN] concrete target content_type={req['content_type']!r} source=current/history")
                pid,preview=_build_plan_preview(user["id"],req)
                return {"reply":preview,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":preview},{"type":"choice","id":"plan_draft","options":[{"label":"Có","action":"plan_apply_draft"},{"label":"Không","action":"plan_cancel_draft"}]}],"learning_progress":None,"study_plan_draft_id":pid}

    if _is_pure_greeting(data.text):
        welcome = _build_welcome_for_user(user, mark_seen=False)
        return {
            "reply": welcome["message"],
            "model": GEMINI_MODEL,
            "sources": [],
            "images": [],
            "content_blocks": welcome.get("content_blocks") or [{"type":"text","text":welcome["message"]}],
            "study_plans": welcome.get("study_plans") or [],
            "learning_history_count": len(welcome.get("learning_history") or []),
            "learning_progress": None,
            "welcome": True,
            "welcome_mode": welcome.get("mode"),
        }

    namespace = data.knowledge_namespace or "__default__"
    query_text = data.text
    if data.proactive:
        plan_hint = _study_plan_brief_for_auto_chat(user["id"])
        if plan_hint:
            query_text = plan_hint

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

    # Force retrieval to the exact unit from the ACTIVE Study Plan when the
    # user just confirmed "Có" to start learning. This keeps the old RAG flow
    # intact for all other questions while preventing accidental lesson drift.
    forced_plan_scope = None
    if planned_start_item:
        forced_plan_scope = {
            "course": None,
            "content_type": _normalize_content_type(active_plan.get("content_type") or "Giáo trình"),
            "lesson": str(planned_start_item.get("lesson") or "").strip() or None,
            "topic": None,
        }
        query_text = (
            f"Hãy bắt đầu dạy đúng bài đầu tiên trong lộ trình: {forced_plan_scope['lesson']}. "
            f"Loại nội dung: {forced_plan_scope['content_type']}. "
            "Không chuyển sang lesson hoặc content type khác. "
            "Dạy theo đúng nội dung có trong kho kiến thức của bài này."
        )

    low = query_text.strip().lower()
    ambiguous_study_request = _is_ambiguous_study_request(query_text)
    general_non_learning_request = _is_general_non_learning_request(query_text)
    casual_conversation_request = (
        _is_casual_conversation_request(query_text)
        or _is_lightweight_casual_message(query_text)
    )
    exercise_suggestion_only_request = _is_exercise_suggestion_only_request(query_text)
    if general_non_learning_request:
        print("[CHAT ROUTING] general non-learning request: bypass study RAG/images/suggestions")
        minimal_prompt = f"""Bạn là Doraemon. Đây là câu hỏi đời thường, không phải yêu cầu học tiếng Nhật.
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.

Trả lời trực tiếp, ngắn gọn và thân thiện. Không giới thiệu bài học, không gợi ý bài tập, không nhắc lộ trình, không đính kèm ảnh học tập.
Nếu câu hỏi yêu cầu dữ liệu thời gian thực mà hệ thống không có công cụ truy cập dữ liệu đó, hãy nói rõ bạn chưa có dữ liệu thời gian thực thay vì đoán.

Câu hỏi của người dùng:
{query_text}"""
        gen_started = time.perf_counter()
        reply, model_used, _ = _generate_chat_reply(minimal_prompt, content_type=None, request_id=request_id, gen_started=gen_started, user_text=query_text)
        return {"reply": reply, "model": model_used, "sources": [], "images": [], "content_blocks": [{"type":"text","text":reply}], "learning_progress": None}

    # Conversational memory / CURRENT CHAT THREAD:
    # The open chatbox is the primary conversational context. Keep at most
    # 10 recent exchanges (up to 20 messages: user+model) from the history
    # supplied by the current chatbox. A new chatbox normally starts with an
    # empty history, so this does not leak the previous chat into a new thread.
    #
    # We deliberately keep more than the old 4-message window because a short
    # correction often refers to something said several turns earlier.
    # Reuse the normalized CURRENT chatbox history that was prepared before plan routing.
    recent_history = plan_recent_history

    # A single compact text view is used for routing/embedding. The full
    # normalized 20-message window is still available to Gemini below.
    thread_history_for_rag = recent_history[-20:]
    thread_history_text = "\n".join(
        f"{h['role']}: {h['text']}" for h in thread_history_for_rag
    )
    if len(thread_history_text) > 9000:
        thread_history_text = thread_history_text[-9000:]

    # Pinecone is required only once the request has passed lightweight
    # conversation routing and is actually entering study retrieval.
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")

    # Resolve explicit intent before semantic retrieval. Kanji/Bộ thủ are
    # lessons under Từ vựng, never standalone content types.
    #
    # IMPORTANT: when this boxchat is still open, its recent history is the
    # primary context. Durable PostgreSQL progress is only a fallback. A clear
    # "chuyển sang..." request is the explicit exception that allows switching.
    thread_scope = _extract_thread_scope(recent_history, catalog)
    thread_switch_requested = _is_explicit_thread_switch(query_text)
    next_lesson_scope = None
    if thread_switch_requested and thread_scope and any(
        phrase in low for phrase in ("học bài tiếp", "học bài tiếp theo", "học tiếp bài", "bài tiếp theo")
    ):
        next_lesson_scope = _catalog_next_lesson(
            catalog, thread_scope.get("content_type"), thread_scope.get("lesson")
        )
        if next_lesson_scope:
            print(
                "[CHAT THREAD] next-lesson switch "
                f"from={thread_scope.get('lesson')!r} to={next_lesson_scope.get('lesson')!r}"
            )
    if thread_scope:
        print(
            "[CHAT THREAD] "
            f"history_messages={len(recent_history)} "
            f"scope={thread_scope} "
            f"switch_requested={thread_switch_requested}"
        )

    # Casual/emotional chat stays inside the open conversation but deliberately
    # bypasses embedding, Pinecone RAG and image retrieval. We keep only a small
    # tail of the current boxchat as lightweight context so a message like
    # "mệt quá hic" is understood naturally without paying for the study stack.
    # Explicit thread switches/study targets always take precedence.
    if (
        casual_conversation_request
        and not thread_switch_requested
        and not data.action
        and not ambiguous_study_request
    ):
        if _is_ultra_simple_casual(query_text):
            reply = _local_casual_reply(query_text)
            print(
                f"[CHAT ROUTING] ultra-simple casual fast-path: no Gemini "
                f"tokens request={request_id} text={query_text!r}"
            )
            return {
                "reply": reply,
                "model": "local-casual-fastpath",
                "sources": [],
                "images": [],
                "content_blocks": [{"type": "text", "text": reply}],
                "learning_progress": None,
            }

        casual_history = recent_history[-4:]
        casual_context_parts = []
        for h in casual_history:
            role = str(h.get("role") or "user")
            txt = str(h.get("text") or "").strip()
            if txt:
                casual_context_parts.append(f"{role}: {txt[-450:]}")
        casual_context = "\n".join(casual_context_parts)
        thread_hint = ""
        if thread_scope:
            scope_parts = [
                str(thread_scope.get("content_type") or ""),
                str(thread_scope.get("lesson") or ""),
                str(thread_scope.get("topic") or ""),
            ]
            scope_label = " / ".join(x for x in scope_parts if x)
            if scope_label:
                thread_hint = f"\nNgữ cảnh nhẹ của boxchat hiện tại: {scope_label}."

        minimal_prompt = f"""Bạn là Doraemon, một người bạn/gia sư thân thiện.
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.

Đây là câu trò chuyện đời thường/cảm xúc, KHÔNG phải yêu cầu truy xuất hay dạy bài học.
Trả lời tự nhiên, ngắn gọn, đồng cảm và không tự mở bài học mới.
Không gọi lại giáo trình, không đề xuất bài tập, không đính kèm ảnh học tập.
Nếu người dùng muốn quay lại học, hãy chờ họ nói rõ hoặc hỏi tiếp.
{thread_hint}

Một phần lịch sử gần nhất của boxchat (chỉ để hiểu ngữ cảnh):
{casual_context}

Tin nhắn hiện tại:
{query_text}"""
        print(
            "[CHAT ROUTING] casual conversation: lightweight thread context; "
            "no embedding/RAG/images"
        )
        gen_started = time.perf_counter()
        reply, model_used, _ = _generate_chat_reply(
            minimal_prompt, content_type=None, request_id=request_id, gen_started=gen_started, user_text=query_text
        )
        return {
            "reply": reply,
            "model": model_used,
            "sources": [],
            "images": [],
            "content_blocks": [{"type": "text", "text": reply}],
            "learning_progress": None,
        }

    named_lesson_topic = _explicit_lesson_topic(low, catalog)

    if ambiguous_study_request:
        requested_scope = {"course": None, "content_type": None, "lesson": None, "topic": None}
    elif named_lesson_topic:
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

    if next_lesson_scope:
        requested_scope = next_lesson_scope.copy()

    if forced_plan_scope:
        requested_scope = forced_plan_scope.copy()

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

    # A fully specified exercise lesson is an actual teaching turn once the
    # learner has explicitly confirmed it; before confirmation the hard lesson
    # confirmation gate below takes precedence.
    requested_ct_norm = _normalize_content_type(requested_content_type or "")
    specific_exercise_lesson_request = bool(
        requested_ct_norm == "Bài tập"
        and (requested_lesson or named_lesson_topic)
        and not ambiguous_study_request
        and not _is_correction_followup(query_text)
        and not (data.action and lesson_confirmed_scope)
    )

    lesson_intro_request = bool(
        not ambiguous_study_request
        and not data.action
        and thread_switch_requested
        and (named_lesson_topic or requested_lesson)
        and not specific_exercise_lesson_request
        and not _is_correction_followup(query_text)
    )

    if specific_exercise_lesson_request:
        print("[CHAT ROUTING] specific exercise lesson request: keep lesson images")

    # A request that explicitly chooses only a content type (e.g.
    # "mình muốn học giáo trình") is a ROUTING turn, not a lesson-teaching turn.
    # Do not let RAG/previous progress choose an arbitrary lesson or introduce a
    # different content type. Ask the learner for the lesson/topic next.
    content_type_only_request = bool(
        requested_content_type
        and not requested_lesson
        and not requested_topic
        and not forced_plan_scope
        and not data.action
        and not ambiguous_study_request
        and not _is_correction_followup(query_text)
    )
    if content_type_only_request:
        examples = {
            "Giáo trình": "Ví dụ: 'Bài 3' hoặc 'Bài 3 giáo trình'.",
            "Ngữ pháp": "Ví dụ: 'Bài 3 ngữ pháp'.",
            "Bài tập": "Ví dụ: 'Bài 3 bài tập'.",
            "Từ vựng": "Ví dụ: 'Bộ thủ' hoặc 'Kanji'.",
            "Truyện đọc": "Ví dụ: tên bài/truyện cậu muốn đọc.",
        }
        msg = (
            f"📚 Doraemon đã xác định cậu muốn học **{requested_content_type}**.\n\n"
            f"Cậu muốn học bài/chủ đề nào? {examples.get(requested_content_type, '')}"
        )
        return {
            "reply": msg,
            "model": GEMINI_MODEL,
            "sources": [],
            "images": [],
            "content_blocks": [{"type": "text", "text": msg}],
            "learning_progress": None,
        }

    # ------------------------------------------------------------------
    # PERSISTENT STUDY SESSION GATE
    # ------------------------------------------------------------------
    # Once a lesson has been confirmed, only questions that plausibly belong to
    # that lesson are allowed into embedding/Pinecone/RAG/images. If the learner
    # changes to casual/off-topic conversation, ask once whether to end today's
    # lesson. The session remains ACTIVE until the learner explicitly presses Có.
    # ------------------------------------------------------------------
    active_session_scope = _active_session_scope(study_session)
    if active_session_scope and not lesson_confirmed_scope and not forced_plan_scope and not data.action:
        if _is_off_topic_during_study(query_text) and not _is_study_followup(query_text):
            if not bool(study_session.get("end_prompt_pending")):
                _set_study_end_prompt_pending(user["id"], True)
                print("[CHAT ROUTING] active study session -> off-topic: ask end-of-lesson confirmation")
                blocks = _study_end_choice_blocks(active_session_scope, prefix="Ừ, tớ hiểu. Tin nhắn này không còn thuộc phần bài đang học.")
                return {"reply":blocks[0]["text"],"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":blocks,"learning_progress":None}
            # Avoid repeated end prompts before the learner has answered.
            msg = f"Mình vẫn đang giữ bài **{active_session_scope.get('lesson') or 'này'}** mở nhé. Cậu muốn hỏi tiếp phần đang học hay bấm **Có** để kết thúc bài hôm nay?"
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

    # ------------------------------------------------------------------
    # HARD LESSON INTENT CONFIRMATION
    # Before Doraemon teaches any concrete lesson, confirm the resolved
    # lesson with an inline Có/Không choice. This prevents RAG from
    # silently selecting a neighbouring lesson and prevents images from
    # being attached before the learner confirms the target.
    # A confirmed UI action (lesson_confirm_yes) bypasses this gate.
    # ------------------------------------------------------------------
    lesson_confirmation_scope = lesson_confirmed_scope
    if lesson_confirmation_scope is None and next_lesson_scope:
        lesson_confirmation_scope = {
            "course": str(next_lesson_scope.get("course") or next_lesson_scope.get("course_name") or "").strip() or None,
            "content_type": _normalize_content_type(next_lesson_scope.get("content_type")),
            "lesson": str(next_lesson_scope.get("lesson") or "").strip() or None,
            "topic": str(next_lesson_scope.get("topic") or "").strip() or None,
        }

    # Explicit current-message lesson target should be confirmed before RAG.
    if lesson_confirmation_scope is None and (named_lesson_topic or _is_specific_lesson_request(query_text)) and not forced_plan_scope and not data.action:
        if named_lesson_topic:
            lesson_confirmation_scope = {
                "course": str(named_lesson_topic.get("course") or named_lesson_topic.get("course_name") or "").strip() or None,
                "content_type": _normalize_content_type(named_lesson_topic.get("content_type")),
                "lesson": str(named_lesson_topic.get("lesson") or "").strip() or None,
                "topic": str(named_lesson_topic.get("topic") or "").strip() or None,
            }
        else:
            # The user clearly asked for a specific lesson, but catalog routing
            # could not resolve it. Never fall into generic RAG; suggest nearby
            # known lessons instead.
            requested_ct_hint = _select_active_scope(low, [], catalog).get("content_type")
            suggestions = _lesson_suggestions(catalog, requested_ct_hint, limit=5)
            if suggestions:
                lines = ["🤖 Doraemon chưa tìm thấy đúng bài cậu muốn học.", "\nCậu có thể thử một trong các bài sau:"]
                for ct, lesson, topic, label in suggestions:
                    lines.append(f"• {label} ({ct})")
                lines.append("\nCậu nói lại tên bài cụ thể, Doraemon sẽ xác nhận trước khi bắt đầu học nhé.")
                msg = "\n".join(lines)
            else:
                msg = "🤖 Doraemon chưa tìm thấy đúng bài cậu muốn học trong kho tài liệu hiện tại. Cậu nói tên bài khác để Doraemon kiểm tra nhé."
            return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":[{"type":"text","text":msg}],"learning_progress":None}

    if lesson_confirmation_scope is not None and not forced_plan_scope and lesson_confirmed_scope is None and not data.action:
        lesson_label = lesson_confirmation_scope.get("lesson") or "bài này"
        ct_label = lesson_confirmation_scope.get("content_type") or "nội dung"
        topic_label = lesson_confirmation_scope.get("topic")
        detail = f" ({topic_label})" if topic_label else ""
        msg = (
            f"🎯 Doraemon hiểu là cậu muốn học **{lesson_label}**{detail} thuộc **{ct_label}**.\n\n"
            f"Có phải cậu muốn học bài này không? 😊"
        )
        token = _encode_lesson_confirm_scope(lesson_confirmation_scope)
        blocks = [{"type":"text","text":msg},{"type":"choice","id":"lesson_confirm",
                  "options":[
                      {"label":"Có","display_label":f"Có — {lesson_label}","action":f"lesson_confirm_yes:{token}"},
                      {"label":"Không","display_label":f"Không — {lesson_label}","action":f"lesson_confirm_no:{token}"}
                  ]}]
        return {"reply":msg,"model":GEMINI_MODEL,"sources":[],"images":[],"content_blocks":blocks,"learning_progress":None}

    # If there is no active confirmed study session and the current turn is not
    # opening/confirming a lesson, answer as lightweight conversation. This is the
    # second cost barrier: non-study chat never reaches the large teacher prompt,
    # Pinecone, embedding, or image stack.
    if not active_session_scope and not lesson_confirmed_scope and not forced_plan_scope and not data.action:
        if not (named_lesson_topic or _is_specific_lesson_request(query_text) or requested_content_type):
            light_history = recent_history[-2:]
            light_context = "\n".join(
                f"{h.get('role')}: {str(h.get('text') or '')[-350:]}" for h in light_history
            )
            minimal_prompt = f"""Bạn là Doraemon, một người bạn/gia sư thân thiện.
Đây là cuộc trò chuyện chưa mở bài học. Trả lời trực tiếp, tự nhiên và ngắn gọn. Quy tắc ngôn ngữ toàn cục ở đầu prompt quyết định ngôn ngữ trả lời.
Không tự mở bài học, không dùng RAG/Pinecone, không đính kèm ảnh học tập.
Nếu người dùng muốn học một bài cụ thể, hãy yêu cầu họ nêu tên bài để Doraemon xác nhận Có/Không trước khi bắt đầu.

Lịch sử rất ngắn của boxchat (chỉ để hiểu đại từ nếu cần):
{light_context}

Tin nhắn hiện tại:
{query_text}"""
            print("[CHAT ROUTING] no active study session: lightweight chat; no embedding/Pinecone/RAG/images")
            gen_started = time.perf_counter()
            reply, model_used, _ = _generate_chat_reply(
                minimal_prompt, content_type=None, request_id=request_id, gen_started=gen_started, user_text=query_text
            )
            return {"reply":reply,"model":model_used,"sources":[],"images":[],"content_blocks":[{"type":"text","text":reply}],"learning_progress":None}

    # Continue the most recent in-progress lesson for short follow-ups. This
    # applies to ALL content types (especially exercises), not only Kanji/Bộ thủ.
    # PostgreSQL is the durable state; chat history is only the conversational hint.
    recommendation_words = (
        "học gì", "học gì hôm nay", "hôm nay học gì",
        "gợi ý", "đề xuất", "chọn bài", "nên học",
        "có gì để dạy", "có gì để học", "dạy gì", "học được gì"
    )
    wants_recommendation = any(w in low for w in recommendation_words) or ambiguous_study_request

    # Recommendation / "what can you teach?" is ALWAYS a routing turn.
    # A currently-open lesson/chat thread must NOT override this intent.
    # Only disable the routing path when the user's current message itself
    # contains a concrete lesson/content target, an action, a plan scope, or is
    # clearly a correction/follow-up.
    current_message_has_concrete_target = bool(
        named_lesson_topic
        or re.search(r"\b(?:bài|lesson|phần)\s*[\wÀ-ỹà-ỹ0-9_-]+", low, flags=re.UNICODE)
        or any(x in low for x in (
            "giáo trình", "ngữ pháp", "bài tập", "từ vựng", "truyện đọc",
            "kanji", "bộ thủ", "theo lộ trình", "lộ trình",
        ))
    )
    recommendation_only_request = bool(
        wants_recommendation
        and not current_message_has_concrete_target
        and not forced_plan_scope
        and not data.action
        and not _is_correction_followup(query_text)
    )

    if recommendation_only_request:
        # Hard reset any inherited thread scope BEFORE RAG routing. This is the
        # critical guard that prevents a prior lesson (e.g. Giáo trình alphav1)
        # from supplying text/images to a generic "cậu có gì để dạy?" request.
        requested_scope = {"course": None, "content_type": None, "lesson": None, "topic": None}
        requested_content_type = None
        requested_course = None
        requested_lesson = None
        requested_topic = None
        thread_scope_locked = False
        active_scope = None
        print("[CHAT ROUTING] recommendation-only request: ignore current thread/active lesson")
    if recommendation_only_request and not ambiguous_study_request:
        msg = (
            "📚 Doraemon có thể đồng hành cùng cậu ở 5 loại nội dung:\n\n"
            "1. **Giáo trình** – học bài theo đúng giáo trình, giải thích từng phần.\n"
            "2. **Ngữ pháp** – học các mẫu câu và điểm ngữ pháp.\n"
            "3. **Bài tập** – làm bài, Doraemon ra đề, gợi ý, chấm và giải chi tiết.\n"
            "4. **Từ vựng** – học từ vựng theo chủ đề, bao gồm **Kanji** và **Bộ thủ**.\n"
            "5. **Truyện đọc** – luyện đọc hiểu qua các bài/truyện tiếng Nhật.\n\n"
            "Cậu muốn học loại nào? Có thể nói luôn tên bài, ví dụ: **Bài 3 giáo trình** hoặc **Bài 3 bài tập** nhé. 😊"
        )
        print("[CHAT ROUTING] recommendation-only request: no RAG/images")
        return {
            "reply": msg,
            "model": GEMINI_MODEL,
            "sources": [],
            "images": [],
            "content_blocks": [{"type": "text", "text": msg}],
            "learning_progress": None,
        }

    active_learning = None
    if not recommendation_only_request and not (requested_content_type or requested_course or requested_lesson or requested_topic) and not wants_recommendation:
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

    # A correction is a continuation of the previous lesson, not a new lesson
    # request. Prefer durable active state over accidental keyword matches in
    # the correction sentence (e.g. "Bài 1" appearing in an old reply).
    correction_followup = _is_correction_followup(query_text) and bool(recent_history)

    # Current-thread context wins unless the student explicitly asks to switch
    # to another lesson/section. A correction always wins over any accidental
    # lesson keyword in the current sentence.
    thread_scope_locked = bool(
        thread_scope
        and not thread_switch_requested
        and not named_lesson_topic
        and not ambiguous_study_request
        and not recommendation_only_request
    )
    if correction_followup and thread_scope:
        thread_scope_locked = True
    if thread_scope_locked:
        requested_scope = dict(thread_scope)
        requested_content_type = requested_scope.get("content_type")
        requested_course = requested_scope.get("course")
        requested_lesson = requested_scope.get("lesson")
        requested_lesson_canonical = _canonical_lesson_key(requested_lesson)
        requested_topic = requested_scope.get("topic")

    if correction_followup and active_learning and not thread_scope_locked:
        requested_content_type = _normalize_content_type(active_learning.get("content_type")) or requested_content_type
        requested_course = str(active_learning.get("subject") or "").strip() or requested_course
        requested_lesson = str(active_learning.get("lesson") or "").strip() or requested_lesson
        requested_topic = str(active_learning.get("topic") or "").strip() or requested_topic
        requested_scope = {
            "course": requested_course or None,
            "content_type": requested_content_type or None,
            "lesson": requested_lesson or None,
            "topic": requested_topic or None,
        }

    if active_session_scope and not lesson_confirmed_scope and not forced_plan_scope:
        requested_scope = dict(active_session_scope)
        requested_content_type = requested_scope.get("content_type")
        requested_course = requested_scope.get("course")
        requested_lesson = requested_scope.get("lesson")
        requested_topic = requested_scope.get("topic")
        thread_scope_locked = False

    if lesson_confirmed_scope:
        requested_scope = dict(lesson_confirmed_scope)
        requested_content_type = requested_scope.get("content_type")
        requested_course = requested_scope.get("course")
        requested_lesson = requested_scope.get("lesson")
        requested_topic = requested_scope.get("topic")
        thread_scope_locked = False

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
    # NOTE: end_prompt_pending is handled only AFTER the hard study gate is
    # computed. Never reference study_retrieval_allowed before that point.
    rag_query_text = query_text
    if thread_scope_locked and recent_history:
        scope_parts = [
            str(thread_scope.get("content_type") or ""),
            str(thread_scope.get("course") or ""),
            str(thread_scope.get("lesson") or ""),
            str(thread_scope.get("topic") or ""),
        ]
        scope_label = " / ".join(x for x in scope_parts if x)
        rag_query_text = (
            "NGỮ CẢNH CHÍNH CỦA BOXCHAT ĐANG MỞ. Ưu tiên đúng luồng hội thoại này; "
            "không chuyển sang lesson/content type khác trừ khi học sinh yêu cầu rõ ràng.\n"
            f"Phạm vi hiện tại: {scope_label}\n"
            f"Lịch sử gần nhất của luồng (tối đa 10 lượt):\n{thread_history_text}\n"
            f"Tin nhắn hiện tại: {query_text}"
        )
    elif correction_followup and recent_history:
        history_tail = recent_history[-6:]
        history_context = "\n".join(f"{h['role']}: {h['text'][-1200:]}" for h in history_tail)
        rag_query_text = (
            "ĐÂY LÀ PHẢN HỒI/SỬA LẠI CÂU TRẢ LỜI TRƯỚC. Giữ nguyên bài/lesson đang học; "
            "không chuyển sang bài khác.\n" + history_context + "\n" +
            "Tin nhắn sửa của học sinh: " + query_text
        )
    elif active_scope:
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
    
    # ================================================================
    # HARD STUDY GATE v2
    # Compute the gate variables BEFORE using them.  The previous build used
    # study_retrieval_allowed before _study_confirmation was initialized,
    # causing UnboundLocalError on ordinary chat messages.
    # IMPORTANT:
    # Having active_learning / thread_scope / history is NOT confirmation
    # to use study retrieval. Only an explicit confirmed lesson may open
    # embedding -> Pinecone -> RAG -> images.
    #
    # We intentionally gate BEFORE embed_text(), so casual messages cannot
    # incur embedding cost even when PostgreSQL has an active lesson.
    # ================================================================
    active_session_scope = _active_session_scope(study_session)
    _study_confirmation = bool(lesson_confirmed_scope or forced_plan_scope or active_session_scope)
    _has_explicit_study_scope = bool(
        active_session_scope
        and requested_lesson
        and _clean_scope_value(requested_lesson) == _clean_scope_value(active_session_scope.get("lesson"))
        and (
            not active_session_scope.get("topic")
            or not requested_topic
            or _clean_scope_value(requested_topic) == _clean_scope_value(active_session_scope.get("topic"))
        )
    )
    # A newly confirmed/start-plan turn already has an exact scope. Existing
    # sessions may answer short follow-ups without repeating the lesson name.
    if lesson_confirmed_scope or forced_plan_scope:
        _has_explicit_study_scope = True
    study_retrieval_allowed = bool(_study_confirmation and _has_explicit_study_scope)

    # Clear the one-shot end prompt only after the request is confirmed as a
    # valid active-study turn. This ordering prevents an UnboundLocalError and
    # also ensures pending end-of-lesson state can never accidentally run during
    # a non-study request.
    if study_retrieval_allowed and study_session and study_session.get("end_prompt_pending"):
        _set_study_end_prompt_pending(user["id"], False)
        study_session["end_prompt_pending"] = False

    runtime_lesson_cache = None
    runtime_cache_hit = False
    runtime_cache_initial = bool(lesson_confirmed_scope or forced_plan_scope)
    if study_retrieval_allowed and requested_lesson:
        runtime_lesson_cache = _load_runtime_lesson_cache(
            requested_content_type or "Giáo trình", requested_lesson, requested_topic, request_id=request_id
        )
        runtime_cache_hit = bool(runtime_lesson_cache)
        if not runtime_cache_hit:
            print(f"[KNOWLEDGE CACHE RUNTIME MISS] request={request_id} lesson={requested_lesson!r} topic={requested_topic!r} reason='no_ready_payload'" )
        if runtime_cache_hit:
            print(f"[KNOWLEDGE CACHE RUNTIME HIT] lesson={requested_lesson!r} topic={requested_topic!r} content_type={requested_content_type!r}")
    print(
        f"[KNOWLEDGE/RAG AUDIT] request={request_id} "
        f"cache_hit={int(runtime_cache_hit)} study_allowed={int(study_retrieval_allowed)} "
        f"embedding_will_run={int(study_retrieval_allowed and not runtime_cache_hit)} "
        f"lesson={requested_lesson!r} topic={requested_topic!r} content_type={requested_content_type!r}"
    )

    # DB-FIRST path for published non-Giáo-trình curriculum.
    # Từ vựng / Ngữ pháp / Bài tập / Truyện đọc all use the same deterministic
    # lesson-step navigation. Gemini is reserved for genuine learner questions,
    # and those turns receive only one prior chat exchange as context.
    if (runtime_cache_hit and (runtime_lesson_cache or {}).get("published_curriculum")
            and requested_content_type in {"Từ vựng", "Bài tập", "Ngữ pháp", "Truyện đọc"} and study_session):
        sections=list((runtime_lesson_cache or {}).get("sections") or [])
        if sections:
            current_step=max(0,min(int((study_session or {}).get("curriculum_step") or 0),len(sections)-1))
            waiting=str((study_session or {}).get("curriculum_waiting") or "continue")
            answered=bool((study_session or {}).get("curriculum_exercise_answered"))

            # Button navigation is deterministic and costs 0 Gemini/embedding/Pinecone.
            if ui_action == "curriculum_next":
                try:
                    expected=int(action_plan_id or -1)
                except Exception:
                    expected=-1
                if expected == current_step and current_step < len(sections)-1:
                    current_step += 1
                    answered=False
                    waiting="continue"
                    _set_curriculum_flow(user["id"],step=current_step,waiting=waiting,exercise_answered=False)
                    study_session["curriculum_step"]=current_step
                    study_session["curriculum_waiting"]=waiting
                    study_session["curriculum_exercise_answered"]=False
                    print(f"[CURRICULUM DB-FIRST FLOW] request={request_id} type={requested_content_type} advance={current_step}")

            # Text 'tiếp' is also a pure DB navigation turn.
            elif not data.action and _is_continue_confirmation(query_text) and current_step < len(sections)-1:
                current_step += 1
                answered=False
                waiting="continue"
                _set_curriculum_flow(user["id"],step=current_step,waiting=waiting,exercise_answered=False)
                study_session["curriculum_step"]=current_step
                study_session["curriculum_waiting"]=waiting
                study_session["curriculum_exercise_answered"]=False
                print(f"[CURRICULUM DB-FIRST FLOW] request={request_id} type={requested_content_type} text_advance={current_step}")

            step=_published_curriculum_step(runtime_lesson_cache,current_step)

            # Deterministic simple/casual exercise turns must not invoke GenAI.
            if requested_content_type == "Bài tập" and not data.action and str(query_text or "").strip():
                direct_ex = _exercise_simple_direct_answer(query_text.strip(), step, runtime_lesson_cache, current_step)
                if direct_ex:
                    mode, msg = direct_ex
                    blocks=[{"type":"text","text":msg}]
                    if mode == "answer_db":
                        _set_curriculum_flow(user["id"], step=current_step, waiting="continue", exercise_answered=True)
                        study_session["curriculum_exercise_answered"]=True
                        blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
                        blocks.extend(_curriculum_continue_blocks(current_step))
                    elif not step.get("is_final"):
                        blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
                        blocks.extend(_curriculum_continue_blocks(current_step))
                    print(f"[CURRICULUM DB QUESTION DIRECT] request={request_id} type=Bài tập mode={mode} genai=0 embedding=0 pinecone=0")
                    return {"reply":"\n\n".join(str(b.get("text") or "") for b in blocks if b.get("type")=="text"),"model":"db-direct","sources":[],"images":[],"content_blocks":blocks,"learning_progress":None}

            # Exercise submission uses GenAI for evaluation/explanation, but the
            # official answer shown to the learner always comes verbatim from DB.
            if requested_content_type == "Bài tập" and waiting == "exercise_answer" and not data.action and str(query_text or "").strip():
                answer_step=_published_curriculum_answer_step(runtime_lesson_cache)
                official_answer=str((answer_step or {}).get("text") or "").strip()
                one_exchange=_last_chat_exchange(recent_history)
                one_exchange_text="\n".join(f"{h['role']}: {h['text'][-900:]}" for h in one_exchange)
                q_prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật. Đây là một lượt hỏi đáp ngắn trong bài tập.
Chỉ dùng MỘT lượt hội thoại ngay trước đó làm ngữ cảnh hội thoại; không cần toàn bộ lịch sử chat.

LƯỢT HỘI THOẠI TRƯỚC:
{one_exchange_text}

TIN NHẮN HIỆN TẠI / CÂU TRẢ LỜI CỦA HỌC SINH:
{query_text.strip()}

ĐÁP ÁN CHÍNH THỨC TRONG DB:
{official_answer}

Hãy đánh giá ngắn gọn đúng/sai hoặc mức độ phù hợp, chỉ ra lỗi và giải thích cách sửa. Không được thay đổi đáp án chính thức."""
                print(f"[CURRICULUM DB QUESTION] request={request_id} type=Bài tập mode=evaluate context=1_exchange prompt_chars={len(q_prompt)} embedding=0 pinecone=0")
                gen_started=time.perf_counter()
                evaluation,response_model,gen_elapsed=_generate_chat_reply(q_prompt,content_type=requested_content_type,request_id=request_id,gen_started=gen_started,user_text=query_text.strip())
                answered=True
                waiting="continue"
                _set_curriculum_flow(user["id"],step=current_step,waiting=waiting,exercise_answered=True)
                study_session["curriculum_exercise_answered"]=True
                blocks=[{"type":"text","text":evaluation or ""}]
                if official_answer:
                    blocks.append({"type":"text","text":"📘 **Đáp án chính thức trong DB:**\n\n"+official_answer})
                blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
                blocks.extend(_curriculum_continue_blocks(current_step))
                print(f"[CURRICULUM DB-FIRST ANSWER] request={request_id} answer_source=curriculum_steps.content_json genai=1")
                return {"reply":"\n\n".join(str(b.get("text") or "") for b in blocks if b.get("type")=="text"),"model":response_model,"sources":[],"images":[{"key":b.get("key"),"url":b.get("url")} for b in blocks if b.get("type")=="image"],"content_blocks":blocks,"learning_progress":None}

            # Cheap DB-only factual vocabulary questions must never spend Gemini
            # tokens. Grammar questions and other non-trivial questions use the separate
            # GenAI teacher turn with exactly one prior exchange as context.
            if not data.action and str(query_text or "").strip() and requested_content_type == "Từ vựng":
                direct_answer = _vocab_direct_answer_from_cache(runtime_lesson_cache, current_step, query_text.strip())
                if direct_answer:
                    blocks=[{"type":"text","text":direct_answer}]
                    if not step.get("is_final"):
                        blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
                        blocks.extend(_curriculum_continue_blocks(current_step))
                    print(f"[CURRICULUM DB QUESTION DIRECT] request={request_id} type=Từ vựng mode=direct_db genai=0 embedding=0 pinecone=0")
                    return {"reply":direct_answer,"model":"db-direct","sources":[],"images":[],"content_blocks":blocks,"learning_progress":None}

            # Any other ordinary learner question in the active lesson is a
            # separate GenAI teacher turn, grounded by the current DB step.
            if not data.action and str(query_text or "").strip():
                question_text=query_text.strip()
                one_exchange=_last_chat_exchange(recent_history)
                one_exchange_text="\n".join(f"{h['role']}: {h['text'][-900:]}" for h in one_exchange)
                q_prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật. Đây là một câu hỏi tiếp nối trong bài {requested_content_type}.
Chỉ dùng MỘT lượt hội thoại ngay trước đó làm ngữ cảnh hội thoại. Không gửi/không cần toàn bộ lịch sử chat hay toàn bộ lesson context.

LƯỢT HỘI THOẠI TRƯỚC:
{one_exchange_text}

CÂU HỎI HIỆN TẠI:
{question_text}

Trả lời ngắn gọn, đúng trọng tâm. Nếu câu hỏi liên quan đến từ vựng, giữ đúng chữ Nhật, cách đọc/phát âm và nghĩa đã xuất hiện trong lượt trước; không tự bịa. Nếu lượt trước không đủ dữ kiện thì nói rõ điều đó."""
                print(f"[CURRICULUM DB QUESTION] request={request_id} type={requested_content_type} step={step.get('code')} context=1_exchange prompt_chars={len(q_prompt)} embedding=0 pinecone=0")
                gen_started=time.perf_counter()
                answer,response_model,gen_elapsed=_generate_chat_reply(q_prompt,content_type=requested_content_type,request_id=request_id,gen_started=gen_started,user_text=question_text)
                blocks=[{"type":"text","text":answer or ""}]
                for im in step.get("images") or []:
                    if im.get("url"): blocks.append({"type":"image","key":im.get("key"),"url":im.get("url"),"page":im.get("page"),"caption":im.get("caption","")})
                if not step.get("is_final"):
                    blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
                    blocks.extend(_curriculum_continue_blocks(current_step))
                print(f"[CURRICULUM DB QUESTION] request={request_id} type={requested_content_type} context=1_exchange genai=1 embedding=0 pinecone=0")
                return {"reply":answer or "","model":response_model,"sources":[],"images":[{"key":b.get("key"),"url":b.get("url")} for b in blocks if b.get("type")=="image"],"content_blocks":blocks,"learning_progress":None}

            step=_published_curriculum_step(runtime_lesson_cache,current_step)
            if requested_content_type == "Bài tập" and answered:
                blocks=_published_curriculum_non_giao_trinh_blocks(step,runtime_lesson_cache,requested_content_type,answered=True)
            else:
                blocks=_published_curriculum_non_giao_trinh_blocks(step,runtime_lesson_cache,requested_content_type,answered=False)

            # A question step waits for an answer but still exposes Tiếp theo, as requested.
            if requested_content_type == "Bài tập" and str(step.get("code") or "").upper() == "B0" and not answered:
                _set_curriculum_flow(user["id"],step=current_step,waiting="exercise_answer",exercise_answered=False)
                study_session["curriculum_waiting"]="exercise_answer"
            elif not step.get("is_final") and waiting != "continue":
                _set_curriculum_flow(user["id"],step=current_step,waiting="continue",exercise_answered=answered)

            print(f"[CURRICULUM DB-FIRST] request={request_id} type={requested_content_type} step={step.get('code')} genai=0 embedding=0 pinecone=0")
            return {"reply":"\n\n".join(str(b.get("text") or "") for b in blocks if b.get("type")=="text"),"model":GEMINI_MODEL,"sources":[],"images":[{"key":b.get("key"),"url":b.get("url")} for b in blocks if b.get("type")=="image"],"content_blocks":blocks,"learning_progress":None}

    # DB-first path for newly published AI Curriculum. A navigation/Continue turn
    # renders the already-published step directly from PostgreSQL: no embedding,
    # no Pinecone and no Gemini. A real learner question uses Gemini with the current
    # DB step as context, still without embedding/Pinecone.
    if runtime_cache_hit and (runtime_lesson_cache or {}).get("published_curriculum") and requested_content_type == "Giáo trình" and study_session:
        current_step=int((study_session or {}).get("curriculum_step") or 0)
        if current_step < 0: current_step=0
        step=_published_curriculum_step(runtime_lesson_cache,current_step)
        if _published_curriculum_db_only_turn(query_text,ui_action,study_session):
            blocks=_published_curriculum_blocks(step,runtime_lesson_cache)
            _set_curriculum_flow(user["id"],step=int(step["index"]),waiting="final" if step.get("is_final") else "continue",exercise_answered=False)
            print(f"[CURRICULUM DB-FIRST] request={request_id} step={step.get('code')} genai=0 embedding=0 pinecone=0")
            return {"reply":"\n\n".join(str(b.get("text") or "") for b in blocks if b.get("type")=="text"),"model":GEMINI_MODEL,"sources":[],"images":[{"key":b.get("key"),"url":b.get("url")} for b in blocks if b.get("type")=="image"],"content_blocks":blocks,"learning_progress":None}

        question_text=query_text.strip()
        q_prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật. Chỉ dùng dữ liệu PUBLISHED trong DB bên dưới để trả lời câu hỏi. Nếu dữ liệu không đủ, nói rõ là chưa có thông tin trong bài học; không bịa.

BÀI: {requested_lesson}
BƯỚC: {step.get('code')} - {step.get('title')}

NỘI DUNG BƯỚC:
{step.get('text','')}

CÂU HỎI:
{question_text}"""
        print(f"[CURRICULUM DB QUESTION] request={request_id} step={step.get('code')} prompt_chars={len(q_prompt)} embedding=0 pinecone=0")
        gen_started=time.perf_counter()
        answer,response_model,gen_elapsed=_generate_chat_reply(q_prompt,content_type=requested_content_type,request_id=request_id,gen_started=gen_started,user_text=question_text)
        blocks=[{"type":"text","text":answer or ""}]
        for im in step.get("images") or []:
            if im.get("url"): blocks.append({"type":"image","key":im.get("key"),"url":im.get("url"),"page":im.get("page"),"caption":im.get("caption","")})
        if step.get("is_final"): blocks.extend(_curriculum_final_blocks())
        else:
            blocks.append({"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"})
            blocks.extend(_curriculum_continue_blocks(step["index"]))
        return {"reply":answer or "","model":response_model,"sources":[],"images":[{"key":b.get("key"),"url":b.get("url")} for b in blocks if b.get("type")=="image"],"content_blocks":blocks,"learning_progress":None}

    curriculum_flow_active = bool(
        runtime_cache_hit and requested_content_type == "Giáo trình" and study_session
        and not bool((runtime_lesson_cache or {}).get("published_curriculum"))
    )
    # Refresh durable curriculum state once more after the runtime cache lookup so
    # questions captured at B0/B1 are visible before deciding whether a global
    # exercise checkpoint is actually needed.
    if curriculum_flow_active:
        refreshed_session = _get_study_session(user["id"], chatbox_id=getattr(data, "chatbox_id", None))
        if refreshed_session:
            study_session = refreshed_session
    curriculum_map = _curriculum_step_map(runtime_lesson_cache) if curriculum_flow_active else None
    curriculum_step = int((study_session or {}).get("curriculum_step") or 0) if curriculum_flow_active else None
    curriculum_waiting = str((study_session or {}).get("curriculum_waiting") or "continue") if curriculum_flow_active else None
    curriculum_exercise_answered = bool((study_session or {}).get("curriculum_exercise_answered")) if curriculum_flow_active else False

    # Do not expose a visible no-op checkpoint when the lesson has no saved
    # whole-lesson exercise. After the last teaching chunk, jump directly to the
    # final summary in the same Continue request. This keeps the UI one-click-per
    # meaningful-step and avoids an extra low-value Gemini detection turn.
    if (
        curriculum_flow_active
        and curriculum_step == curriculum_map["global_exercise_step"]
        and not str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
    ):
        next_step = curriculum_map["summary_step"]
        _set_curriculum_flow(user["id"], step=next_step, waiting="final", exercise_answered=False)
        study_session["curriculum_step"] = next_step
        study_session["curriculum_waiting"] = "final"
        study_session["curriculum_exercise_answered"] = False
        curriculum_step = next_step
        curriculum_waiting = "final"
        curriculum_exercise_answered = False
        print(
            f"[CURRICULUM FLOW] skip_empty_global_exercise_step "
            f"global_step={curriculum_map['global_exercise_step']} -> summary_step={next_step}"
        )

    # In the curriculum flow, the Continue button is always available—even when
    # a chunk/global exercise is still unanswered. Pressing Continue advances one
    # state without embedding/RAG changes, intentionally allowing the learner to skip.
    # Text confirmations are supported the same way.
    if curriculum_flow_active and curriculum_waiting == "continue" and not data.action and _is_continue_confirmation(query_text):
        next_step=curriculum_step+1
        waiting_state = "continue"
        _set_curriculum_flow(user["id"], step=next_step, waiting=waiting_state, exercise_answered=False)
        study_session["curriculum_step"]=next_step
        study_session["curriculum_waiting"]=waiting_state
        curriculum_step=next_step
        curriculum_waiting=waiting_state
        curriculum_exercise_answered=False
        print(f"[CURRICULUM FLOW] text-confirm advance to step={next_step} waiting={waiting_state!r}")

    if curriculum_flow_active and curriculum_waiting in {"continue_after_global_exercise", "continue_after_global_check"} and not data.action and _is_continue_confirmation(query_text):
        next_step = curriculum_map["summary_step"]
        _set_curriculum_flow(user["id"], step=next_step, waiting="final", exercise_answered=True)
        study_session["curriculum_step"] = next_step
        study_session["curriculum_waiting"] = "final"
        curriculum_step = next_step
        curriculum_waiting = "final"
        curriculum_exercise_answered = True
        print(f"[CURRICULUM FLOW] text-confirm advance to summary step={next_step}")

    # When a section/exercise is waiting for review, ordinary questions remain inside
    # the same lesson and can use the relevant cache context.

    if not study_retrieval_allowed:
        print(
            "[CHAT ROUTING] study hard-gate: NOT CONFIRMED -> "
            "no embedding/Pinecone/RAG/images"
        )
        query_vector = None
        result = type("_EmptyResult", (), {"matches": []})()
        rich_images = []
        perf_embed = time.perf_counter()
    elif runtime_cache_hit:
        print("[RAG BYPASS] upload-time knowledge cache is authoritative; skip embedding/Pinecone")
        query_vector = None
        result = type("_EmptyResult", (), {"matches": []})()
        rich_images = []
        perf_embed = time.perf_counter()
    else:
        query_vector = embed_text(rag_query_text)
        perf_embed = time.perf_counter()

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
        if query_vector is None:
            return type('_EmptyResult', (), {'matches': []})()
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
        # Lesson is a hard retrieval boundary once explicitly identified.
        # Never relax to content_type-only because that could surface the next
        # lesson (e.g. Bài 4) when the user asked for Bài 3.
        add_priority_filter(requested_content_type, None, requested_lesson, None)
        add_priority_filter(None, None, requested_lesson, None)
        if requested_topic:
            add_priority_filter(requested_content_type, None, requested_lesson, requested_topic)
            add_priority_filter(None, None, requested_lesson, requested_topic)
    elif requested_topic:
        add_priority_filter(requested_content_type, None, None, requested_topic)
        add_priority_filter(None, None, None, requested_topic)
        if requested_content_type:
            add_priority_filter(requested_content_type, None, None, None)
    elif requested_content_type:
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
                legacy_candidate = query_text_matches(pf) if query_vector is None else index.query(
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

    # Last semantic fallback is allowed only when no lesson/topic is known.
    # Once the learner has explicitly selected a lesson, NEVER fall back to the
    # whole content type: that is how Bài 4 can leak into a Bài 3 lesson.
    if result is None and not requested_lesson and not requested_topic:
        result = query_text_matches(
            build_scope_filter("text", requested_content_type, None, None, None)
            if requested_content_type else None
        )
        print(
            "[RAG semantic-fallback] "
            f"content_type={requested_content_type!r} chunks={len(_usable_matches(result.matches))}"
        )

    if result is None and (requested_lesson or requested_topic):
        print(
            "[RAG scoped-miss] no text found inside requested lesson/topic; "
            f"lesson={requested_lesson!r} topic={requested_topic!r} "
            "no cross-lesson fallback"
        )
        # Preserve the result shape expected by the downstream pipeline.
        class _EmptyResult:
            matches = []
        result = _EmptyResult()

    # HARD NO-CONTENT FAST PATH: if a confirmed lesson has neither a usable
    # Knowledge Cache nor any scoped RAG text, never spend another LLM call just
    # to say that content is unavailable. Return a deterministic response instead.
    if study_retrieval_allowed and (requested_lesson or requested_topic) and not runtime_cache_hit and not _usable_matches(result.matches):
        msg = (
            f"Tớ chưa tìm thấy nội dung chi tiết của bài **{requested_lesson or requested_topic}** "
            "trong thư viện tài liệu hiện tại. Cậu kiểm tra lại tên bài hoặc tải lại tài liệu của bài này nhé! 🤖"
        )
        print(
            f"[NO-CONTENT FAST-PATH] request={request_id} lesson={requested_lesson!r} "
            f"topic={requested_topic!r} llm=0 embedding={int(query_vector is not None)}"
        )
        return {
            "reply": msg,
            "model": GEMINI_MODEL,
            "sources": [],
            "images": [],
            "content_blocks": [{"type": "text", "text": msg}],
            "learning_progress": None,
        }

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

        if query_vector is not None:
            try:
                compat_result = index.query(
                    vector=query_vector,
                    top_k=retrieval_k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=compat_filter or None,
                )
                compat_usable = _usable_text_matches(compat_result.matches)

                # If the old data uses `subject` instead of `course`, retry once.
                if not compat_usable and requested_course:
                    compat_filter2 = {k: v for k, v in compat_filter.items() if k != "course"}
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
    if ambiguous_study_request or recommendation_only_request:
        # Never let semantic RAG ranking decide what the learner wants to study.
        # The top hit is often Grammar 1 and is not an intent signal.
        result.matches = []
        active_scope = None
        if recommendation_only_request:
            print("[CHAT ROUTING] recommendation-only request: no lesson/content scope inferred")
        else:
            print("[CHAT ROUTING] ambiguous study request: no lesson/content scope inferred")
    elif not explicit_scope and not thread_scope_locked:
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
    cache_selected_sections = []
    if runtime_cache_hit:
        if curriculum_flow_active:
            secs=curriculum_map["sections"]
            if curriculum_step == 0:
                cache_selected_sections=[]
            elif 1 <= curriculum_step <= len(secs):
                # Exactly one source chunk per teaching step.
                cache_selected_sections=[secs[curriculum_step-1]]
            elif curriculum_step in {curriculum_map["global_exercise_step"], curriculum_map["summary_step"]}:
                cache_selected_sections=list(secs)
            else:
                cache_selected_sections=[]
            print(f"[CURRICULUM FLOW] lesson={runtime_lesson_cache.get('lesson')!r} step={curriculum_step} exact_chunk={cache_selected_sections[0].get('chunk_index') if len(cache_selected_sections)==1 else None} sections_for_prompt={len(cache_selected_sections)}")
        else:
            cache_selected_sections = _select_runtime_cache_sections(
                runtime_lesson_cache, query_text, max_sections=2, initial=runtime_cache_initial
            )
        for sec in cache_selected_sections:
            md = {
                "record_type": "text",
                "text": str(sec.get("text") or ""),
                "course": runtime_lesson_cache.get("subject"),
                "subject": runtime_lesson_cache.get("subject"),
                "content_type": runtime_lesson_cache.get("content_type"),
                "source_file": runtime_lesson_cache.get("source_file"),
                "page": sec.get("page"),
                "chunk_index": sec.get("chunk_index"),
                "content_unit_id": sec.get("content_unit_id"),
                "lesson": runtime_lesson_cache.get("lesson"),
                "topic": runtime_lesson_cache.get("topic"),
                "image_keys": json.dumps(sec.get("image_keys") or [], ensure_ascii=False),
            }
            text_chunks.append({"text": md["text"], "metadata": md})
        if curriculum_flow_active and 1 <= curriculum_step <= len(curriculum_map["sections"]):
            # One teaching step = exactly one chunk and only that chunk's images.
            rich_images = _curriculum_chunk_images(runtime_lesson_cache, cache_selected_sections[0])
        elif curriculum_flow_active and (curriculum_step == 0 or curriculum_step in {curriculum_map["global_exercise_step"], curriculum_map["summary_step"]}):
            # Global exercise and summary use cached vision facts as text; images are
            # not automatically dumped into the UI.
            rich_images=[]
        else:
            rich_images = _runtime_cache_images(runtime_lesson_cache, cache_selected_sections)
        print(f"[KNOWLEDGE CACHE CONTEXT] selected_sections={len(cache_selected_sections)} images_ui={len(rich_images)}")
    if not study_retrieval_allowed or runtime_cache_hit:
        result.matches = []
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
    # IMPORTANT: an ambiguous "I want to study" request is a routing/clarification
    # turn, not a lesson turn. Never attach images from the active lesson, RAG
    # ranking, or previous chat while Doraemon is asking the learner to choose
    # a study direction. This is intentionally enforced here as a hard guard,
    # immediately before image retrieval, so future retrieval changes cannot
    # accidentally re-introduce unrelated images into the clarification turn.
    if runtime_cache_hit:
        vision_fact_chars = 0
        if curriculum_flow_active and 1 <= curriculum_step <= len(curriculum_map["sections"]):
            exact_imgs = _curriculum_chunk_images(runtime_lesson_cache, curriculum_map["sections"][curriculum_step-1])
            for _img in exact_imgs:
                # Re-find the cached image payload by key so the audit reflects
                # exactly what this chunk can use.
                for _cached_img in (runtime_lesson_cache.get("images") or []):
                    if str(_cached_img.get("image_key") or "").strip() == str(_img.get("key") or "").strip():
                        vision_fact_chars += len(json.dumps(_cached_img.get("vision") or {}, ensure_ascii=False, separators=(",", ":")))
                        break
        else:
            selected_keys = {str(k).strip() for sec in cache_selected_sections for k in (sec.get("image_keys") or []) if str(k).strip()}
            for _img in (runtime_lesson_cache.get("images") or []):
                if str(_img.get("image_key") or "").strip() in selected_keys:
                    vision_fact_chars += len(json.dumps(_img.get("vision") or {}, ensure_ascii=False, separators=(",", ":")))
        print(
            f"[VISION CACHE AUDIT] request={request_id} mode=cache "
            f"cache_images={len(runtime_lesson_cache.get('images') or [])} "
            f"selected_images={len(rich_images)} vision_fact_chars={vision_fact_chars} "
            f"image_parts_sent_to_gemini=0"
        )
        print(f"[IMAGE CACHE] reused={len(rich_images)} cached images; not sent to Gemini")
    elif ambiguous_study_request or lesson_intro_request or recommendation_only_request or exercise_suggestion_only_request:
        rich_images = []
        if ambiguous_study_request:
            print("[IMAGE SKIP] ambiguous study request: no image retrieval/attachment")
        elif recommendation_only_request:
            print("[IMAGE SKIP] recommendation-only request: no image retrieval/attachment")
        else:
            print("[IMAGE SKIP] lesson introduction/selection turn: no image retrieval/attachment")
    elif study_retrieval_allowed:
        rich_images = _retrieve_images_for_text_chunks(
            text_chunks, index, namespace, query_vector
        )
        print(
            f"[VISION CACHE AUDIT] request={request_id} mode=live_retrieval "
            f"selected_images={len(rich_images)} image_parts_sent_to_gemini=0"
        )
    else:
        rich_images = []

        # When the current boxchat is still inside an active lesson and the
        # learner has NOT explicitly switched lesson/content type, images must
        # belong to that same active lesson. This prevents a "wrap up / next
        # lesson suggestion" turn from accidentally attaching visuals from the
        # next lesson merely because RAG surfaced them. The text answer may still
        # mention the next lesson as a suggestion, but its images are deferred
        # until the learner explicitly starts that lesson.
        if (thread_scope_locked and thread_scope) or requested_lesson or requested_topic:
            source_scope = thread_scope if (thread_scope_locked and thread_scope) else requested_scope
            scope_content_type = _normalize_content_type(source_scope.get("content_type")) or None
            scope_lesson = str(source_scope.get("lesson") or "").strip() or None
            scope_topic = str(source_scope.get("topic") or "").strip() or None
            filtered_images = []
            for item in rich_images:
                same_type = True
                item_type = _normalize_content_type(item.get("content_type")) or None
                if scope_content_type and item_type and item_type != scope_content_type:
                    same_type = False
                item_lesson = str(item.get("lesson") or "").strip() or None
                item_topic = str(item.get("topic") or "").strip() or None
                same_lesson = (not scope_lesson or not item_lesson or item_lesson == scope_lesson)
                same_topic = (not scope_topic or not item_topic or item_topic == scope_topic)
                if same_type and same_lesson and same_topic:
                    filtered_images.append(item)
                else:
                    print(
                        f"[IMAGE SKIP] outside active thread scope: "
                        f"item_lesson={item_lesson!r} item_topic={item_topic!r} "
                        f"scope_lesson={scope_lesson!r} scope_topic={scope_topic!r}"
                    )
            rich_images = filtered_images

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
    if runtime_cache_hit and runtime_cache_initial:
        prompt_history = []
    else:
        prompt_history = (recent_history[-4:] if runtime_cache_hit else (recent_history[-20:] if study_retrieval_allowed else recent_history[-2:])) if recent_history else []

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
    if ambiguous_study_request or lesson_intro_request or recommendation_only_request or exercise_suggestion_only_request:
        image_orders = []
    if image_orders:
        markers = ", ".join(f"[[IMG_CHUNK_{n}]]" for n in image_orders)
        image_marker_rule = (
            f"\n- Các chunk có ảnh tương ứng là: {markers}. "
            "Khi phần trả lời của cậu sử dụng nội dung của một chunk có ảnh, "
            "hãy đặt marker tương ứng NGAY SAU đúng đoạn/câu trả lời của chunk đó. "
            "Không gom nhiều marker về cuối câu trả lời. Không đổi thứ tự marker. "
            "Marker chỉ là kỹ thuật nội bộ, không được giải thích cho học sinh."
        )

    if lesson_intro_request:
        mode_specific_rules = """
QUY TẮC RIÊNG CHO LƯỢT GIỚI THIỆU/CHỌN BÀI (BẮT BUỘC):
- Người học vừa yêu cầu mở một bài/chủ đề cụ thể. Đây là lượt giới thiệu/chuyển vào bài, chưa phải lượt dạy chi tiết.
- Giới thiệu ngắn gọn bài/chủ đề đã chọn và hỏi người học muốn bắt đầu phần nào nếu cần.
- TUYỆT ĐỐI KHÔNG chèn marker ảnh và không yêu cầu/đính kèm ảnh trong lượt này.
- Không lấy ảnh chỉ vì RAG có image_keys; ảnh sẽ được lấy ở lượt học nội dung tiếp theo.
"""
    elif ambiguous_study_request:
        mode_specific_rules = """
QUY TẮC RIÊNG CHO YÊU CẦU HỌC CHƯA RÕ Ý (BẮT BUỘC):
- Người học đang nói rằng họ muốn học nhưng CHƯA xác định học theo cách nào hoặc học nội dung nào.
- KHÔNG tự mở Ngữ pháp Bài 1, Bài tập Bài 1, hay bất kỳ lesson nào chỉ vì đó là kết quả RAG đứng đầu hoặc là tiến độ cũ.
- Hãy hỏi ngắn gọn và thân thiện để người học chọn một trong 3 hướng:
  1) Học theo lộ trình từ đầu/tiếp theo,
  2) Học tiếp hoặc ôn lại bài đang dở,
  3) Học một bài/chủ đề cụ thể — người học chỉ cần nói tên bài/chủ đề.
- Nếu có tiến độ cũ, có thể nhắc rằng Doraemon đang có một bài đang dở, nhưng KHÔNG tự động mở bài đó; phải để người học chọn.
- Không trình bày nội dung của một lesson cụ thể ở lượt này.
"""
    else:
        mode_specific_rules = ""

    if exercise_suggestion_only_request:
        mode_specific_rules = """QUY TẮC RIÊNG CHO LƯỢT GỢI Ý BÀI TẬP (BẮT BUỘC):
- Người học chỉ đang xin gợi ý bài tập, chưa yêu cầu mở/giải một bài cụ thể.
- Có thể giới thiệu 1-3 dạng/bài tập phù hợp dựa trên RAG nếu nguồn đủ dữ liệu.
- Không được tự chuyển sang dạy bài mới.
- Tuyệt đối không đính kèm ảnh ở lượt gợi ý này; ảnh chỉ xuất hiện khi người học chọn/mở bài tập cụ thể và hệ thống xác định đúng chunk ảnh.
- Không yêu cầu người học học một bài khác nếu họ chỉ xin gợi ý bài tập.
"""

    # Content-type-specific teacher behavior. These rules are intentionally
    # explicit so Gemini follows a consistent teaching workflow rather than
    # merely summarizing retrieved material.
    if requested_content_type == "Bài tập":
        mode_specific_rules = """
QUY TẮC RIÊNG CHO BÀI TẬP — PHẢI ĐÓNG VAI GIÁO VIÊN (BẮT BUỘC):
- Nếu đây là lượt RA BÀI / bắt đầu một bài tập và học sinh chưa nộp đáp án: hãy đưa ra đề bài rõ ràng, đúng dữ liệu RAG; ngay bên dưới phải có mục **💡 Gợi ý cách làm**. Gợi ý chỉ hướng dẫn phương pháp/định hướng, KHÔNG nói luôn đáp án.
- Nếu học sinh đã gửi đáp án hoặc lời giải: coi đó là BÀI NỘP. Hãy tự chấm bằng RAG và ảnh đúng chunk, nêu rõ ĐÚNG/SAI cho từng câu/ý, đáp án đúng (nếu có), rồi giải thích cách giải cụ thể, từng bước, để học sinh hiểu vì sao.
- Với bài tính tiền/gọi món: đọc đúng món và giá từ CHUNK + ẢNH của đúng khách/order; thực hiện phép tính đầy đủ; không suy đoán giá hoặc món không có trong nguồn.
- Nếu học sinh sai: chỉ ra chính xác bước sai, giải thích lỗi và làm mẫu lại từ đầu/đến bước cần thiết. Nếu đúng: vẫn giải thích vì sao đúng, không chỉ nói “đúng”.
- Không yêu cầu học sinh tự kiểm tra lại khi nguồn đã đủ dữ kiện để chấm.
- Sau khi giải xong, có thể đưa câu tiếp theo hoặc bài luyện tương tự ngắn nếu phù hợp, nhưng không làm mất trọng tâm bài đang học.
"""
    elif requested_content_type == "Ngữ pháp":
        mode_specific_rules = """
QUY TẮC RIÊNG CHO NGỮ PHÁP — PHẢI ĐÓNG VAI GIÁO VIÊN (BẮT BUỘC):
- Luồng chính của bài dùng nội dung đã publish trong DB và luôn có nút **Tiếp tục** ở mỗi bước, không gọi Gemini chỉ để chuyển bước.
- Nếu học sinh hỏi/giải thích thêm ngoài luồng chính, chỉ dùng đúng **1 lượt hội thoại ngay trước đó** + câu hiện tại làm context cho Gemini. Không gửi toàn bộ lịch sử chat, không gửi toàn bộ lesson/RAG context.
- Trả lời ngắn gọn, đúng trọng tâm, bám nội dung bài ngữ pháp đang học.
- Nếu câu hỏi chỉ hỏi cách đọc/nghĩa của một ví dụ đã xuất hiện, ưu tiên dữ kiện DB/lượt chat trước; không bịa lại nội dung.
"""
    elif requested_content_type == "Giáo trình":
        mode_specific_rules = """
QUY TẮC RIÊNG CHO GIÁO TRÌNH — PHẢI ĐÓNG VAI GIÁO VIÊN (BẮT BUỘC):
- Khi học sinh yêu cầu học một bài/lesson của Giáo trình, trước tiên phải có **📚 Mở đầu bài học**: giới thiệu ngắn gọn mục đích của bài, bài này giúp học sinh làm được gì và các kiến thức/chủ điểm chính sẽ học.
- Sau phần mở đầu, dạy **từng phần của giáo trình theo đúng thứ tự nguồn RAG**. Mỗi phần phải được giải thích chi tiết, dễ hiểu, có ví dụ từ chính nguồn khi nguồn có, và liên hệ với mục tiêu của bài. Không chỉ tóm tắt toàn bài trong một đoạn ngắn.
- Khi có nhiều mục/điểm kiến thức, trình bày tuần tự: giải thích → ví dụ → lưu ý/dễ nhầm (nếu nguồn hỗ trợ) → chuyển sang mục tiếp theo.
- Cuối bài phải có **📝 Tổng kết**: tổng hợp các từ vựng mới và ngữ pháp/cấu trúc mới xuất hiện trong bài, bám theo RAG CONTEXT; không tự bịa danh sách ngoài nguồn.
- Không tự thêm bài tập bổ sung. Chỉ cho người học làm câu hỏi/yêu cầu thực sự có trong từng chunk nguồn.
- Nếu học sinh chỉ hỏi một chi tiết nhỏ của giáo trình, không cần ép toàn bộ cấu trúc trên; chỉ áp dụng đầy đủ khi học sinh yêu cầu học/trình bày cả bài hoặc một phần bài đủ lớn.
"""
    elif any(
        _normalize_chunk_text_for_match(c.get("metadata",{}).get("content_type")) == "giáo trình"
        and "facts nguồn của bảng" in _normalize_chunk_text_for_match(c.get("text"))
        for c in text_chunks
    ):
        mode_specific_rules = """
QUY TẮC RIÊNG CHO NỘI DUNG CÓ BẢNG:
- FACTS NGUỒN CỦA BẢNG là dữ kiện đã được Vision đọc từ ẢNH gốc. Dùng chúng để suy luận, không chỉ mô tả lại bảng.
- Khi câu hỏi yêu cầu tìm một thời điểm/ngày/giá trị từ nhiều bảng, hãy thực hiện phép đối chiếu logic giữa các facts rồi đưa ra đáp án nếu dữ kiện đủ.
- Với bảng lịch, ô trống có thể mang nghĩa "rảnh" nếu chính bố cục bảng thể hiện không có hoạt động ở khung đó; ký hiệu ／ không đồng nghĩa với ô trống/rảnh.
- Không yêu cầu học sinh tự kiểm tra lại nếu chính các facts nguồn đã đủ để suy ra đáp án.
- Nếu có nhiều bảng trong cùng bài, giữ đúng quan hệ giữa từng bảng và ảnh của bảng đó.
"""

    prompt = f"""Bạn là Doraemon, gia sư tiếng Nhật cá nhân.

QUY TẮC NGÔN NGỮ:
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, hãy trả lời bằng tiếng Nhật, trừ khi người dùng yêu cầu ngôn ngữ khác.
- Nếu người dùng đang dùng tiếng Việt, tiếp tục trả lời bằng tiếng Việt.


NGUYÊN TẮC:
- Thực hiện ngay yêu cầu học tập cụ thể; không hỏi lại nếu đã rõ bài/chủ đề.
- Nếu người học chỉ nói chung chung "muốn học" mà chưa nói học theo lộ trình, học tiếp bài đang dở hay học bài/chủ đề cụ thể, PHẢI hỏi họ chọn hướng; tuyệt đối không tự chọn một bài dựa trên RAG hoặc tiến độ cũ.
- Nội dung gồm đúng 5 loại ngang hàng: Giáo trình, Từ vựng, Ngữ pháp, Bài tập, Truyện đọc. Kanji và Bộ thủ là lesson của Từ vựng, không phải content type.
- Mỗi content type có thể có nhiều sách/tài liệu; chỉ sử dụng đúng nguồn mà RAG và ACTIVE LEARNING STATE xác định.
- Với Giáo trình: học theo FLOW CỐ ĐỊNH của server: B0 giới thiệu mục tiêu + từ vựng cần học + ngữ pháp cần học; B1..Bn mỗi bước chỉ giải thích đúng MỘT CHUNK của Knowledge Cache; sau mỗi chunk chỉ bắt người học làm bài nếu chính LLM khi dạy chunk đó xác định trong source/vision facts có bài tập thật; không được tự bịa. Bước cuối luôn là Tổng kết. Không được tự đổi thứ tự hoặc gộp nhiều chunk vào một teaching step.
- Bài tập nằm trong chunk của Giáo trình vẫn thuộc content type Giáo trình. Khi chấm câu hỏi đó, được phép lấy toàn bộ các chunk của đúng bài để đối chiếu nếu câu hỏi liên quan nhiều phần.
- Khi người học yêu cầu học/trình bày trọn một bài của Giáo trình, sau phần nội dung chính hãy thêm một mục ngắn “🤖 Doraemon nhận xét” (khoảng 3-5 ý hoặc đoạn ngắn): nêu bài này trọng tâm gì, 1-3 điểm cần nhớ, một lỗi dễ nhầm hoặc mẹo học, và gợi ý bước luyện tiếp. Nhận xét phải được suy ra từ chính RAG CONTEXT/ACTIVE LEARNING STATE, không bịa thêm kiến thức ngoài nguồn.
- “Doraemon nhận xét” là phần hỗ trợ sư phạm, không thay thế hay viết lại toàn bộ giáo trình. Nếu người học chỉ hỏi một chi tiết nhỏ trong bài, không cần ép thêm một phần nhận xét dài; chỉ thêm khi phù hợp hoặc khi người học đang kết thúc/ôn lại toàn bài.
- Khi BOXCHAT ĐANG MỞ, RECENT CHAT là ngữ cảnh hội thoại ưu tiên số 1 cho tối đa 10 lượt gần nhất. ACTIVE LEARNING STATE chỉ là ngữ cảnh dự phòng. Không được dùng tiến độ cũ để ghi đè chủ đề đang được trao đổi trong boxchat.
- Nếu RECENT CHAT cho thấy tin nhắn hiện tại đang sửa/chất vấn câu trả lời trước (ví dụ "...có lịch rồi mà", "không đúng", "cậu nhầm"), bắt buộc coi đó là PHẢN HỒI TIẾP NỐI của bài đang học: xem lại câu trả lời ngay trước, đối chiếu RAG/ảnh nguồn, sửa đúng chi tiết bị chỉ ra và KHÔNG chuyển sang lesson/content type/bài tập khác.
- Chỉ chuyển sang lesson/content type khác khi chính tin nhắn hiện tại thể hiện rõ yêu cầu chuyển (ví dụ "chuyển sang...", "mình muốn học bài...").
- Không được lấy một tên bài xuất hiện trong câu trả lời cũ để tự chuyển lesson khi học sinh chỉ đang sửa một chi tiết.
- Với Giáo trình đang chạy curriculum flow, KHÔNG dùng marker `[[LESSON_END_READY]]`; server tự điều khiển bước tiếp theo bằng state/buttons.
- Với Bài tập: để học sinh làm trước, nhưng ngay khi học sinh gửi đáp án/câu trả lời, phải tự chấm bằng nguồn RAG và ảnh đúng chunk; không bắt học sinh tự tính lại nếu dữ kiện đã đủ.
- Với Truyện đọc: bám tài liệu được RAG cung cấp. Nếu chunk nguồn có OCR/text thì coi đó là văn bản nguồn hợp lệ.
- Không bịa nội dung/trang không có trong RAG.
- Với Study Plan: khi người học đã đi đến cuối một bài/đơn vị học và câu hỏi cho thấy họ đang kết thúc bài, hãy hỏi ngắn: "Cậu đã học xong bài này chưa? Nếu xong báo Doraemon nhé." Không tự đánh dấu completed chỉ vì đã trình bày nội dung. Chỉ khi người học xác nhận thì hệ thống mới coi bài là completed.
- Khi bài hiện tại mới kết thúc và Doraemon chỉ đang gợi ý/nhắc bài tiếp theo, KHÔNG được dạy nội dung của bài tiếp theo và KHÔNG được chèn ảnh của bài tiếp theo. Chỉ bắt đầu lấy nội dung/ảnh bài mới sau khi người học xác nhận hoặc yêu cầu học bài mới rõ ràng.
- Quan trọng: ảnh không được tìm theo độ giống câu hỏi. Ảnh table phải thuộc đúng CHUNK chứa explanation của chính table đó.
- Ảnh có image_scope=lesson là ngoại lệ có chủ đích: đó là hình minh họa chung cho toàn bài/lesson, chỉ được dùng khi trả lời trong đúng lesson và không được coi là ảnh của riêng một table chunk.
- Không được dùng ảnh của chunk khác, trang khác hoặc lesson khác chỉ vì nó có vẻ phù hợp.
{image_marker_rule}

{mode_specific_rules}

ACTIVE LEARNING STATE:
{json.dumps(active_state, ensure_ascii=False, default=str, separators=(",", ":"))}

DANH MỤC (chỉ có khi cần gợi ý):
{json.dumps(prompt_catalog, ensure_ascii=False, default=str, separators=(",", ":"))}

RAG CONTEXT:
{chr(10).join(prompt_contexts)}

RECENT CHAT — NGỮ CẢNH ƯU TIÊN CỦA BOXCHAT ĐANG MỞ (tối đa 10 lượt gần nhất):
{json.dumps(prompt_history, ensure_ascii=False, default=str, separators=(",", ":"))}
- Đây là lịch sử của chính boxchat hiện tại, không phải lịch sử học tập chung.
- Dùng nó để hiểu "cậu", "đó", "bảng này", "sáng thứ 6", "mình nói ý này", "câu trước", v.v.
- Không được bỏ qua ngữ cảnh này để nhảy sang bài khác chỉ vì ACTIVE LEARNING STATE hoặc RAG metadata cũ gợi ý một lesson khác.

TIN NHẮN HIỆN TẠI:
{query_text}"""

    if runtime_cache_hit:
        cache_context = []
        for order, sec in enumerate(cache_selected_sections):
            label = (
                f"[SECTION_{order}] [Bài: {runtime_lesson_cache.get('lesson','')} | "
                f"Chủ đề: {runtime_lesson_cache.get('topic','')} | Trang: {sec.get('page','')}]"
            )
            cache_context.append(label + "\n" + str(sec.get("text") or ""))
        selected_keys = {str(k).strip() for sec in cache_selected_sections for k in (sec.get("image_keys") or []) if str(k).strip()}
        if curriculum_flow_active and curriculum_step in {curriculum_map["global_exercise_step"], curriculum_map["summary_step"]}:
            selected_vision = [x.get("vision", {}) for x in (runtime_lesson_cache.get("images") or []) if x.get("vision")]
            vision_text = json.dumps(selected_vision, ensure_ascii=False, separators=(',', ':'))[:7000]
        else:
            selected_vision = [x.get("vision", {}) for x in (runtime_lesson_cache.get("images") or []) if str(x.get("image_key") or "").strip() in selected_keys]
            vision_text = json.dumps(selected_vision, ensure_ascii=False, separators=(',', ':'))[:2600]
        marker_rule = ""
        if curriculum_flow_active:
            secs=curriculum_map["sections"]
            vocab_grammar_context=""
            if curriculum_step == 0:
                # B0 chỉ làm nhiệm vụ định hướng: không dạy lại từ vựng/ngữ pháp.
                # Dùng một preview ngắn của từng chunk để giữ prompt nhỏ, tránh lặp lại
                # nội dung chi tiết sẽ được dạy ở B1/B2/...
                overview_text = "\n\n".join(
                    f"[CHUNK_{i}] {str(sec.get('text') or '')[:1200]}"
                    for i, sec in enumerate(secs)
                )
                cache_prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật cá nhân. Đây là BƯỚC 0 — GIỚI THIỆU
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
 của bài {runtime_lesson_cache.get('lesson','')}.

MỤC TIÊU CỦA BƯỚC 0:
- Chỉ giới thiệu TỔNG QUAN bài học: bài này nói về chủ đề gì, người học sẽ làm quen với những nội dung/phần nào, và mục tiêu sau khi học xong.
- KHÔNG dạy từ vựng ở bước 0.
- KHÔNG liệt kê từ vựng, Kanji, cách đọc hoặc phát âm ở bước 0. Những nội dung này chỉ được dạy trong từng chunk tương ứng ở các bước sau.
- KHÔNG giải thích ngữ pháp ở bước 0.
- KHÔNG lặp lại chi tiết của các chunk ở bước 0.
- Không bịa kiến thức ngoài nguồn.

Nếu nguồn có một câu hỏi/bài tập TOÀN BÀI không thuộc riêng một chunk, hãy ghi ở cuối bằng marker nội bộ:
[[GLOBAL_LESSON_EXERCISE_Q]] Q: <đúng câu hỏi/yêu cầu>. Không yêu cầu người học giải ở bước này.

Cuối cùng hỏi người học có muốn sang phần 1 không. Đây là bước giới thiệu, không dùng [[LESSON_END_READY]].

SOURCE PREVIEW (chỉ để hiểu chủ đề tổng quan, không được biến thành bài giảng chi tiết):\n{overview_text}\n\nTIN NHẮN HIỆN TẠI:\n{query_text}"""
            elif 1 <= curriculum_step <= len(secs):
                sec=secs[curriculum_step-1]
                if curriculum_waiting == "chunk_answer":
                    cache_prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật. Đây vẫn là PHẦN {curriculum_step}
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
 của bài {runtime_lesson_cache.get('lesson','')}.
Đây là lượt HỌC SINH TRẢ LỜI CÂU HỎI/BÀI TẬP THỰC SỰ NẰM TRONG CHUNK NÀY.
CHỈ sử dụng đúng chunk bên dưới và vision facts của chính chunk này.
Hãy đánh giá đáp án của học sinh, chỉ ra đúng/sai, giải thích ngắn gọn dựa trên nguồn, và nếu cần cho biết đáp án đúng.
Không lấy nội dung từ chunk khác.
Sau khi nhận xét xong, hỏi học sinh có muốn sang phần tiếp theo không.

CHUNK SOURCE:
{str(sec.get('text') or '')}

VISION FACTS CỦA CHUNK NÀY:
{vision_text}

LỊCH SỬ HỘI THOẠI: KHÔNG DÙNG LẠI LỊCH SỬ DÀI. Chỉ xem câu trả lời hiện tại bên dưới.

ĐÁP ÁN CỦA HỌC SINH:
{query_text}"""
                else:
                    cache_prompt=f"""Bạn là Doraemon, gia sư tiếng Nhật. Đây là PHẦN {curriculum_step}
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
 của bài {runtime_lesson_cache.get('lesson','')}.
CHỈ giải thích đúng MỘT CHUNK dưới đây. Không lấy nội dung, dữ kiện, hình/bảng hoặc ví dụ từ bất kỳ chunk nào khác. Không tóm tắt các chunk khác và không dạy trước phần sau.
- Giải thích rõ ràng, dễ hiểu.
- Khai thác từ vựng/kanji/grammar có trong chính chunk này khi phù hợp; từ vựng phải kèm cách đọc và hướng dẫn phát âm nếu nguồn có.
- Dùng chính ví dụ/số liệu/bảng trong nguồn.
- Nếu chunk có hình/bảng, chỉ dùng VISION FACTS được gắn chính xác với chunk này; không dùng vision facts của chunk khác và không yêu cầu nhìn lại ảnh.
- QUAN TRỌNG: Trong chính CHUNK + VISION FACTS này, hãy kiểm tra xem nguồn có đưa ra một CÂU HỎI/BÀI TẬP/YÊU CẦU THỰC SỰ mà người học phải trả lời hay không.
- Chỉ đánh dấu `[[CHUNK_EXERCISE]]` khi câu hỏi/yêu cầu thực sự nằm trong chunk này VÀ có thể trả lời đầy đủ chỉ bằng chunk này + vision facts của chính chunk.
- Nếu nguồn có một câu hỏi/bài tập nhưng để trả lời phải đối chiếu nội dung từ chunk khác/toàn bài, KHÔNG hỏi user ở đây; ở cuối câu trả lời ghi:
  `[[WHOLE_LESSON_EXERCISE]]`
  `Q: <đúng câu hỏi/yêu cầu>`
  để backend lưu câu hỏi cho bước bài tập toàn bài.
- Nếu không có bài tập thật, đánh dấu `[[NO_CHUNK_EXERCISE]]`.
- TUYỆT ĐỐI KHÔNG tự tạo câu hỏi/bài tập mới. Không suy luận từ việc chunk có dấu `?`, có bảng, có số liệu, hoặc có nội dung có thể dùng để đặt câu hỏi rằng đó là bài tập.
- Nếu có bài tập thật thuộc riêng chunk này, hãy giải thích phần nội dung trước rồi nêu ĐÚNG câu hỏi/yêu cầu đó cho học sinh làm. Nếu không có bài tập hoặc là bài tập toàn bài, chỉ giải thích chunk.
- Marker phải xuất hiện ở CUỐI câu trả lời. Nếu là `[[WHOLE_LESSON_EXERCISE]]`, đặt ngay sau đó một dòng `Q: <đúng câu hỏi/yêu cầu>`. Không giải thích marker cho học sinh.
- Nếu không có bài tập tại chunk này, kết thúc phần giải thích bằng một lời mời sang phần tiếp theo; backend sẽ thêm nút “Bạn muốn sang phần tiếp theo không?”.

CHUNK SOURCE DUY NHẤT:
{str(sec.get('text') or '')}

VISION FACTS CỦA CHUNK DUY NHẤT:
{vision_text}

QUY TẮC CONTEXT: Không dùng lịch sử chat dài, không lấy dữ kiện từ chunk khác, không lấy context của phần trước. Chỉ sử dụng chunk hiện tại + vision facts của chunk hiện tại.

TIN NHẮN HIỆN TẠI:
{query_text}"""
            elif curriculum_step == curriculum_map["global_exercise_step"]:
                saved_q = str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
                saved_ev = str((study_session or {}).get("curriculum_global_exercise_evidence") or "").strip()
                core_history = str((study_session or {}).get("curriculum_intro_b0b1_history") or "").strip()
                _global_answer_turn = bool(saved_q) and bool((query_text or "").strip()) and not bool(data.action) and not _is_continue_confirmation(query_text)
                if _global_answer_turn:
                    # Answer/evaluation/follow-up turn: USE THE FULL KNOWLEDGE of the lesson.
                    full_sections = []
                    for i, sec2 in enumerate(runtime_lesson_cache.get("sections") or []):
                        full_sections.append(f"[CHUNK_{i}]\n" + str(sec2.get("text") or ""))
                    full_vision = json.dumps([x.get("vision", {}) for x in (runtime_lesson_cache.get("images") or []) if x.get("vision")], ensure_ascii=False, separators=(",", ":"))[:12000]
                    cache_prompt=f"""Bạn là Doraemon, chấm và giải bài tập cuối bài.
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
\n\nCÂU HỎI: {saved_q}\n\nBẰNG CHỨNG BAN ĐẦU: {saved_ev}\n\nKNOWLEDGE ĐẦY ĐỦ CỦA TOÀN BÀI:\n{chr(10).join(full_sections)}\n\nVISION FACTS CỦA TOÀN BÀI:\n{full_vision}\n\nCÂU TRẢ LỜI / YÊU CẦU HIỆN TẠI CỦA HỌC SINH:\n{query_text}\n\nNHIỆM VỤ:\n- Dùng TOÀN BỘ Knowledge + Vision Facts để giải và kiểm tra.\n- Nếu học sinh hỏi “đáp án là gì?”, “không biết”, hoặc hỏi lại đáp án, hãy tự giải và đưa ra đáp án đúng.\n- Nếu học sinh đã trả lời, đánh giá đúng/sai và giải thích ngắn gọn dựa trên toàn bộ nguồn.\n- TUYỆT ĐỐI không nói “giáo trình không có đáp án” chỉ vì không thấy một answer key riêng; hãy tự suy luận từ dữ kiện nguồn khi đủ dữ kiện.\n- Nếu dữ kiện không đủ để kết luận, nói rõ phần dữ kiện nào còn thiếu; không bịa đáp án.\n- Không tạo câu hỏi mới."""
                else:
                    if saved_q:
                        cache_prompt=f"""Chỉ kiểm tra câu hỏi đã được Doraemon xác định trước đây có phải là bài tập TOÀN BÀI chưa giải không.
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
 Chỉ dùng context B0+B1.\n[[GLOBAL_EXERCISE]]\nQ: {saved_q}\nE: {core_history[-1200:]}\nNếu không đủ căn cứ: [[NO_GLOBAL_EXERCISE]]."""
                    else:
                        cache_prompt=f"""Bạn là Doraemon. Chỉ dùng CONTEXT B0 + B1
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
 bên dưới để xác định có câu hỏi/bài tập TOÀN BÀI chưa giải hay không.\nNếu có, trả:\n[[GLOBAL_EXERCISE]]\nQ: <đúng câu hỏi>\nE: <bằng chứng ngắn>\nNếu không: [[NO_GLOBAL_EXERCISE]]. Không tạo câu hỏi mới và không dùng Knowledge khác.\n\nCONTEXT B0 + B1:\n{core_history[-5000:] or '(chưa có context B0+B1)'}"""
            elif curriculum_step == curriculum_map["summary_step"]:
                # Summary uses ONLY B0+B1 teaching context, plus the already stored exercise result.
                core_history = str((study_session or {}).get("curriculum_intro_b0b1_history") or "").strip()
                global_q = str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
                global_result = str((study_session or {}).get("curriculum_global_exercise_result") or "").strip()
                cache_prompt=f"""Bạn là Doraemon. Đây là BƯỚC CUỐI — TỔNG KẾT
- Nếu người dùng đang giao tiếp bằng tiếng Nhật, trả lời bằng tiếng Nhật, trừ khi họ yêu cầu ngôn ngữ khác.
 bài {runtime_lesson_cache.get('lesson','')}.\n\nCONTEXT ĐƯỢC PHÉP DÙNG:\n- CHỈ dùng phần Doraemon đã nói ở B0 (mở đầu) và B1 (chunk 1).\n- Không dùng các chunk 2+, Vision Facts hay Knowledge Cache trực tiếp.\n- Nếu có kết quả bài tập cuối bài đã được lưu bên dưới, dùng nó để nêu kết quả.\n- Không nói rằng “giáo trình không có đáp án” nếu bài tập đã có kết quả/đáp án được lưu; giữ cách diễn đạt tự nhiên và dựa trên dữ liệu được cung cấp.\n\nHãy tổng kết ngắn gọn theo context trên, nêu mục tiêu, từ vựng/phát âm, ngữ pháp, các điểm chính đã được giới thiệu ở B0+B1, và kết quả bài tập nếu có. Cuối cùng hỏi: “Cậu thấy mình đã nắm được bài này chưa?”\n\nCONTEXT B0 + B1:\n{core_history[-5000:] or '(chưa có context B0+B1)'}\n\nCÂU HỎI BÀI TẬP CUỐI BÀI (nếu có):\n{global_q or '(không có)'}\n\nKẾT QUẢ BÀI TẬP ĐÃ LƯU (nếu có):\n{global_result or '(chưa có)'}\n\nTIN NHẮN HIỆN TẠI:\n{query_text}"""
            else:
                cache_prompt=None
            if cache_prompt is not None:
                prompt=cache_prompt
                print(f"[CURRICULUM PROMPT] request={request_id} step={curriculum_step} prompt_chars={len(prompt)} image_parts_sent_to_gemini=0")
                # Skip the legacy cache prompt builder below.
                marker_rule = "__CURRICULUM_PROMPT_READY__"
        if curriculum_flow_active:
            # Curriculum-specific prompt was built above. Do not overwrite it with the legacy cache prompt.
            if prompt is None:
                raise RuntimeError("curriculum prompt was not constructed")
            if curriculum_step == curriculum_map["summary_step"]:
                _ih = str((study_session or {}).get("curriculum_intro_b0b1_history") or "").strip()
                _sq = str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
                _sev = str((study_session or {}).get("curriculum_global_exercise_evidence") or "").strip()
                _res = str((study_session or {}).get("curriculum_global_exercise_result") or "").strip()
                print(
                    f"[CURRICULUM SUMMARY CONTEXT] request={request_id} "
                    f"mode=b0_b1_only sections_sent=0 vision_facts_sent=0 "
                    f"b0b1_history_chars={len(_ih[-5000:])} question_chars={len(_sq)} result_chars={len(_res)} "
                    f"prompt_chars={len(prompt)}"
                )
            elif curriculum_step == curriculum_map["global_exercise_step"]:
                _ih = str((study_session or {}).get("curriculum_intro_b0b1_history") or "").strip()
                _sq = str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
                _sev = str((study_session or {}).get("curriculum_global_exercise_evidence") or "").strip()
                _full_answer_turn = bool((study_session or {}).get("curriculum_global_exercise_question")) and bool((query_text or "").strip()) and not bool(data.action) and not _is_continue_confirmation(query_text)
                print(
                    f"[CURRICULUM GLOBAL EXERCISE CONTEXT] request={request_id} "
                    f"mode={'full_knowledge_answer' if _full_answer_turn else 'b0_b1_detection'} "
                    f"sections_sent={len(cache_selected_sections) if _full_answer_turn else 0} "
                    f"vision_facts_sent={len(vision_text) if _full_answer_turn else 0} "
                    f"b0b1_history_chars={len(_ih[-5000:])} question_chars={len(_sq)} evidence_chars={len(_sev)} "
                    f"prompt_chars={len(prompt)}"
                )
            else:
                print(
                    f"[KNOWLEDGE CACHE PROMPT] request={request_id} "
                    f"curriculum_step={curriculum_step} sections={len(cache_selected_sections)} "
                    f"prompt_chars={len(prompt)} vision_fact_chars={len(vision_text)} image_parts_sent_to_gemini=0"
                )
            if curriculum_flow_active and 1 <= curriculum_step <= len(curriculum_map["sections"]):
                print(f"[CURRICULUM CHUNK CONTEXT AUDIT] request={request_id} mode=exact_chunk_only history_in_prompt=0 selected_chunks=1 selected_vision_chars={len(vision_text)}")
        else:
            marker_rule = ""
            if rich_images:
                marker_names = ", ".join(
                    f"[[IMG_CHUNK_{i}]]" for i in sorted({int(x.get('_chunk_order', 0)) for x in rich_images})
                )
                marker_rule = f"- Nếu dùng dữ kiện từ section có ảnh, đặt marker tương ứng {marker_names} ngay sau đoạn giải thích; marker chỉ dùng cho UI.\n"
            cache_prompt = f"""Bạn là Doraemon, gia sư tiếng Nhật cá nhân.
DẠY DỰA TRÊN KNOWLEDGE CACHE ĐÃ ĐƯỢC XỬ LÝ KHI UPLOAD; không được yêu cầu nhìn lại ảnh.
Bài hiện tại: {runtime_lesson_cache.get('lesson','')} | Chủ đề: {runtime_lesson_cache.get('topic') or ''}

QUY TẮC DẠY:
- Với Giáo trình: giải thích tuần tự, dễ hiểu; dùng ví dụ trong nguồn nếu có; không bịa ngoài nguồn.
- Khi bắt đầu bài: giới thiệu ngắn mục tiêu rồi dạy phần đang có trong CACHE; không tóm tắt dài toàn bài nếu chưa dạy đến đó.
- Khi học sinh hỏi chi tiết: trả lời đúng chi tiết đó dựa vào CACHE.
- Khi thực sự hoàn tất phần hướng dẫn lớn/trọn bài, có thể đặt marker [[LESSON_END_READY]] ở cuối.
- Ảnh chỉ để UI hiển thị; suy luận nội dung chỉ dựa vào text/facts đã cache.
{marker_rule}
VISION FACTS CỦA CÁC ẢNH LIÊN QUAN SECTION HIỆN TẠI:
{vision_text}

KNOWLEDGE CACHE CONTEXT:
{chr(10).join(cache_context)}

RECENT CHAT CỦA BOXCHAT HIỆN TẠI:
{json.dumps(prompt_history, ensure_ascii=False, separators=(',', ':'))}

TIN NHẮN HIỆN TẠI:
{query_text}"""
            prompt = cache_prompt
            print(
                f"[KNOWLEDGE CACHE PROMPT] request={request_id} "
                f"sections={len(cache_selected_sections)} prompt_chars={len(prompt)} "
                f"vision_fact_chars={len(vision_text)} image_parts_sent_to_gemini=0"
            )

    gen_started = time.perf_counter()
    reply, response_model, gen_elapsed = _generate_chat_reply(
        prompt,
        content_type=requested_content_type,
        request_id=request_id,
        gen_started=gen_started,
        user_text=query_text,
    )
    perf_gen = time.perf_counter()

    detected_chunk_exercise = None
    detected_global_exercise = None
    _detected_global_question = ""
    if curriculum_flow_active and curriculum_step == 0:
        raw_intro = reply or ""
        m = re.search(r"\[\[GLOBAL_LESSON_EXERCISE_Q\]\]\s*Q:\s*(.+)$", raw_intro, re.S)
        if m:
            q = m.group(1).strip()
            _set_curriculum_compact_state(user["id"], global_question=q, global_evidence="")
            print(f"[CURRICULUM GLOBAL EXERCISE CANDIDATE] request={request_id} step=0 question_chars={len(q)}")
            reply = re.sub(r"\[\[GLOBAL_LESSON_EXERCISE_Q\]\]\s*Q:\s*.+$", "", reply or "", flags=re.S).rstrip()
    if curriculum_flow_active and 1 <= curriculum_step <= len(curriculum_map["sections"]) and curriculum_waiting != "chunk_answer":
        raw_whole = reply or ""
        mwhole = re.search(r"\[\[WHOLE_LESSON_EXERCISE\]\]\s*Q:\s*(.+)$", raw_whole, re.S)
        if mwhole:
            q = mwhole.group(1).strip()
            _set_curriculum_compact_state(user["id"], global_question=q, global_evidence="")
            print(f"[CURRICULUM GLOBAL EXERCISE CANDIDATE] request={request_id} step={curriculum_step} question_chars={len(q)}")
        if "[[CHUNK_EXERCISE]]" in (reply or ""):
            detected_chunk_exercise = True
            print(f"[CURRICULUM EXERCISE DETECT] request={request_id} step={curriculum_step} has_exercise=1 source='llm_chunk_inspection'")
        elif "[[WHOLE_LESSON_EXERCISE]]" in (reply or ""):
            detected_chunk_exercise = False
            print(f"[CURRICULUM EXERCISE DEFER] request={request_id} step={curriculum_step} whole_lesson=1 source='llm_chunk_inspection'")
        elif "[[NO_CHUNK_EXERCISE]]" in (reply or ""):
            detected_chunk_exercise = False
            print(f"[CURRICULUM EXERCISE DETECT] request={request_id} step={curriculum_step} has_exercise=0 source='llm_chunk_inspection'")
        else:
            detected_chunk_exercise = False
            print(f"[CURRICULUM EXERCISE DETECT] request={request_id} step={curriculum_step} has_exercise=0 source='missing_marker_fail_closed'")
        reply=(reply or '').replace('[[CHUNK_EXERCISE]]','').replace('[[NO_CHUNK_EXERCISE]]','').replace('[[WHOLE_LESSON_EXERCISE]]','')
        if mwhole:
            reply = re.sub(r'\n?Q:\s*' + re.escape(mwhole.group(1).strip()) + r'\s*$', '', reply or '', flags=re.S).rstrip()
    elif curriculum_flow_active and curriculum_step == curriculum_map["global_exercise_step"] and curriculum_waiting != "global_exercise_answer":
        raw_reply = reply or ""
        if "[[GLOBAL_EXERCISE]]" in raw_reply:
            qm = re.search(r"(?:^|\n)Q:\s*(.+?)(?=\nE:|$)", raw_reply, re.S)
            em = re.search(r"(?:^|\n)E:\s*(.+)$", raw_reply, re.S)
            question = qm.group(1).strip() if qm else ""
            evidence = em.group(1).strip() if em else ""
            detected_global_exercise = bool(question)
            globals()["_detected_global_question"] = question
            if detected_global_exercise:
                _set_curriculum_compact_state(user["id"], global_question=question, global_evidence=evidence)
            print(f"[CURRICULUM GLOBAL EXERCISE DETECT] request={request_id} has_exercise={int(detected_global_exercise)} cached_question_chars={len(question)} cached_evidence_chars={len(evidence)}")
        elif "[[NO_GLOBAL_EXERCISE]]" in raw_reply:
            detected_global_exercise = False
            _set_curriculum_compact_state(user["id"], global_question="", global_evidence="")
            print(f"[CURRICULUM GLOBAL EXERCISE DETECT] request={request_id} has_exercise=0")
        else:
            detected_global_exercise = False
            print(f"[CURRICULUM GLOBAL EXERCISE DETECT] request={request_id} has_exercise=0 source='missing_marker_fail_closed'")
        reply = ""

    # Fixed curriculum flow owns the ending for Giáo trình. Legacy LESSON_END_READY remains for non-curriculum types.
    if curriculum_flow_active:
        reply=(reply or '').replace('[[LESSON_END_READY]]','').rstrip()
    lesson_end_ready = bool((not curriculum_flow_active) and study_session and "[[LESSON_END_READY]]" in (reply or ""))
    if lesson_end_ready:
        reply = (reply or "").replace("[[LESSON_END_READY]]", "").rstrip()
        _set_study_end_prompt_pending(user["id"], True)
        study_session["end_prompt_pending"] = True
        print(f"[STUDY SESSION] lesson_end_ready user={user['id']} lesson={study_session.get('lesson')!r}")

    if curriculum_flow_active and (curriculum_step == 0 or (1 <= curriculum_step <= len(curriculum_map["sections"]) and curriculum_waiting != "chunk_answer")):
        intro_piece = re.sub(r"\[\[(?:CHUNK_EXERCISE|NO_CHUNK_EXERCISE|WHOLE_LESSON_EXERCISE|GLOBAL_LESSON_EXERCISE_Q)(?:_[^\]]+)?\]\]", "", reply or "").strip()
        if intro_piece:
            _append_curriculum_intro_history(user["id"], intro_piece)
            _append_curriculum_intro_b0b1_history(user["id"], curriculum_step, intro_piece)
            study_session = _get_study_session(user["id"], chatbox_id=getattr(data, "chatbox_id", None)) or study_session
            print(f"[CURRICULUM INTRO HISTORY] request={request_id} step={curriculum_step} chars={len(study_session.get('curriculum_intro_history','')) if study_session else 0}")
            print(f"[CURRICULUM B0B1 HISTORY] request={request_id} step={curriculum_step} chars={len(study_session.get('curriculum_intro_b0b1_history','')) if study_session else 0}")

    if curriculum_flow_active and 1 <= curriculum_step <= len(curriculum_map["sections"]):
        clean_note = re.sub(r"\[\[(?:CHUNK_EXERCISE|NO_CHUNK_EXERCISE|WHOLE_LESSON_EXERCISE)\]\]", "", reply or "").strip()
        if clean_note:
            prev = str((study_session or {}).get("curriculum_summary_notes") or "").strip()
            combined = (prev + "\n\n" + clean_note).strip()[-6000:]
            _set_curriculum_compact_state(user["id"], summary_notes=combined)
            print(f"[CURRICULUM SUMMARY NOTES] request={request_id} chars={len(combined)}")

    # rich_images was resolved BEFORE Gemini from the exact text chunks.
    # Do not perform any second semantic image search here.
    content_blocks = build_rich_content_blocks(reply, rich_images)
    is_curriculum_answer_turn = bool((query_text or "").strip()) and not bool(data.action)

    if curriculum_flow_active:
        # Step 0 and teaching chunks: normal chunks wait for Continue.
        # Chunks containing an embedded exercise instead wait for an answer,
        # then explicitly return to Continue only after Doraemon has evaluated it.
        if curriculum_step == 0:
            _set_curriculum_flow(user["id"], step=curriculum_step, waiting="continue", exercise_answered=False)
            content_blocks.extend([{"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"}] + _curriculum_continue_blocks(curriculum_step))
        elif 1 <= curriculum_step <= len(curriculum_map["sections"]):
            if curriculum_waiting == "chunk_answer" and is_curriculum_answer_turn:
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="continue", exercise_answered=True)
                content_blocks.extend([{"type":"text","text":"✅ Doraemon đã nhận xét xong. Cậu muốn sang phần tiếp theo chứ? 😊"}] + _curriculum_continue_blocks(curriculum_step))
            elif curriculum_waiting == "chunk_answer" and not curriculum_exercise_answered:
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="chunk_answer", exercise_answered=False)
                content_blocks.extend([
                    {"type":"text","text":"✍️ Cậu có thể trả lời câu hỏi/bài tập có trong phần này. Nếu chưa muốn làm, cậu vẫn có thể sang phần tiếp theo nhé."},
                    {"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"},
                ] + _curriculum_continue_blocks(curriculum_step))
            elif detected_chunk_exercise:
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="chunk_answer", exercise_answered=False)
                content_blocks.extend([
                    {"type":"text","text":"✍️ Trong phần này có một câu hỏi/bài tập từ chính tài liệu. Cậu có thể trả lời ngay; nếu chưa muốn làm, cậu vẫn có thể sang phần tiếp theo nhé."},
                    {"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"},
                ] + _curriculum_continue_blocks(curriculum_step))
            else:
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="continue", exercise_answered=False)
                content_blocks.extend([{"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"}] + _curriculum_continue_blocks(curriculum_step))
        elif curriculum_step == curriculum_map["global_exercise_step"]:
            _saved_global_q_now = str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
            _is_global_answer_turn_now = bool(_saved_global_q_now) and is_curriculum_answer_turn and not _is_continue_confirmation(query_text)
            if _is_global_answer_turn_now:
                # Any user answer/question about the whole-lesson exercise is a FULL-KNOWLEDGE turn.
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="continue_after_global_exercise", exercise_answered=True)
                eval_note = (reply or "").strip()
                if eval_note:
                    _set_curriculum_global_exercise_result(user["id"], eval_note)
                    study_session = _get_study_session(user["id"], chatbox_id=getattr(data, "chatbox_id", None)) or study_session
                    prev = str((study_session or {}).get("curriculum_summary_notes") or "").strip()
                    _set_curriculum_compact_state(user["id"], summary_notes=(prev + "\n\nBÀI TẬP TOÀN BÀI - ĐÁNH GIÁ: " + eval_note).strip()[-6000:])
                content_blocks.extend([{"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"}] + _curriculum_continue_blocks(curriculum_step))
            elif detected_global_exercise:
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="global_exercise_answer", exercise_answered=False)
                # Show the actual whole-lesson question clearly, while keeping the step compact.
                q = str((study_session or {}).get("curriculum_global_exercise_question") or "").strip()
                if not q:
                    q = str(globals().get("_detected_global_question") or "").strip()
                question_text = ("📝 CÂU HỎI BÀI TẬP TOÀN BÀI:\n\n" + q) if q else "📝 BÀI TẬP TOÀN BÀI: Hãy trả lời câu hỏi trong tài liệu."
                content_blocks.extend([
                    {"type":"text","text":question_text},
                    {"type":"text","text":"Cậu hãy trả lời câu hỏi trên nhé. Cậu vẫn có thể bấm Tiếp tục nếu chưa muốn làm. 😊"},
                    {"type":"text","text":"Cậu muốn sang phần tiếp theo chứ? 😊"},
                ] + _curriculum_continue_blocks(curriculum_step))
            elif curriculum_waiting == "continue_after_global_exercise" and not is_curriculum_answer_turn:
                next_summary_step = curriculum_map["summary_step"]
                _set_curriculum_flow(user["id"], step=next_summary_step, waiting="final", exercise_answered=True)
                study_session["curriculum_step"] = next_summary_step
                study_session["curriculum_waiting"] = "final"
                content_blocks.extend(_curriculum_final_blocks())
            else:
                _set_curriculum_flow(user["id"], step=curriculum_step, waiting="continue_after_global_check", exercise_answered=True)
                content_blocks.extend(_curriculum_continue_blocks(curriculum_step))
        elif curriculum_step == curriculum_map["summary_step"]:
            _set_curriculum_flow(user["id"], step=curriculum_step, waiting="final", exercise_answered=True)
            content_blocks.extend(_curriculum_final_blocks())
    if lesson_end_ready:
        content_blocks.extend(_study_end_choice_blocks(_active_session_scope(study_session)))
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
        "[PERF proxy_chat] provider=%s auth=%.3fs state=%.3fs embed=%.3fs rag=%.3fs "
        "llm=%.3fs blocks=%.3fs total=%.3fs text_k=%d image_k=%d prompt_catalog=%d prompt_active_state=%d"
        % (
            LLM_PROVIDER,
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

    print(f"[CHAT END] request={request_id} total={perf_total_done-perf_total:.3f}s")
    return {
        "reply": reply,
        "model": response_model,
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
            cur.execute("DELETE FROM study_plans WHERE user_id=%s", (user["id"],))

            # Mark the account as a fresh learner. The next /session/welcome
            # therefore returns the same onboarding flow as a brand-new user.
            cur.execute("""
                INSERT INTO user_learning_state(user_id,welcome_seen,reset_count,learning_mode,onboarding_completed,updated_at)
                VALUES(%s,FALSE,1,NULL,FALSE,NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    welcome_seen=FALSE,
                    reset_count=user_learning_state.reset_count + 1,
                    learning_mode=NULL,
                    onboarding_completed=FALSE,
                    study_session_active=FALSE,
                    study_session_content_type=NULL,
                    study_session_course=NULL,
                    study_session_lesson=NULL,
                    study_session_topic=NULL,
                    study_session_started_at=NULL,
                    study_end_prompt_pending=FALSE,
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


@app.get("/learning/plan")
def get_learning_plan(authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization)
    profile=_get_learning_profile(user["id"])
    plans=_active_plans(user["id"])
    plan=plans[0] if plans else None
    draft=_latest_draft(user["id"])
    return {"learning_mode":profile.get("learning_mode"),"onboarding_completed":profile.get("onboarding_completed"),"plan":plan,"plans":plans,"draft":draft}

@app.delete("/learning/plan/{plan_id}")
def delete_learning_plan(plan_id: int, authorization: Optional[str] = Header(default=None)):
    """Soft-delete one active Study Plan owned by the current user.

    The plan and its items remain in PostgreSQL for historical/audit purposes,
    but it is no longer ACTIVE, so _active_plans(), auto-chat and plan routing
    will stop selecting or teaching from it.
    """
    user=require_active_user(authorization)
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id,goal_name,content_type,status FROM study_plans WHERE id=%s AND user_id=%s FOR UPDATE",(int(plan_id),user["id"]))
            plan=cur.fetchone()
            if not plan:
                raise HTTPException(404,"Không tìm thấy lộ trình này.")
            if str(plan.get("status") or "").upper() != "ACTIVE":
                raise HTTPException(409,"Lộ trình này không còn đang hoạt động.")

            cur.execute("UPDATE study_plans SET status='DELETED', superseded_at=NOW() WHERE id=%s AND user_id=%s AND status='ACTIVE'",(int(plan_id),user["id"]))

            # If this was the last active plan, keep onboarding completed but
            # move the learner back to free study so deleted plans cannot be
            # selected by future chat routing. Other active plans keep planned mode.
            cur.execute("SELECT COUNT(*) AS n FROM study_plans WHERE user_id=%s AND status='ACTIVE'",(user["id"],))
            active_count=int(cur.fetchone()["n"] or 0)
            if active_count == 0:
                cur.execute("UPDATE user_learning_state SET learning_mode='free', updated_at=NOW() WHERE user_id=%s",(user["id"],))

        conn.commit()
        plans=_active_plans(user["id"])
        return {"success":True,"deleted_plan_id":int(plan_id),"plan":dict(plan),"plans":plans,"remaining_plan_count":len(plans)}
    except HTTPException:
        conn.rollback(); raise
    except Exception as exc:
        conn.rollback()
        print(f"[STUDY PLAN] delete error user={user['id']} plan={plan_id}: {exc}")
        raise HTTPException(500,"Không thể xoá lộ trình lúc này.")
    finally:
        conn.close()


@app.post("/learning/plan/confirm")
def confirm_learning_plan(authorization: Optional[str] = Header(default=None)):
    user=require_active_user(authorization)
    plan=_confirm_latest_draft(user["id"])
    if not plan:
        raise HTTPException(404,"Không có lộ trình nháp để xác nhận.")
    _set_learning_profile(user["id"],"planned",True)
    return {"success":True,"plan":_active_plan(user["id"]),"plans":_active_plans(user["id"])}

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
            cur.execute("""
                SELECT subject,content_type,lesson,lesson_pages,topic,topic_pages,question_pages,answer_pages,source_file,namespace
                FROM knowledge_documents
                UNION ALL
                SELECT subject,content_type,lesson,NULL::VARCHAR AS lesson_pages,NULL::VARCHAR AS topic,
                       NULL::VARCHAR AS topic_pages,NULL::VARCHAR AS question_pages,NULL::VARCHAR AS answer_pages,
                       source_file,'__default__'::VARCHAR AS namespace
                FROM curriculum_lessons WHERE status='PUBLISHED'
                ORDER BY subject,lesson,topic,source_file
            """)
            return {"success":True,"documents":[dict(x) for x in cur.fetchall()]}
    finally: conn.close()

def check_admin(password: str):
    expected = os.getenv("ADMIN_PANEL_PASSWORD", os.getenv("ADMIN_WS_TOKEN", ""))
    if not expected:
        raise HTTPException(500, "ADMIN_PANEL_PASSWORD chưa được cấu hình trên Render.")
    if password != expected:
        raise HTTPException(401, "Admin password không đúng.")


def _knowledge_catalog_rows():
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT source_file, subject, content_type, lesson, lesson_pages,
                       topic, topic_pages, question_pages, answer_pages, namespace
                FROM knowledge_documents
                UNION ALL
                SELECT source_file, subject, content_type, lesson, NULL::VARCHAR AS lesson_pages,
                       NULL::VARCHAR AS topic, NULL::VARCHAR AS topic_pages,
                       NULL::VARCHAR AS question_pages, NULL::VARCHAR AS answer_pages,
                       '__default__'::VARCHAR AS namespace
                FROM curriculum_lessons
                WHERE status='PUBLISHED'
                ORDER BY source_file, subject, content_type, lesson, topic
            """)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _knowledge_catalog_tree(rows):
    by_source = {}; tree = []
    for row in rows:
        sf = str(row.get("source_file") or "").strip()
        if not sf: continue
        doc = by_source.get(sf)
        if doc is None:
            doc = {"source_file":sf,"subject":row.get("subject") or "","namespace":row.get("namespace") or "__default__","content_types":{}}
            by_source[sf]=doc; tree.append(doc)
        ct=str(row.get("content_type") or "Từ vựng").strip() or "Từ vựng"
        ctn=doc["content_types"].setdefault(ct,{})
        lesson=str(row.get("lesson") or "").strip()
        if not lesson: continue
        ln=ctn.setdefault(lesson,{"lesson":lesson,"lesson_pages":row.get("lesson_pages"),"topics":{}})
        topic=str(row.get("topic") or "").strip()
        if topic:
            ln["topics"][topic]={"topic":topic,"topic_pages":row.get("topic_pages"),"question_pages":row.get("question_pages"),"answer_pages":row.get("answer_pages")}
    for doc in tree:
        doc["content_types"]=[{"content_type":ct,"lessons":[{**ln,"topics":list(ln["topics"].values())} for ln in lessons.values()]} for ct,lessons in doc["content_types"].items()]
    return tree


@app.get("/admin/api/knowledge/catalog")
def admin_knowledge_catalog(password: str):
    check_admin(password)
    rows=_knowledge_catalog_rows()
    tree=_knowledge_catalog_tree(rows)
    published_curriculum=sum(1 for r in rows if str(r.get("content_type") or "").strip()=="Giáo trình" and r.get("lesson"))
    resp=JSONResponse({"success":True,"documents":tree,"raw_count":len(rows),"published_curriculum_count":published_curriculum,"server_time":datetime.now(timezone.utc).isoformat()})
    resp.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"
    return resp


def _pinecone_scope_filter(source_file: str, content_type: str | None = None,
                            lesson: str | None = None, topic: str | None = None):
    """Build an exact Pinecone metadata identity filter.

    Deletion scope is hierarchical and intentionally NEVER uses semantic similarity:
      DOCUMENT = source_file
      LESSON   = source_file + content_type + lesson
      TOPIC    = source_file + content_type + lesson + topic
    """
    filt = {"source_file": {"$eq": source_file}}
    if content_type:
        filt["content_type"] = {"$eq": content_type}
    if lesson:
        filt["lesson"] = {"$eq": lesson}
    if topic:
        filt["topic"] = {"$eq": topic}
    return filt


def _pinecone_scope_candidates(namespace: str, pine_filter: dict, *, top_k: int = 10000):
    """Return every currently query-visible vector ID in the exact scope.

    Pinecone supports metadata-filtered delete directly, but for admin deletes we
    want a stronger safety/verification path: enumerate matching IDs, defensively
    re-check their metadata, then delete those exact IDs in batches.
    """
    if not index:
        raise RuntimeError("Pinecone chưa được khởi tạo.")
    probe = index.query(
        vector=[0.0] * 768,
        top_k=max(1, min(int(top_k), 10000)),
        include_metadata=True,
        namespace=namespace,
        filter=pine_filter,
    )
    return list(getattr(probe, "matches", None) or [])


def _pinecone_scope_matches_metadata(md: dict, expected: dict) -> bool:
    """Defensive exact identity check before an ID is deleted."""
    def norm(v):
        return str(v or "").strip().casefold()
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        if norm(md.get(key)) != norm(expected_value):
            return False
    return True


def _delete_pinecone_scope_exact(namespace: str, pine_filter: dict, *,
                                 source_file: str, content_type: str | None = None,
                                 lesson: str | None = None, topic: str | None = None,
                                 attempts: int = 8):
    """Delete an exact scope by enumerated IDs, then verify until clean.

    We intentionally do not trust one filter-delete call as proof of removal.
    Each pass:
      1) query all currently visible matches for the exact filter;
      2) re-check every candidate's identity in Python;
      3) delete exact IDs in batches of <=1000;
      4) wait for Pinecone eventual consistency and query again.
    """
    expected = {
        "source_file": source_file,
        "content_type": content_type,
        "lesson": lesson,
        "topic": topic,
    }
    deleted_ids = []
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            matches = _pinecone_scope_candidates(namespace, pine_filter, top_k=10000)
            safe_ids = []
            unsafe_ids = []
            for m in matches:
                md = getattr(m, "metadata", None) or {}
                mid = str(getattr(m, "id", "") or "").strip()
                if not mid:
                    continue
                if _pinecone_scope_matches_metadata(md, expected):
                    safe_ids.append(mid)
                else:
                    unsafe_ids.append(mid)

            if unsafe_ids:
                raise RuntimeError(
                    "Pinecone returned vectors outside the requested exact identity scope; "
                    f"refusing to delete {len(unsafe_ids)} candidate(s)."
                )

            # Nothing left in the exact scope -> success.
            if not safe_ids:
                return {
                    "namespace": namespace,
                    "attempts": attempt,
                    "verified_empty": True,
                    "deleted_ids": deleted_ids,
                }

            # Pinecone documents up to 1000 IDs per delete-by-ID request.
            for i in range(0, len(safe_ids), 1000):
                batch = safe_ids[i:i + 1000]
                index.delete(ids=batch, namespace=namespace)
                deleted_ids.extend(batch)

            # Allow the index to converge, then loop and verify again.
            time.sleep(min(8.0, 0.75 * attempt))
        except Exception as exc:
            last_error = exc
            break

    if last_error:
        raise RuntimeError(
            f"Pinecone strict delete/verify failed: {type(last_error).__name__}: {last_error}"
        )
    raise RuntimeError(
        f"Pinecone vẫn còn vector trong exact scope sau {attempts} vòng xóa/kiểm tra."
    )

def _delete_knowledge_scope(*, source_file: str, content_type: str | None = None,
                             lesson: str | None = None, topic: str | None = None):
    source_file = os.path.basename(str(source_file or "").strip())
    content_type = str(content_type or "").strip() or None
    lesson = str(lesson or "").strip() or None
    topic = str(topic or "").strip() or None
    if not source_file:
        raise HTTPException(400, "source_file là bắt buộc.")
    if lesson and not content_type:
        raise HTTPException(400, "Xóa Bài học cần có Loại nội dung.")
    if topic and (not lesson or not content_type):
        raise HTTPException(400, "Xóa Chủ đề cần có Loại nội dung và Bài học.")
    if not index:
        raise HTTPException(500, "Pinecone chưa được khởi tạo.")

    # Normalize the requested values before any destructive operation.
    where = ["source_file=%s"]
    params = [source_file]
    for col, val in (("content_type", content_type), ("lesson", lesson), ("topic", topic)):
        if val:
            where.append(f"lower(trim({col}))=lower(trim(%s))")
            params.append(val)
    ws = " AND ".join(where)

    conn = db()
    image_keys = []
    namespaces = []
    matching_lessons = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT DISTINCT namespace FROM knowledge_documents WHERE {ws}",
                tuple(params),
            )
            namespaces = [str(r.get("namespace") or "__default__") for r in (cur.fetchall() or [])]

            # If the DB row was already removed in a previous partial attempt,
            # the production uploader still uses __default__; use it as the safe
            # fallback so an orphan Pinecone scope can be cleaned explicitly.
            if not namespaces:
                namespaces = ["__default__"]

            cur.execute(
                f"""SELECT DISTINCT lesson,content_type FROM knowledge_documents WHERE {ws}
                    UNION
                    SELECT DISTINCT lesson,content_type FROM curriculum_lessons WHERE {ws} AND status='PUBLISHED'""",
                tuple(params) * 2,
            )
            matching_lessons = [dict(r) for r in (cur.fetchall() or [])]

            cur.execute(
                f"SELECT DISTINCT image_key FROM knowledge_images WHERE {ws} "
                "AND image_key IS NOT NULL AND TRIM(image_key)<>''",
                tuple(params),
            )
            image_keys = [str(r.get("image_key") or "").strip() for r in (cur.fetchall() or []) if r.get("image_key")]

        pine_filter = _pinecone_scope_filter(
            source_file,
            content_type=content_type,
            lesson=lesson,
            topic=topic,
        )

        pinecone_results = []
        # IMPORTANT: Pinecone is deleted AND verified BEFORE PostgreSQL is mutated.
        # If verification fails, the DB catalog/cache remains intact so we never
        # create the opposite inconsistency (DB says gone while vectors remain).
        for ns in namespaces:
            pinecone_results.append(_delete_pinecone_scope_exact(
                ns, pine_filter,
                source_file=source_file,
                content_type=content_type,
                lesson=lesson,
                topic=topic,
            ))

        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM knowledge_documents WHERE {ws}", tuple(params))
            deleted_documents = cur.rowcount

            cur.execute(f"DELETE FROM curriculum_lessons WHERE {ws}", tuple(params))
            deleted_curriculum_lessons = cur.rowcount

            cur.execute(f"DELETE FROM knowledge_lesson_cache WHERE {ws}", tuple(params))
            deleted_lesson_cache = cur.rowcount

            cur.execute(f"DELETE FROM knowledge_vision_cache WHERE {ws}", tuple(params))
            deleted_vision_cache = cur.rowcount

            cur.execute(f"DELETE FROM knowledge_images WHERE {ws}", tuple(params))
            deleted_images = cur.rowcount

            # Assets are document-level. Remove them only when the entire source
            # file no longer has any catalog rows; never remove a shared asset while
            # another lesson/topic from the same PDF remains.
            cur.execute(
                "SELECT COUNT(*) AS row_count FROM knowledge_documents WHERE source_file=%s",
                (source_file,),
            )
            row = cur.fetchone()
            row_count = int((row[0] if isinstance(row, tuple) else row.get("row_count")) or 0)
            if row_count == 0:
                cur.execute("DELETE FROM knowledge_assets WHERE source_file=%s", (source_file,))
                deleted_assets = cur.rowcount
            else:
                deleted_assets = 0

            # Close only sessions that belong to the exact lesson(s) being removed.
            # A topic deletion does not accidentally terminate another lesson.
            for lr in matching_lessons:
                ls = str(lr.get("lesson") or "").strip()
                ct = str(lr.get("content_type") or "").strip()
                if not ls:
                    continue
                cur.execute(
                    """
                    UPDATE user_learning_state SET
                        study_session_active=FALSE,
                        study_session_content_type=NULL,
                        study_session_course=NULL,
                        study_session_lesson=NULL,
                        study_session_topic=NULL,
                        study_session_started_at=NULL,
                        study_end_prompt_pending=FALSE,
                        curriculum_step=0,
                        curriculum_waiting='continue',
                        curriculum_exercise_answered=FALSE,
                        curriculum_global_exercise_question='',
                        curriculum_global_exercise_evidence='',
                        curriculum_summary_notes='',
                        curriculum_intro_history='',
                        curriculum_intro_b0b1_history='',
                        curriculum_global_exercise_result=''
                    WHERE lower(trim(coalesce(study_session_content_type,'')))=lower(trim(%s))
                      AND lower(trim(coalesce(study_session_lesson,'')))=lower(trim(%s))
                      AND lower(trim(coalesce(study_session_lesson,'')))<>''
                    """,
                    (ct, ls),
                )
        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(500, f"Lỗi xóa Knowledge Base: {type(exc).__name__}: {exc}")
    finally:
        conn.close()

    _invalidate_catalog_cache()

    # B2 deletion is conservative. A topic/lesson may reference an image key that
    # is shared elsewhere, so only delete physical objects automatically when the
    # entire source document is removed.
    b2_deleted = 0
    if not content_type and not lesson and not topic:
        if b2_delete_key(f"pdf/{re.sub(r'[^A-Za-z0-9_.-]+','_', source_file)}"):
            b2_deleted += 1
        for key in image_keys:
            if b2_delete_key(key):
                b2_deleted += 1

    return {
        "source_file": source_file,
        "content_type": content_type,
        "lesson": lesson,
        "topic": topic,
        "deleted_documents": deleted_documents,
        "deleted_curriculum_lessons": deleted_curriculum_lessons,
        "deleted_lesson_cache": deleted_lesson_cache,
        "deleted_vision_cache": deleted_vision_cache,
        "deleted_images": deleted_images,
        "deleted_assets": deleted_assets,
        "b2_deleted": b2_deleted,
        "namespaces": namespaces,
        "pinecone_filter": pine_filter,
        "pinecone_verified": pinecone_results,
    }


@app.post("/admin/api/knowledge/pinecone-probe")
def admin_knowledge_pinecone_probe(payload: dict):
    """Admin-only probe for cleaning Pinecone vectors that have become orphaned from PostgreSQL catalog."""
    check_admin(str(payload.get("password") or ""))
    source_file = os.path.basename(str(payload.get("source_file") or "").strip())
    content_type = str(payload.get("content_type") or "").strip() or None
    lesson = str(payload.get("lesson") or "").strip() or None
    topic = str(payload.get("topic") or "").strip() or None
    if not source_file:
        raise HTTPException(400, "source_file là bắt buộc.")
    if lesson and not content_type:
        raise HTTPException(400, "Probe Bài học cần có Loại nội dung.")
    if topic and (not lesson or not content_type):
        raise HTTPException(400, "Probe Chủ đề cần có Loại nội dung và Bài học.")
    pine_filter = _pinecone_scope_filter(source_file, content_type, lesson, topic)
    namespaces = ["__default__"]
    rows = []
    for ns in namespaces:
        matches = _pinecone_scope_candidates(ns, pine_filter, top_k=10000)
        for m in matches:
            rows.append({
                "id": str(getattr(m, "id", "") or ""),
                "score": float(getattr(m, "score", 0) or 0),
                "metadata": dict(getattr(m, "metadata", None) or {}),
            })
    return {"success": True, "namespace": namespaces, "pinecone_filter": pine_filter, "count": len(rows), "matches": rows[:200]}


@app.post("/admin/api/knowledge/delete")
def admin_knowledge_delete(payload: dict):
    check_admin(str(payload.get("password") or ""))
    return {"success":True,"message":"Đã xóa Knowledge Base scope.","result":_delete_knowledge_scope(source_file=payload.get("source_file"),content_type=payload.get("content_type"),lesson=payload.get("lesson"),topic=payload.get("topic"))}




def init_curriculum_db():
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS curriculum_drafts (
                id BIGSERIAL PRIMARY KEY, source_file VARCHAR(500) NOT NULL, subject VARCHAR(255) NOT NULL,
                content_type VARCHAR(30) NOT NULL, lesson VARCHAR(255) NOT NULL, status VARCHAR(30) NOT NULL DEFAULT 'AI_DRAFT',
                version INTEGER NOT NULL DEFAULT 1, draft_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(source_file, content_type, lesson, version)
            );""")
            cur.execute("""CREATE TABLE IF NOT EXISTS curriculum_lessons (
                id BIGSERIAL PRIMARY KEY, draft_id BIGINT, source_file VARCHAR(500) NOT NULL, subject VARCHAR(255) NOT NULL,
                content_type VARCHAR(30) NOT NULL, lesson VARCHAR(255) NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'PUBLISHED',
                version INTEGER NOT NULL DEFAULT 1, raw_source_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(source_file, content_type, lesson, version)
            );""")
            cur.execute("""CREATE TABLE IF NOT EXISTS curriculum_steps (
                id BIGSERIAL PRIMARY KEY, lesson_id BIGINT NOT NULL REFERENCES curriculum_lessons(id) ON DELETE CASCADE,
                step_code VARCHAR(20) NOT NULL, step_order INTEGER NOT NULL, title VARCHAR(500) NOT NULL DEFAULT '',
                step_type VARCHAR(50) NOT NULL DEFAULT 'lesson', content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(lesson_id, step_code)
            );""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_curriculum_lessons_scope ON curriculum_lessons(content_type, lesson, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_curriculum_steps_lesson ON curriculum_steps(lesson_id, step_order);")
        conn.commit()
    finally:
        conn.close()


CURRICULUM_STEP_RULES = {
    'Giáo trình': [
        {'code':'B0','title':'Giới thiệu mục đích bài học','type':'intro'},
        {'code':'B1','title':'Giới thiệu từ vựng và ngữ pháp, câu hỏi toàn bài nếu có','type':'overview'},
        {'code':'SECTION','title':'Các section và câu hỏi từng section nếu có','type':'section'},
        {'code':'FINAL','title':'Tổng kết','type':'summary'},
    ],
    'Từ vựng': [
        {'code':'B0','title':'Giới thiệu từ vựng: cách đọc, nghĩa, cách viết','type':'vocabulary'},
        {'code':'B1','title':'Một số ví dụ','type':'examples'},
        {'code':'B2','title':'Một số bài tập','type':'exercise'},
    ],
    'Ngữ pháp': [
        {'code':'B0','title':'Giới thiệu cấu trúc ngữ pháp','type':'grammar'},
        {'code':'B1','title':'Một số ví dụ','type':'examples'},
        {'code':'B2','title':'Một số bài tập','type':'exercise'},
    ],
    'Bài tập': [
        {'code':'B0','title':'Giới thiệu bài tập','type':'exercise_intro'},
        {'code':'B1','title':'Đánh giá câu trả lời user','type':'evaluation'},
        {'code':'B2','title':'Đáp án','type':'answer'},
        {'code':'B3','title':'Bài tập tương tự','type':'similar_exercise'},
    ],
    'Truyện đọc': [
        {'code':'B0','title':'Nội dung truyện','type':'story'},
        {'code':'B1','title':'Bản dịch tiếng Việt','type':'translation'},
        {'code':'B2','title':'Từ vựng','type':'vocabulary'},
        {'code':'B3','title':'Ngữ pháp','type':'grammar'},
    ],
}

def _normalize_curriculum_steps(content_type, steps, source_digest):
    raw=[s for s in (steps or []) if isinstance(s,dict)]
    if content_type == 'Giáo trình':
        b0=next((dict(s) for s in raw if str(s.get('code') or '').upper()=='B0'), {'code':'B0','title':'Giới thiệu mục đích bài học','type':'intro'})
        b1=next((dict(s) for s in raw if str(s.get('code') or '').upper()=='B1'), {'code':'B1','title':'Giới thiệu từ vựng và ngữ pháp, câu hỏi toàn bài nếu có','type':'overview'})
        final=next((dict(s) for s in raw if str(s.get('code') or '').upper() in {'FINAL','SUMMARY'}), {'code':'FINAL','title':'Tổng kết','type':'summary'})
        sections=[dict(s) for s in raw if str(s.get('code') or '').upper() not in {'B0','B1','FINAL','SUMMARY'}]
        for i,s in enumerate(sections,1):
            s['code']=f'B{i+1}'
            s.setdefault('type','section')
            s.setdefault('title',f'Section {i}')
        return [b0,b1,*sections,final]
    rules=CURRICULUM_STEP_RULES.get(content_type) or []
    out=[]
    for rule in rules:
        code=rule['code']; found=next((dict(s) for s in raw if str(s.get('code') or '').upper()==code.upper()),{})
        found['code']=code; found.setdefault('title',rule['title']); found.setdefault('type',rule['type']); out.append(found)
    return out

def reindex_curriculum_draft_steps_safe(content_type, steps):
    """Reindex draft step codes without touching any published lesson.

    Giáo trình keeps B0, B1 and FINAL as structural anchors; only section
    steps are renumbered B2, B3, ... . Other content types are re-numbered
    by current draft order so deleting a step shifts following steps up.
    This function operates on draft_json only.
    """
    raw=[dict(x) for x in (steps or []) if isinstance(x,dict)]
    ct=str(content_type or '').strip()
    if ct == 'Giáo trình':
        b0=next((x for x in raw if str(x.get('code') or '').upper()=='B0'), None)
        b1=next((x for x in raw if str(x.get('code') or '').upper()=='B1'), None)
        final=next((x for x in raw if str(x.get('code') or '').upper() in {'FINAL','SUMMARY'}), None)
        sections=[x for x in raw if str(x.get('code') or '').upper() not in {'B0','B1','FINAL','SUMMARY'}]
        out=[]
        if b0 is not None:
            b0['code']='B0'; out.append(b0)
        if b1 is not None:
            b1['code']='B1'; out.append(b1)
        for i,x in enumerate(sections,start=2):
            x['code']=f'B{i}'; out.append(x)
        if final is not None:
            final['code']='FINAL'; out.append(final)
        return out
    out=[]
    for i,x in enumerate(raw):
        x['code']=f'B{i}'; out.append(x)
    return out

# Backward-compatible alias kept after the safe helper definition.
# Routes should call reindex_curriculum_draft_steps_safe directly.
_reindex_curriculum_draft_steps = reindex_curriculum_draft_steps_safe

def _parse_curriculum_page_ranges(value, max_page):
    """Parse page ranges like '1-3,5,7-8' into sorted unique PDF page numbers."""
    raw=str(value or "").strip()
    if not raw:
        raise ValueError("Số trang là bắt buộc, ví dụ 7-8 hoặc 7,9-10.")
    pages=set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a,b=part.split("-",1)
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Khoảng trang không hợp lệ: {part}")
            a,b=int(a),int(b)
            if a<1 or b<a or b>int(max_page):
                raise ValueError(f"Khoảng trang ngoài phạm vi PDF: {part} (1-{max_page})")
            pages.update(range(a,b+1))
        else:
            if not part.isdigit():
                raise ValueError(f"Trang không hợp lệ: {part}")
            n=int(part)
            if n<1 or n>int(max_page):
                raise ValueError(f"Trang ngoài phạm vi PDF: {n} (1-{max_page})")
            pages.add(n)
    if not pages:
        raise ValueError("Không có trang hợp lệ.")
    return sorted(pages)

def _curriculum_page_range_label(pages):
    vals=[int(x) for x in (pages or [])]
    if not vals:
        return ""
    out=[]; start=prev=vals[0]
    for n in vals[1:]:
        if n==prev+1:
            prev=n
        else:
            out.append(str(start) if start==prev else f"{start}-{prev}")
            start=prev=n
    out.append(str(start) if start==prev else f"{start}-{prev}")
    return ",".join(out)

def _curriculum_source_digest(pages):
    parts=[]
    for p in pages:
        text=str(p.get('text') or '').strip()
        if text:
            parts.append(f"[TRANG {p.get('page')}]\n{text[:6000]}")
        for img in (p.get('images') or []):
            vision=img.get('vision') or {}
            desc='; '.join(str(vision.get(k) or '').strip() for k in ('term','reading','meaning','associated_text','description','explanation','caption') if str(vision.get(k) or '').strip())
            image_key=str(img.get('image_key') or '').strip()
            image_url=str(img.get('image_url') or '').strip()
            image_line=f"[ẢNH NGUỒN trang {p.get('page')}] image_key={image_key} image_url={image_url}"
            if desc:
                image_line += f" vision={desc}"
            if image_line.strip():
                parts.append(image_line)
    return '\n\n'.join(parts)[:50000]

def _resolve_curriculum_step_images(step_content, pages):
    """Normalize step image references against the uploaded page/image inventory.

    AI may return image_key only; the Admin must still see the actual B2 URL.
    We never invent URLs: an image can only resolve when its key exists in the
    original extracted pages.
    """
    content=dict(step_content or {})
    raw=list(content.get('images') or [])
    inventory={}
    for page in pages or []:
        for img in page.get('images') or []:
            key=str(img.get('image_key') or '').strip()
            if key:
                inventory[key]={
                    'image_key':key,
                    'image_url':str(img.get('image_url') or '').strip(),
                    'page':page.get('page'),
                    'vision':img.get('vision') or {},
                }
    resolved=[]
    seen=set()
    for item in raw:
        if not isinstance(item,dict):
            continue
        key=str(item.get('image_key') or item.get('key') or '').strip()
        base=inventory.get(key)
        if not base:
            # Safe visual fallback: map an AI-selected image to the source
            # inventory only when its caption exactly matches a stored Vision
            # caption/description/explanation. Never map by vector similarity.
            wanted=str(item.get('caption') or '').strip().casefold()
            if wanted:
                for cand in inventory.values():
                    vision=cand.get('vision') or {}
                    candidates=[vision.get('caption'),vision.get('description'),vision.get('explanation'),vision.get('associated_text')]
                    if any(str(v or '').strip().casefold()==wanted for v in candidates if str(v or '').strip()):
                        base=cand; key=str(cand.get('image_key') or '').strip(); break
        if base:
            row=dict(item)
            row['image_key']=key or str(base.get('image_key') or '').strip()
            row['image_url']=str(row.get('image_url') or base.get('image_url') or '').strip()
            row['page']=row.get('page') or base.get('page')
            if not row.get('caption'):
                vision=base.get('vision') or {}
                row['caption']=str(vision.get('caption') or vision.get('description') or vision.get('explanation') or '').strip()
        else:
            row=dict(item)
        if key and key not in seen:
            seen.add(key); resolved.append(row)
    content['images']=resolved
    return content


def _curriculum_ai_json(prompt, operation):
    if not gemini:
        raise HTTPException(500, 'Gemini chưa được khởi tạo.')
    response=gemini.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1, thinking_config=types.ThinkingConfig(thinking_level='low'), response_mime_type='application/json')
    )
    _log_gemini_usage(response, operation=operation)
    return _parse_gemini_json(response.text or '{}')

def _curriculum_step_plan(content_type, source_digest):
    rule=json.dumps(CURRICULUM_STEP_RULES.get(content_type) or [], ensure_ascii=False)
    prompt=f"""Bạn là AI biên soạn giáo trình cho Doraemon.\nLoại nội dung: {content_type}\n\nQUY TẮC BƯỚC BẮT BUỘC:\n{rule}\n\nNguồn tài liệu dưới đây là nguồn sự thật. Không được tạo ra kiến thức không có trong nguồn. Với Giáo trình, phải biến marker SECTION thành số section thực tế được tìm thấy; luôn giữ B0, B1 và bước cuối FINAL. Các loại khác phải giữ đúng số bước và mã bước đã quy định.\n\nNGUỒN:\n{source_digest}\n\nTrả JSON: {{\"steps\":[{{\"code\":\"B0\",\"title\":\"...\",\"instruction\":\"...\"}}]}}"""
    data=_curriculum_ai_json(prompt, 'curriculum_step_plan')
    # Gemini đôi khi trả thẳng JSON array dù schema yêu cầu object.
    if isinstance(data, list):
        steps=data
        print(f"[CURRICULUM STEP PLAN NORMALIZE] content_type={content_type!r} response_shape=list steps={len(steps)}")
    elif isinstance(data, dict):
        raw_steps=data.get('steps')
        steps=raw_steps if isinstance(raw_steps,list) else []
    else:
        steps=[]
    if not steps: raise HTTPException(500,'AI không tạo được số bước.')
    return steps

def _curriculum_generate_step(content_type, lesson, step, source_digest):
    extra_rules=""
    if str(content_type or "").strip() == "Từ vựng":
        extra_rules="""\n\nQUY TẮC BẮT BUỘC CHO TỪ VỰNG:
- Mỗi item phải có writing/word, reading/kana, pronunciation và meaning khi nguồn có.
- pronunciation phải là cách phát âm dựa trên reading/kana ĐÚNG TỪ NGUỒN; không tự đoán IPA/romaji.
- Nếu nguồn có reading thì tuyệt đối không được để thiếu reading/pronunciation.
- Ưu tiên schema item: {{"writing":"...","reading":"...","pronunciation":"...","meaning":"...","example":"..."}}.
- Nếu nguồn thật sự không có reading thì để reading/pronunciation rỗng, không bịa.
"""
    elif str(content_type or "").strip() == "Giáo trình":
        extra_rules="""\n\nQUY TẮC KHI BƯỚC NÀY DẠY TỪ VỰNG:
- Mọi từ mới phải kèm chữ Nhật + cách đọc/kana + phát âm dựa trên reading trong nguồn + nghĩa tiếng Việt nếu nguồn có.
- Không chỉ liệt kê nghĩa và không tự đoán reading/phát âm.
"""
    elif str(content_type or "").strip() == "Truyện đọc":
        extra_rules="""\n\nQUY TẮC BẮT BUỘC CHO TRUYỆN ĐỌC:
- B0 = NỘI DUNG TRUYỆN GỐC. Phải giữ nguyên tiếng Nhật từ nguồn, không dịch sang tiếng Việt, không viết lại theo ý mình, không tóm tắt và không diễn giải thay cho bản gốc.
- Với B0, trường content phải ưu tiên chép nguyên văn phần tiếng Nhật của truyện từ nguồn. Giữ nguyên câu, đoạn, dấu câu và thứ tự nội dung; chỉ sửa lỗi OCR rõ ràng khi có bằng chứng trực tiếp từ nguồn.
- Tuyệt đối không đưa bản dịch tiếng Việt vào B0. Bản dịch chỉ thuộc B1.
- B1 = BẢN DỊCH TIẾNG VIỆT của chính B0; không được thay đổi nội dung B0.
- B2 = TỪ VỰNG lấy từ chính B0, gồm writing/word, reading/kana, pronunciation và meaning tiếng Việt khi nguồn có; không tự đoán reading/phát âm.
- B3 = NGỮ PHÁP dựa trên chính B0, không thay nội dung truyện gốc.
- Nếu nguồn không có bản tiếng Nhật rõ ràng cho một đoạn, không tự sáng tác lại; giữ nguyên dữ liệu nguồn và để phần thiếu rỗng hoặc ghi rõ không xác định.
"""
    prompt=f"""Bạn đang soạn nội dung cho Doraemon.\nLoại nội dung: {content_type}\nBài học: {lesson}\nBước: {step.get('code')} - {step.get('title')}\n\nCHỈ DÙNG THÔNG TIN CÓ TRONG NGUỒN. Có thể sắp xếp, diễn giải và rút gọn, nhưng không được bịa dữ kiện mới. Phải đưa cả text từ bài học và tri thức OCR/Vision phù hợp vào content. Nếu nguồn không có dữ kiện cho một trường, để chuỗi rỗng hoặc mảng rỗng.{extra_rules}\n\nQUY TẮC ƯU TIÊN: nếu loại nội dung là Truyện đọc và bước là B0, tuyệt đối ưu tiên văn bản tiếng Nhật nguyên gốc trong NGUỒN; không được chuyển ngữ sang tiếng Việt ở B0.\n\nTrả JSON với schema: {{"title":"...","content":"...","source_refs":[{{"page":1,"reason":"..."}}],"images":[{{"image_url":"...","image_key":"...","caption":"..."}}],"items":[]}}\n\nNGUỒN:\n{source_digest}"""
    data=_curriculum_ai_json(prompt, f"curriculum_step_{step.get('code','X')}")
    if isinstance(data, list):
        data=next((x for x in data if isinstance(x,dict)), {})
        print(f"[CURRICULUM STEP CONTENT NORMALIZE] content_type={content_type!r} step={step.get('code')} response_shape=list")
    if not isinstance(data, dict):
        data={}
    data.setdefault('title', step.get('title') or '')
    data.setdefault('content', '')
    data.setdefault('items', [])
    data.setdefault('source_refs', [])
    data.setdefault('images', [])
    return data

@app.post('/admin/api/curriculum/draft-upload')
async def admin_curriculum_draft_upload(
    password: str = Form(''),
    file: UploadFile = File(...),
    subject: str = Form(''),
    content_type: str = Form(''),
    lesson: str = Form(''),
    metadata_json: str = Form('[]'),
    articles_json: str = Form('[]'),
):
    """Create one or many AI curriculum drafts from selected page ranges only.

    A single PDF can define multiple lessons, each with its own page range. Pages
    not belonging to any configured lesson are never processed by OCR/Vision,
    never saved to B2, and never included in the AI source digest.
    """
    check_admin(password)
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400,'Vui lòng chọn file PDF.')
    if not gemini:
        raise HTTPException(500,'GEMINI_API_KEY chưa được cấu hình.')
    if not b2_ready():
        raise HTTPException(500,'Backblaze B2 chưa được cấu hình. AI Curriculum Studio cần B2 để lưu ảnh nguồn.')
    subject=subject.strip()
    if not subject:
        raise HTTPException(400,'Môn học là bắt buộc.')

    # Backward compatible single-lesson configuration.
    configs=[]
    try:
        parsed=json.loads(articles_json or '[]')
        if isinstance(parsed,list):
            configs=[x for x in parsed if isinstance(x,dict)]
    except Exception:
        configs=[]
    if not configs:
        content_type=_normalize_content_type(content_type)
        single_lesson=str(lesson or '').strip()
        if not single_lesson:
            raise HTTPException(400,'Tên bài học là bắt buộc.')
        configs=[{'content_type':content_type,'lesson':single_lesson,'pages':str('') if False else ''}]
    else:
        normalized=[]
        for idx,cfg in enumerate(configs,1):
            ct=_normalize_content_type(cfg.get('content_type'))
            ls=str(cfg.get('lesson') or '').strip()
            pg=str(cfg.get('pages') or cfg.get('page_ranges') or '').strip()
            if not ls:
                raise HTTPException(400,f'Bài #{idx}: Tên bài học là bắt buộc.')
            normalized.append({'content_type':ct,'lesson':ls,'pages':pg})
        configs=normalized

    source_file=os.path.basename(file.filename)
    temp_pdf_path=None
    try:
        with tempfile.NamedTemporaryFile(prefix='doraemon_curriculum_',suffix='.pdf',delete=False) as tf:
            temp_pdf_path=tf.name
            while True:
                chunk=await file.read(1024*1024)
                if not chunk: break
                tf.write(chunk)
        reader=PdfReader(temp_pdf_path)
        total_pages=len(reader.pages)
        if total_pages<=0:
            raise HTTPException(400,'PDF không có trang.')

        # Parse the selected page ranges before touching OCR/Vision.
        for idx,cfg in enumerate(configs,1):
            if not str(cfg.get('pages') or '').strip():
                raise HTTPException(400,f'Bài #{idx} ({cfg.get("lesson") or ""}): phải nhập số trang, ví dụ 7-8.')
            try:
                cfg['selected_pages']=_parse_curriculum_page_ranges(cfg['pages'], total_pages)
            except ValueError as exc:
                raise HTTPException(400,f'Bài #{idx} ({cfg.get("lesson") or ""}): {exc}')
            cfg['pages_label']=_curriculum_page_range_label(cfg['selected_pages'])

        # Do not allow overlapping configured pages. One PDF page should have one
        # curriculum owner, otherwise B2/image provenance would become ambiguous.
        owners={}
        overlaps=[]
        for idx,cfg in enumerate(configs,1):
            for pg in cfg['selected_pages']:
                if pg in owners:
                    overlaps.append(f"trang {pg} (bài #{owners[pg]} và #{idx})")
                else:
                    owners[pg]=idx
        if overlaps:
            raise HTTPException(400,'Phạm vi trang bị chồng lấn: '+', '.join(overlaps)+'. Hãy tách trang cho từng bài.')

        records_meta=normalize_kb_records(metadata_json,total_pages)
        results=[]
        for idx,cfg in enumerate(configs,1):
            ct=str(cfg['content_type']).strip()
            ls=str(cfg['lesson']).strip()
            selected_pages=cfg['selected_pages']
            page_texts,page_images,page_units=process_pdf_pages(
                temp_pdf_path, reader, records_meta, source_file, subject, selected_pages=selected_pages
            )
            pages=[]
            selected_set=set(int(x) for x in selected_pages)
            for page_no in selected_pages:
                imgs=[]
                for img in page_images.get(page_no,[]) or []:
                    vision={k:v for k,v in img.items() if k not in {'key','url','image_url'}}
                    key=str(img.get('key') or '')
                    if not key: continue
                    imgs.append({'image_key':key,'image_url':b2_url(key),'vision':vision})
                pages.append({'page':page_no,'text':page_texts.get(page_no,'')[:12000],'images':imgs})
            # Hard invariant: the AI Draft payload may contain ONLY configured pages.
            page_keys={int(pg.get('page')) for pg in pages if str(pg.get('page')).isdigit()}
            if page_keys != selected_set:
                raise HTTPException(500, f'Page-scope lỗi cho bài {ls}: expected={sorted(selected_set)} actual={sorted(page_keys)}')

            digest=_curriculum_source_digest(pages)
            normalized_steps=[]
            if ct == 'Truyện đọc':
                # Story has fixed steps. Skip the planning call and copy B0 from source directly.
                story_text = "\n\n".join(
                    str(pg.get('text') or '').strip()
                    for pg in pages
                    if str(pg.get('text') or '').strip()
                ).strip()
                story_b0 = {
                    'title': 'Nội dung truyện',
                    'content': story_text,
                    'source_refs': [
                        {'page': pg.get('page'), 'reason': 'Văn bản truyện gốc từ OCR/text nguồn'}
                        for pg in pages if pg.get('text')
                    ],
                    'images': [],
                    'items': [],
                }
                story_b0 = _resolve_curriculum_step_images(story_b0, pages)
                normalized_steps.append({'code':'B0','title':'Nội dung truyện','type':'story','content':story_b0})
                for code,title,step_type in [
                    ('B1','Bản dịch tiếng Việt','translation'),
                    ('B2','Từ vựng','vocabulary'),
                    ('B3','Ngữ pháp','grammar'),
                ]:
                    content=_curriculum_generate_step(ct,ls,{'code':code,'title':title},digest)
                    content=_resolve_curriculum_step_images(content,pages)
                    normalized_steps.append({'code':code,'title':title,'type':step_type,'content':content})
                print('[CURRICULUM STORY LOWTOKEN] step_plan_genai=0 B0_genai=0 B1_B2_B3_genai=1_each total_calls=3')
            else:
                plan=_normalize_curriculum_steps(ct,_curriculum_step_plan(ct,digest),digest)
                for st in plan:
                    code=str(st.get('code') or '').strip(); title=str(st.get('title') or '').strip()
                    if not code or not title: continue
                    content=_curriculum_generate_step(ct,ls,{'code':code,'title':title},digest)
                    content=_resolve_curriculum_step_images(content,pages)
                    normalized_steps.append({'code':code,'title':title,'type':st.get('type') or 'lesson','content':content})

            payload={
                'source_file':source_file,
                'subject':subject,
                'content_type':ct,
                'lesson':ls,
                'page_ranges':cfg['pages_label'],
                'selected_pages':selected_pages,
                'page_count':len(pages),
                'pages':pages,
                'steps':normalized_steps,
            }
            conn=db()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT COALESCE(MAX(version),0)+1 AS next_version FROM curriculum_drafts WHERE source_file=%s AND content_type=%s AND lesson=%s",(source_file,ct,ls))
                    version=int(cur.fetchone()['next_version'])
                    cur.execute("INSERT INTO curriculum_drafts(source_file,subject,content_type,lesson,status,version,draft_json) VALUES(%s,%s,%s,%s,'AI_DRAFT',%s,%s::jsonb) RETURNING id",(source_file,subject,ct,ls,version,json.dumps(payload,ensure_ascii=False)))
                    draft_id=int(cur.fetchone()['id'])
                conn.commit()
            finally:
                conn.close()
            results.append({
                'draft_id':draft_id,'status':'AI_DRAFT','version':version,
                'source_file':source_file,'subject':subject,'content_type':ct,'lesson':ls,
                'page_ranges':cfg['pages_label'],'selected_pages':selected_pages,
                'selected_page_count':len(selected_pages),
                'steps':normalized_steps,'pages':pages,'page_count':len(pages),
            })

        return {
            'success':True,
            'source_file':source_file,
            'subject':subject,
            'pdf_page_count':total_pages,
            'configured_articles':len(results),
            'selected_page_count':sum(int(x.get('selected_page_count') or 0) for x in results),
            'drafts':results,
            # Backward compatible single-result keys.
            **(results[0] if len(results)==1 else {'draft_id':results[0]['draft_id'] if results else None}),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500,f'Không tạo được AI Draft: {type(exc).__name__}: {exc}')
    finally:
        if temp_pdf_path:
            try: os.unlink(temp_pdf_path)
            except Exception: pass

@app.get('/admin/api/curriculum/drafts')
def admin_curriculum_drafts(password: str):
    check_admin(password); conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id,source_file,subject,content_type,lesson,status,version,created_at,updated_at
                           FROM curriculum_drafts
                           WHERE status <> 'PUBLISHED'
                           ORDER BY updated_at DESC, id DESC LIMIT 200""")
            rows=[dict(r) for r in cur.fetchall()]
            return {'success':True,'drafts':rows,'count':len(rows)}
    finally: conn.close()

@app.post('/admin/api/curriculum/drafts/{draft_id}/delete')
async def admin_curriculum_draft_delete_post(draft_id:int, password:str = '', payload:dict | None = None):
    """Robust Draft delete: accepts password from query OR JSON body."""
    # Prefer explicit query password; fall back to JSON body for older clients.
    pw = str(password or '').strip()
    if not pw and isinstance(payload, dict):
        pw = str(payload.get('password') or '').strip()
    check_admin(pw)
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id,status,source_file,content_type,lesson FROM curriculum_drafts WHERE id=%s FOR UPDATE",(draft_id,))
            row=cur.fetchone()
            if not row:
                raise HTTPException(404,'Draft không tồn tại.')
            status=str(row.get('status') or '').upper()
            if status == 'PUBLISHED':
                raise HTTPException(400,'Draft đã publish, không thể xóa khỏi danh sách Draft.')
            # Explicitly remove only this draft. Other drafts for the same PDF/lesson remain untouched.
            cur.execute("DELETE FROM curriculum_drafts WHERE id=%s AND status <> 'PUBLISHED' RETURNING id",(draft_id,))
            deleted=cur.fetchone()
            if not deleted:
                raise HTTPException(409,'Draft không còn ở trạng thái có thể xóa.')
        conn.commit()
    except HTTPException:
        conn.rollback(); raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(500, f'Xóa Draft thất bại: {type(exc).__name__}: {exc}')
    finally:
        conn.close()
    return {'success':True,'draft_id':draft_id,'message':'Đã xóa Draft.'}

@app.post('/admin/api/curriculum/drafts/{draft_id}/remove')
async def admin_curriculum_draft_remove(draft_id:int, password:str = '', request: Request | None = None):
    """Robust Draft delete endpoint: query password is preferred; JSON body is accepted as fallback."""
    payload=None
    if not str(password or '').strip() and request is not None:
        try:
            payload = await request.json()
        except Exception:
            payload = None
    return await admin_curriculum_draft_delete_post(draft_id=draft_id, password=password, payload=payload)

@app.get('/admin/api/curriculum/drafts/{draft_id}')
def admin_curriculum_draft_get(draft_id:int,password:str):
    check_admin(password); conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM curriculum_drafts WHERE id=%s',(draft_id,)); row=cur.fetchone()
            if not row: raise HTTPException(404,'Draft không tồn tại.')
            return dict(row)
    finally: conn.close()

@app.post('/admin/api/curriculum/drafts/{draft_id}')
def admin_curriculum_draft_save(draft_id:int,payload:dict):
    check_admin(str(payload.get('password') or '')); draft=payload.get('draft')
    if not isinstance(draft,dict): raise HTTPException(400,'draft phải là object.')
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT content_type FROM curriculum_drafts WHERE id=%s',(draft_id,)); row=cur.fetchone()
            if not row: raise HTTPException(404,'Draft không tồn tại.')
            ct=str(draft.get('content_type') or row.get('content_type') or '').strip()
            draft['steps']=reindex_curriculum_draft_steps_safe(ct,draft.get('steps') or [])
            draft_json_text=json.dumps(draft,ensure_ascii=False)
            print(f"[CURRICULUM DRAFT SAVE] draft_id={draft_id} steps={len(draft.get('steps') or [])} chars={len(draft_json_text)} content_fields={[str((st.get('content') or {}).get('content') or '')[:80] for st in (draft.get('steps') or [])]}")
            cur.execute("UPDATE curriculum_drafts SET draft_json=%s::jsonb,status='ADMIN_REVIEW',updated_at=NOW() WHERE id=%s",(draft_json_text,draft_id))
        conn.commit()
    finally: conn.close()
    return {'success':True,'draft_id':draft_id,'status':'ADMIN_REVIEW','steps':draft.get('steps') or []}


@app.delete('/admin/api/curriculum/drafts/{draft_id}')
def admin_curriculum_draft_delete(draft_id:int,password:str):
    check_admin(password)
    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status,source_file,content_type,lesson FROM curriculum_drafts WHERE id=%s",(draft_id,))
            row=cur.fetchone()
            if not row:
                raise HTTPException(404,'Draft không tồn tại.')
            if str(row[0] or '').upper() == 'PUBLISHED':
                raise HTTPException(400,'Draft đã publish, không thể xóa khỏi danh sách Draft.')
            cur.execute("DELETE FROM curriculum_drafts WHERE id=%s",(draft_id,))
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {'success':True,'draft_id':draft_id,'message':'Đã xóa Draft.'}


@app.post('/admin/api/curriculum/drafts/{draft_id}/delete-step')
def admin_curriculum_delete_step(draft_id:int,payload:dict):
    check_admin(str(payload.get('password') or ''))
    step_code=str(payload.get('step_code') or '').strip().upper()
    if not step_code: raise HTTPException(400,'step_code là bắt buộc.')
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT content_type,draft_json FROM curriculum_drafts WHERE id=%s',(draft_id,)); row=cur.fetchone()
            if not row: raise HTTPException(404,'Draft không tồn tại.')
            ct=str(row.get('content_type') or '').strip()
            draft=dict(row.get('draft_json') or {})
            steps=[dict(x) for x in (draft.get('steps') or []) if isinstance(x,dict)]
            target=next((x for x in steps if str(x.get('code') or '').strip().upper()==step_code),None)
            if target is None: raise HTTPException(404,f'Không tìm thấy bước {step_code}.')
            if ct == 'Giáo trình' and step_code in {'B0','B1','FINAL','SUMMARY'}:
                raise HTTPException(400,f'{step_code} là bước cấu trúc bắt buộc của Giáo trình, không thể xóa.')
            steps=[x for x in steps if str(x.get('code') or '').strip().upper()!=step_code]
            draft['steps']=reindex_curriculum_draft_steps_safe(ct,steps)
            cur.execute("UPDATE curriculum_drafts SET draft_json=%s::jsonb,status='ADMIN_REVIEW',updated_at=NOW() WHERE id=%s",(json.dumps(draft,ensure_ascii=False),draft_id))
        conn.commit()
    finally: conn.close()
    return {'success':True,'draft_id':draft_id,'steps':draft.get('steps') or []}


@app.post('/admin/api/curriculum/drafts/{draft_id}/regenerate-step')
def admin_curriculum_regenerate_step(draft_id:int,payload:dict):
    check_admin(str(payload.get('password') or '')); step_code=str(payload.get('step_code') or '').strip()
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT draft_json FROM curriculum_drafts WHERE id=%s',(draft_id,)); row=cur.fetchone()
        if not row: raise HTTPException(404,'Draft không tồn tại.')
        draft=dict(row['draft_json'] or {}); step=next((s for s in draft.get('steps',[]) if str(s.get('code'))==step_code),None)
        if not step: raise HTTPException(404,'Step không tồn tại.')
        digest=_curriculum_source_digest(draft.get('pages') or [])
        new_content=_curriculum_generate_step(str(draft.get('content_type') or ''),str(draft.get('lesson') or ''),step,digest)
        new_content=_resolve_curriculum_step_images(new_content, draft.get('pages') or [])
        step['content']=new_content; draft['steps']=[s if s is not step else step for s in draft['steps']]
        with conn.cursor() as cur: cur.execute("UPDATE curriculum_drafts SET draft_json=%s::jsonb,status='ADMIN_REVIEW',updated_at=NOW() WHERE id=%s",(json.dumps(draft,ensure_ascii=False),draft_id))
        conn.commit(); return {'success':True,'step':step}
    finally: conn.close()

@app.post('/admin/api/curriculum/drafts/{draft_id}/publish')
def admin_curriculum_publish(draft_id:int,payload:dict):
    check_admin(str(payload.get('password') or ''))
    client_draft = payload.get('draft')
    conn=db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM curriculum_drafts WHERE id=%s FOR UPDATE',(draft_id,)); dr=cur.fetchone()
            if not dr: raise HTTPException(404,'Draft không tồn tại.')
            if str(dr.get('status') or '').upper() == 'PUBLISHED':
                raise HTTPException(400,'Draft này đã được publish.')

            # CRITICAL: publish the exact draft currently present in the Admin editor.
            # The browser sends the fully collected draft so a last-second text edit
            # cannot be lost between Save and Publish. Server-side normalization still
            # runs before anything is persisted/published.
            draft = dict(client_draft) if isinstance(client_draft, dict) else dict(dr.get('draft_json') or {})
            ct = str(draft.get('content_type') or dr.get('content_type') or '').strip()
            steps = draft.get('steps') or []
            draft['steps'] = reindex_curriculum_draft_steps_safe(ct, steps)
            print(f"[CURRICULUM PUBLISH INPUT] draft_id={draft_id} steps={len(draft.get('steps') or [])} content_fields={[str((st.get('content') or {}).get('content') or '')[:120] for st in (draft.get('steps') or [])]}")
            source_file=str(draft.get('source_file') or dr['source_file']).strip()
            lesson=str(draft.get('lesson') or dr['lesson']).strip()
            subject=str(draft.get('subject') or dr['subject']).strip()
            draft['source_file']=source_file; draft['lesson']=lesson; draft['subject']=subject; draft['content_type']=ct

            # Persist the exact edited draft first. Publish and DB step insertion happen
            # in the same transaction, so the published content is byte-for-byte sourced
            # from this normalized draft snapshot.
            cur.execute("UPDATE curriculum_drafts SET draft_json=%s::jsonb,status='ADMIN_REVIEW',updated_at=NOW() WHERE id=%s",(json.dumps(draft,ensure_ascii=False),draft_id))
            cur.execute("UPDATE curriculum_lessons SET status='ARCHIVED' WHERE source_file=%s AND content_type=%s AND lesson=%s AND status='PUBLISHED'",(source_file,ct,lesson))
            cur.execute("SELECT COALESCE(MAX(version),0)+1 AS next_version FROM curriculum_lessons WHERE source_file=%s AND content_type=%s AND lesson=%s",(source_file,ct,lesson)); version=int(cur.fetchone()['next_version'])
            cur.execute("INSERT INTO curriculum_lessons(draft_id,source_file,subject,content_type,lesson,status,version,raw_source_json) VALUES(%s,%s,%s,%s,%s,'PUBLISHED',%s,%s::jsonb) RETURNING id",(draft_id,source_file,subject,ct,lesson,version,json.dumps({'pages':draft.get('pages') or []},ensure_ascii=False)))
            lesson_id=int(cur.fetchone()['id'])
            for order,step in enumerate(draft.get('steps') or [],1):
                content=dict(step.get('content') or {})
                # Keep the editable text/content exactly as saved by Admin. Do not
                # regenerate or reconstruct it from source pages during publish.
                cur.execute("INSERT INTO curriculum_steps(lesson_id,step_code,step_order,title,step_type,content_json) VALUES(%s,%s,%s,%s,%s,%s::jsonb)",(lesson_id,str(step.get('code') or f'B{order-1}'),order,str(step.get('title') or content.get('title') or ''),str(step.get('type') or 'lesson'),json.dumps(content,ensure_ascii=False)))
            cur.execute("UPDATE curriculum_drafts SET status='PUBLISHED',updated_at=NOW() WHERE id=%s",(draft_id,))
        conn.commit()
    except HTTPException:
        conn.rollback(); raise
    except Exception:
        conn.rollback(); raise
    finally:
        conn.close()

    _invalidate_catalog_cache()

    # Index exactly the just-published edited content, never the original AI draft/source.
    if index:
        try:
            _delete_pinecone_scope_exact(
                '__default__',
                _pinecone_scope_filter(source_file, content_type=ct, lesson=lesson),
                source_file=source_file, content_type=ct, lesson=lesson, topic=None,
            )
            vectors=[]
            for step in draft.get('steps') or []:
                content=step.get('content') or {}
                text='\n'.join(str(content.get(k) or '') for k in ('title','content'))
                if not text.strip(): continue
                vec=embed_text(text[:8000])
                vectors.append({'id':f'curriculum:{lesson_id}:{step.get("code")}', 'values':vec, 'metadata':{'record_type':'curriculum_step','content_id':str(lesson_id),'lesson_id':str(lesson_id),'step_code':str(step.get('code') or ''),'source_file':source_file,'content_type':ct,'lesson':lesson,'text':text[:6000]}})
                if len(vectors)>=50:
                    index.upsert(vectors=vectors,namespace='__default__'); vectors=[]
            if vectors: index.upsert(vectors=vectors,namespace='__default__')
        except Exception as exc:
            print('[CURRICULUM PINECONE INDEX] failed:',type(exc).__name__,str(exc))
    return {'success':True,'lesson_id':lesson_id,'version':version,'status':'PUBLISHED'}

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
<h3>🧠 AI Curriculum Studio</h3>
<div class="small" style="margin-bottom:10px">Upload 1 bài học → Gemini OCR/Vision → AI dựng số bước → AI soạn từng bước → Admin sửa/duyệt → Publish vào PostgreSQL và Pinecone.</div>
<form onsubmit="createCurriculumDraft(event)">
<input id="curPdf" type="file" accept=".pdf,application/pdf" required style="width:100%;margin-bottom:8px">
<div style="display:flex;gap:8px;flex-wrap:wrap">
<input id="curSubject" value="Tiếng Nhật" placeholder="Môn học" required style="flex:1;min-width:180px">
</div>
<div style="margin-top:12px;padding:10px;border:1px solid #ddd;border-radius:9px;background:#fafafa">
  <div style="font-weight:700;margin-bottom:6px">📚 Cấu hình các bài trong PDF</div>
  <div class="small" style="margin-bottom:8px">Một PDF có thể tạo nhiều bài. Với mỗi bài, nhập <b>loại nội dung</b>, <b>tên bài</b> và <b>số trang</b> (ví dụ <b>7-8</b> hoặc <b>7,9-10</b>). Chỉ các trang được cấu hình mới được OCR/Vision, lưu ảnh và đưa vào AI Draft.</div>
  <div id="curArticleRows"></div>
  <button type="button" class="gray" onclick="addCurriculumArticleRow()" style="margin-top:8px">＋ Thêm bài</button>
</div>
<div style="display:flex;gap:8px;margin-top:10px">
  <button id="curGenBtn">🤖 AI tạo Draft</button>
  <button type="button" class="gray" onclick="clearCurriculumArticleRows()">↺ Đặt lại</button>
</div>
</form>
<div id="curStatus" class="small" style="margin-top:8px"></div>
<div id="curDraftList" style="margin-top:15px"></div>
<div id="curDraftEditor" style="margin-top:15px"></div>
</div>
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
<h3>🗂️ Quản lý tài liệu / bài học / chủ đề</h3>
<div class="small" style="margin-bottom:10px">Xóa tại đây sẽ xóa cả Pinecone vector và Knowledge Cache PostgreSQL tương ứng.</div>
<div id="knowledgeCatalogAdmin">Đang tải...</div>
</div>

<div class="card">
<h3>🧹 Dọn vector Pinecone mồ côi</h3>
<div class="small" style="margin-bottom:10px">Dùng khi PostgreSQL đã xóa mục nhưng Pinecone vẫn còn chunk. Rule: Tài liệu = source_file; Bài học = source_file + Loại nội dung + Bài học; Chủ đề = thêm Chủ đề.</div>
<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:8px">
  <input id="orphanSource" placeholder="source_file, ví dụ bai3.pdf">
  <input id="orphanContentType" placeholder="Loại nội dung">
  <input id="orphanLesson" placeholder="Bài học">
  <input id="orphanTopic" placeholder="Chủ đề">
</div>
<div style="margin-top:9px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
  <button class="gray" onclick="probeOrphanKnowledge()">🔎 Kiểm tra</button>
  <button class="red" onclick="deleteOrphanKnowledge()">🗑️ Xóa vector mồ côi</button>
  <span id="orphanProbeResult" class="small"></span>
</div>
</div>

<div class="card">
<h3>💳 Cấu hình gói thanh toán</h3>
<div class="small" style="margin-bottom:10px">Thiết lập giá và QR code cho 1 tháng / 3 tháng / 6 tháng. Sau khi user chuyển khoản, Admin xác nhận rồi cấp gói tương ứng.</div>
<div id="paymentPackagesAdmin"></div>
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
    await loadPaymentPackages();
    await loadKnowledgeCatalog();
    await loadCurriculumDrafts();
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


function curriculumTypeOptions(selected){
  const types=['Giáo trình','Từ vựng','Ngữ pháp','Bài tập','Truyện đọc'];
  return types.map(t=>`<option value="${esc(t)}" ${t===selected?'selected':''}>${esc(t)}</option>`).join('');
}
function addCurriculumArticleRow(values={}){
  const wrap=document.getElementById('curArticleRows'); if(!wrap)return;
  const row=document.createElement('div'); row.className='cur-article-row';
  row.style.cssText='display:grid;grid-template-columns:1.1fr 1.5fr 1fr auto;gap:7px;margin-bottom:7px;align-items:center';
  const ct=values.content_type||'Giáo trình';
  row.innerHTML=`<select class="cur-a-type" style="min-width:0">${curriculumTypeOptions(ct)}</select>
  <input class="cur-a-lesson" placeholder="Tên bài học" value="${esc(values.lesson||'')}">
  <input class="cur-a-pages" placeholder="Số trang, ví dụ 7-8" value="${esc(values.pages||'')}">
  <button type="button" class="red" title="Xóa dòng" onclick="this.parentElement.remove()">✕</button>`;
  wrap.appendChild(row);
}
function clearCurriculumArticleRows(){
  const wrap=document.getElementById('curArticleRows'); if(!wrap)return; wrap.innerHTML=''; addCurriculumArticleRow();
}
function getCurriculumArticleRows(){
  return [...document.querySelectorAll('#curArticleRows .cur-article-row')].map((row,idx)=>({
    index:idx+1,
    content_type:row.querySelector('.cur-a-type')?.value||'Giáo trình',
    lesson:(row.querySelector('.cur-a-lesson')?.value||'').trim(),
    pages:(row.querySelector('.cur-a-pages')?.value||'').trim()
  }));
}
addCurriculumArticleRow();

async function createCurriculumDraft(event){
 event.preventDefault(); const btn=document.getElementById('curGenBtn'); const st=document.getElementById('curStatus'); const file=document.getElementById('curPdf').files[0]; if(!file)return;
 const rows=getCurriculumArticleRows().filter(x=>x.lesson||x.pages);
 if(!rows.length){st.textContent='❌ Hãy thêm ít nhất 1 bài và nhập tên bài + số trang.';return;}
 for(const r of rows){if(!r.lesson||!r.pages){st.textContent=`❌ Bài #${r.index}: cần đủ tên bài và số trang.`;return;}}
 btn.disabled=true; st.textContent=`⏳ Đang xử lý ${rows.length} bài, chỉ OCR/Vision các trang đã cấu hình...`;
 try{
   const fd=new FormData(); fd.append('password',pw); fd.append('file',file); fd.append('subject',document.getElementById('curSubject').value.trim()); fd.append('articles_json',JSON.stringify(rows)); fd.append('metadata_json','[]');
   const r=await fetch('/admin/api/curriculum/draft-upload',{method:'POST',body:fd});
   const t=await r.text(); let d={}; try{d=JSON.parse(t)}catch{d={detail:t}} if(!r.ok)throw Error(d.detail||('HTTP '+r.status));
   const drafts=Array.isArray(d.drafts)?d.drafts:[];
   st.textContent=`✅ Đã tạo ${drafts.length} Draft. PDF có ${d.pdf_page_count||'?'} trang; chỉ các trang đã cấu hình được OCR/Vision và lưu vào Draft.`;
   await loadCurriculumDrafts();
   if(drafts.length===1){await openCurriculumDraft(drafts[0].draft_id);}
   else if(drafts.length){document.getElementById('curStatus').textContent += ` · ${drafts.length} bài đang chờ duyệt.`;}
 }catch(e){st.textContent='❌ '+e.message;} finally{btn.disabled=false;}
}
async function loadCurriculumDrafts(){
  try{
    const d=await api('/admin/api/curriculum/drafts?password='+encodeURIComponent(pw));
    const list=Array.isArray(d.drafts)?d.drafts.filter(x=>String(x.status||'').toUpperCase()!=='PUBLISHED'):[];
    const box=document.getElementById('curDraftList');
    if(!box)return;
    if(!list.length){box.innerHTML='';return;}
    box.innerHTML=`<div style="border-top:1px solid #ddd;padding-top:12px"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap"><b>📝 Draft đang chờ duyệt (${list.length})</b><button type="button" class="gray" onclick="loadCurriculumDrafts()">🔄 Làm mới</button></div>${list.map(x=>{
      const status=String(x.status||'AI_DRAFT').toUpperCase();
      const label=status==='ADMIN_REVIEW'?'Đang chỉnh sửa':'AI_DRAFT';
      return `<div class="cur-draft-row" style="display:flex;justify-content:space-between;gap:10px;align-items:center;border:1px solid #ddd;border-radius:8px;padding:9px;margin-top:7px;background:#fafafa">
        <div><b>Draft #${esc(x.id)}</b> · ${esc(x.content_type)} · ${esc(x.lesson)}<div class="small">${esc(x.source_file||'')} · ${esc(label)} · v${esc(x.version||1)}</div></div>
        <div style="display:flex;gap:6px;flex-wrap:wrap"><button type="button" onclick="openCurriculumDraft(${Number(x.id)})">✏️ Mở & sửa</button><button type="button" class="red js-delete-draft" data-draft-id="${Number(x.id)}" data-draft-label="${esc(x.lesson||('Draft #'+x.id))}">🗑️ Xóa Draft</button></div>
      </div>`;
    }).join('')}</div>`;
  }catch(e){
    const box=document.getElementById('curDraftList'); if(box)box.innerHTML='<div class="small" style="color:#b00">Không tải được danh sách Draft: '+esc(e.message)+'</div>';
  }
}

async function deleteCurriculumDraft(id,lesson){
  const label=String(lesson||'Draft #'+id);
  const st=document.getElementById('curStatus');
  if(st) st.textContent=`⏳ Đang xóa Draft #${id}...`;
  try{
    if(!window.confirm(`Xóa Draft "${label}"?\n\nChỉ xóa bản Draft này, không ảnh hưởng giáo trình PUBLISHED của Doraemon.`)){
      if(st) st.textContent='';
      return;
    }
    const url='/admin/api/curriculum/drafts/'+encodeURIComponent(String(id))+'/remove?password='+encodeURIComponent(pw);
    console.log('[CURRICULUM DRAFT DELETE] request', {id, url});
    const d=await api(url,{method:'POST'});
    console.log('[CURRICULUM DRAFT DELETE] response', d);
    if(Number(window.currentCurriculumDraftId||0)===Number(id)){const ed=document.getElementById('curDraftEditor');if(ed)ed.innerHTML='';window.currentCurriculumDraftId=null;}
    await loadCurriculumDrafts();
    if(st) st.textContent=`✅ Đã xóa Draft #${id}.`;
  }catch(e){
    console.error('[CURRICULUM DRAFT DELETE] failed', e);
    if(st) st.textContent=`❌ Xóa Draft thất bại: ${e.message}`;
    alert('❌ Xóa Draft thất bại: '+e.message);
  }
}

document.addEventListener('click', async (event)=>{
  const btn=event.target.closest('.js-delete-draft');
  if(!btn) return;
  event.preventDefault();
  event.stopPropagation();
  const id=Number(btn.dataset.draftId||0);
  const label=btn.dataset.draftLabel||('Draft #'+id);
  if(!id) return;
  await deleteCurriculumDraft(id,label);
});
async function openCurriculumDraft(id){
 try{const d=await api('/admin/api/curriculum/drafts/'+id+'?password='+encodeURIComponent(pw)); const dj=d.draft_json||{}; renderCurriculumDraft(id,{...dj,draft_id:id,status:d.status,version:d.version});}
 catch(e){alert('❌ '+e.message)}
}
function escJson(v){return esc(JSON.stringify(v||{}));}
function _curriculumStepStateKey(code){return String(code||'').trim();}
function _curriculumGetStepState(code,fallbackContent){
  window.currentCurriculumImageState=window.currentCurriculumImageState||{};
  const k=_curriculumStepStateKey(code);
  if(!window.currentCurriculumImageState[k]) window.currentCurriculumImageState[k]=JSON.parse(JSON.stringify(fallbackContent||{}));
  const st=window.currentCurriculumImageState[k]||{}; if(!Array.isArray(st.images)) st.images=[]; return st;
}
function _curriculumSetStepState(code,content){
  window.currentCurriculumImageState=window.currentCurriculumImageState||{};
  window.currentCurriculumImageState[_curriculumStepStateKey(code)]=JSON.parse(JSON.stringify(content||{}));
}
function curriculumImageGallery(step,pages){
  pages=Array.isArray(pages)?pages:[]; const code=String(step?.code||''); const content=_curriculumGetStepState(code,step?.content||{}); const imgs=Array.isArray(content.images)?content.images:[];
  const inv={}; pages.forEach(pg=>(Array.isArray(pg?.images)?pg.images:[]).forEach(im=>{const k=String(im.image_key||'').trim();if(k)inv[k]=im;}));
  const selected=new Set(imgs.map(im=>String(im?.image_key||im?.key||'').trim()).filter(Boolean)); const all=[];
  pages.forEach(pg=>(Array.isArray(pg?.images)?pg.images:[]).forEach(im=>{const k=String(im.image_key||'').trim();if(k&&!all.some(x=>x.key===k))all.push({key:k,page:pg.page,src:String(im.image_url||'').trim(),vision:im.vision||{}});}));
  const action=(key,add)=>`<button type="button" class="${add?'':'red'}" style="margin-top:7px" data-cur-image-action="1" data-code="${esc(code)}" data-image-key="${esc(key)}" data-add="${add?'1':'0'}" onclick='changeCurriculumImage(${JSON.stringify(code)},${JSON.stringify(key)},${add?'true':'false'});return false;'>${add?'＋ Thêm ảnh':'🗑️ Bỏ ảnh'}</button>`;
  const card=(im,idx,isSel)=>{const key=String(im?.image_key||im?.key||'').trim();const src=String(im?.image_url||inv[key]?.image_url||im?.src||'').trim();const page=im?.page||inv[key]?.page||'';const v=im?.vision||inv[key]?.vision||{};const cap=im?.caption||v.caption||v.description||v.explanation||'';return `<div style="border:2px solid ${isSel?'#1677ff':'#ddd'};border-radius:10px;overflow:hidden;background:#fff"><div style="height:150px;background:#f6f7f9;display:flex;align-items:center;justify-content:center">${src?`<img src="${esc(src)}" style="width:100%;height:150px;object-fit:contain" onerror="this.style.display='none';this.nextElementSibling.style.display='block'">`:''}<div style="display:${src?'none':'block'};padding:10px;color:#888">Ảnh không tải được</div></div><div style="padding:9px"><b>Ảnh ${idx+1}${page?` · Trang ${esc(page)}`:''}</b><div class="small" style="word-break:break-all">${esc(key)}</div>${cap?`<div class="small" style="margin-top:4px">${esc(cap)}</div>`:''}${action(key,!isSel)}</div></div>`};
  const selectedHtml=imgs.map((im,i)=>card(im,i,true)).join(''); const remaining=all.filter(x=>!selected.has(x.key));
  return `<div id="cur-gallery-${encodeURIComponent(code)}" style="margin-top:10px"><b>🖼️ Ảnh của bước</b>${selectedHtml?`<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-top:8px">${selectedHtml}</div>`:`<div class="small" style="margin-top:6px;color:#b76b00">⚠️ Chưa chọn ảnh</div>`}<details style="margin-top:10px"><summary style="cursor:pointer;font-weight:700">＋ Thêm ảnh từ nguồn (${remaining.length})</summary>${remaining.length?`<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-top:8px">${remaining.map((im,i)=>card(im,i,false)).join('')}</div>`:`<div class="small" style="padding:7px">Không còn ảnh nguồn khác.</div>`}</details></div>`;
}
function _findCurriculumJsonTextarea(code){const wanted=String(code||'');for(const ta of document.querySelectorAll('#curSteps .cur-json'))if(String(ta.getAttribute('data-code')||'')===wanted)return ta;return null;}
function _findCurriculumTextTextarea(code){const wanted=String(code||'');for(const ta of document.querySelectorAll('#curSteps .cur-text'))if(String(ta.getAttribute('data-code')||'')===wanted)return ta;return null;}
function _findCurriculumTextarea(code){return _findCurriculumJsonTextarea(code);}
function changeCurriculumImage(code,key,add){
  const wanted=String(code||'').trim(), k=String(key||'').trim(); if(!wanted||!k)return false;
  let c=_curriculumGetStepState(wanted,{}); const ta=_findCurriculumTextarea(wanted);
  if(ta){try{c=JSON.parse(ta.value||'{}');}catch(e){}}
  if(!Array.isArray(c.images))c.images=[]; const pages=Array.isArray(window.currentCurriculumPages)?window.currentCurriculumPages:[]; const inv={};
  pages.forEach(pg=>(Array.isArray(pg?.images)?pg.images:[]).forEach(im=>{const ik=String(im.image_key||'').trim();if(ik)inv[ik]=im;}));
  if(add){if(!c.images.some(im=>String(im?.image_key||im?.key||'').trim()===k)){const src=inv[k]||{};const v=src.vision||{};c.images.push({image_key:k,image_url:src.image_url||'',page:src.page||null,caption:v.caption||v.description||v.explanation||''});}}
  else c.images=c.images.filter(im=>String(im?.image_key||im?.key||'').trim()!==k);
  _curriculumSetStepState(wanted,c); if(ta){ta.value=JSON.stringify(c,null,2);ta.dispatchEvent(new Event('input',{bubbles:true}));}
  const host=document.getElementById('cur-gallery-'+encodeURIComponent(wanted)); if(host)host.outerHTML=curriculumImageGallery({code:wanted,content:c},pages); return false;
}
function deleteCurriculumStep(id,code){const label=String(code||'');if(!confirm(`Xóa bước ${label} khỏi DRAFT? Giáo trình PUBLISHED của Doraemon KHÔNG bị ảnh hưởng cho tới khi bạn Publish lại.`))return;api('/admin/api/curriculum/drafts/'+id+'/delete-step',{method:'POST',body:JSON.stringify({password:pw,step_code:label})}).then(d=>{renderCurriculumDraft(id,d);const st=document.getElementById('curStatus');if(st)st.textContent=`✅ Đã xóa ${label} khỏi Draft. Các bước sau đã được đánh lại.`;}).catch(e=>alert('❌ '+e.message));}
document.addEventListener('click',function(ev){const btn=ev.target.closest&&ev.target.closest('[data-cur-image-action]');if(!btn)return;ev.preventDefault();ev.stopPropagation();changeCurriculumImage(btn.getAttribute('data-code')||'',btn.getAttribute('data-image-key')||'',btn.getAttribute('data-add')==='1');});
function renderCurriculumDraft(id,data){
  window.currentCurriculumDraftId=id; window.currentCurriculumPages=Array.isArray(data.pages)?data.pages:[]; window.currentCurriculumImageState={}; const box=document.getElementById('curDraftEditor'); const steps=Array.isArray(data.steps)?data.steps:[]; steps.forEach(s=>_curriculumSetStepState(String(s.code||''),s.content||{}));
  box.innerHTML=`<div style="border-top:1px solid #ddd;padding-top:12px"><b>Draft #${id}</b> · ${esc(data.content_type)} · ${esc(data.lesson)} ${data.page_ranges?`· Trang ${esc(data.page_ranges)}`:''}<div id="curSteps">${steps.map((s)=>{const code=String(s.code||'');const required=(data.content_type==='Giáo trình'&&['B0','B1','FINAL'].includes(code));return `<div class="card cur-step-card" data-step-code="${esc(code)}" style="box-shadow:none;border:1px solid #ddd;margin-top:9px;padding:12px"><div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap"><div style="display:flex;align-items:center;gap:7px"><b>${esc(code)} · </b><input class="cur-title" value="${esc(s.title)}" style="flex:1;min-width:200px"></div><div style="display:flex;gap:6px;align-items:center">${required?`<span class="small" style="color:#888">🔒 Bắt buộc</span>`:`<button class="red" type="button" onclick='deleteCurriculumStep(${id},${JSON.stringify(code)});return false;'>🗑️ Xóa bước</button>`}<button class="gray" type="button" onclick='regenerateCurriculumStep(${id},${JSON.stringify(code)});return false;'>🤖 Gen lại</button></div></div>${curriculumImageGallery(s,data.pages||[])}<label class="small" style="display:block;margin-top:8px"><b>✏️ Nội dung bước (Doraemon sẽ dùng nội dung này)</b></label><textarea class="cur-text" data-code="${esc(code)}" style="width:100%;min-height:150px;margin-top:5px">${esc((s.content&&typeof s.content==='object')?String(s.content.content||''):'')}</textarea><details style="margin-top:8px"><summary style="cursor:pointer;font-weight:700">⚙️ Dữ liệu JSON nâng cao</summary><textarea class="cur-json" data-code="${esc(code)}" style="width:100%;min-height:180px;margin-top:8px;font-family:monospace">${esc(JSON.stringify(s.content||{},null,2))}</textarea></details></div>`;}).join('')}</div><div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px"><button class="gray" onclick="saveCurriculumDraft(${id})">💾 Lưu chỉnh sửa</button><button onclick="publishCurriculumDraft(${id})">✅ Duyệt & Publish</button></div></div>`;
}
function reindexCurriculumDraftStepsClient(contentType,steps){const raw=(Array.isArray(steps)?steps:[]).filter(x=>x&&typeof x==='object').map(x=>({...x}));const ct=String(contentType||'').trim();if(ct==='Giáo trình'){const b0=raw.find(x=>String(x.code||'').toUpperCase()==='B0');const b1=raw.find(x=>String(x.code||'').toUpperCase()==='B1');const final=raw.find(x=>['FINAL','SUMMARY'].includes(String(x.code||'').toUpperCase()));const sections=raw.filter(x=>!['B0','B1','FINAL','SUMMARY'].includes(String(x.code||'').toUpperCase()));const out=[];if(b0){b0.code='B0';out.push(b0);}if(b1){b1.code='B1';out.push(b1);}sections.forEach((x,i)=>{x.code='B'+(i+2);out.push(x);});if(final){final.code='FINAL';out.push(final);}return out;}raw.forEach((x,i)=>{x.code='B'+i;});return raw;}
async function collectCurriculumDraft(id){const base=await api('/admin/api/curriculum/drafts/'+id+'?password='+encodeURIComponent(pw));const d=base.draft_json||{};d.steps=(d.steps||[]).map(s=>{const code=String(s.code||'');const jsonTa=_findCurriculumJsonTextarea(code);const textTa=_findCurriculumTextTextarea(code);const titleEl=textTa?.closest('.cur-step-card')?.querySelector('.cur-title');let content=_curriculumGetStepState(code,s.content||{});if(jsonTa){try{content=JSON.parse(jsonTa.value||JSON.stringify(content));}catch(e){throw new Error(`Bước ${code}: JSON nâng cao không hợp lệ. Hãy sửa JSON hoặc để nguyên phần nâng cao.`);}}if(textTa){content={...(content||{}),content:String(textTa.value||'')};}if(!Array.isArray(content.images))content.images=[];content.images=content.images.map(im=>({...im,image_key:String(im?.image_key||im?.key||'').trim()})).filter(im=>im.image_key);_curriculumSetStepState(code,content);return {...s,title:titleEl?.value||s.title,content};});d.steps=reindexCurriculumDraftStepsClient(String(d.content_type||''),d.steps||[]);return d;}
async function saveCurriculumDraft(id){try{const draft=await collectCurriculumDraft(id);const saved=await api('/admin/api/curriculum/drafts/'+id,{method:'POST',body:JSON.stringify({password:pw,draft})});const merged={...draft,...saved,steps:saved.steps||draft.steps};renderCurriculumDraft(id,merged);await loadCurriculumDrafts();alert('✅ Đã lưu chỉnh sửa.');}catch(e){alert('❌ '+e.message);}}
async function regenerateCurriculumStep(id,code){try{const d=await api('/admin/api/curriculum/drafts/'+id+'/regenerate-step',{method:'POST',body:JSON.stringify({password:pw,step_code:code})});const jsonTa=_findCurriculumJsonTextarea(String(code));const textTa=_findCurriculumTextTextarea(String(code));if(jsonTa)jsonTa.value=JSON.stringify(d.step.content||{},null,2);if(textTa)textTa.value=String((d.step.content||{}).content||'');_curriculumSetStepState(String(code),d.step.content||{});const host=document.getElementById('cur-gallery-'+encodeURIComponent(String(code)));if(host)host.outerHTML=curriculumImageGallery({code,content:d.step.content||{}},Array.isArray(window.currentCurriculumPages)?window.currentCurriculumPages:[]);alert('✅ Đã gen lại '+code);}catch(e){alert('❌ '+e.message);}}
async function publishCurriculumDraft(id){try{if(!confirm('Publish giáo trình này? Sau khi publish Doraemon mới được phép dùng nội dung này.'))return;const draft=await collectCurriculumDraft(id);const d=await api('/admin/api/curriculum/drafts/'+id+'/publish',{method:'POST',body:JSON.stringify({password:pw,draft})});alert(`✅ Published lesson #${d.lesson_id}, version ${d.version}.`);await loadCurriculumDrafts();await loadKnowledgeCatalog();const st=document.getElementById('curStatus');if(st)st.textContent=`✅ Published lesson #${d.lesson_id}, version ${d.version}. Draft này đã được ẩn; các Draft chưa publish vẫn được giữ.`;const ed=document.getElementById('curDraftEditor');if(ed)ed.innerHTML='';window.currentCurriculumDraftId=null;}catch(e){alert('❌ '+e.message);}}

function toggleKbSection(id,btn){
  const el=document.getElementById(id);
  if(!el) return;
  const hidden=el.style.display==="none";
  el.style.display=hidden?"":"none";
  if(btn) btn.textContent=hidden?"▾ Thu gọn":"▸ Mở rộng";
}
function setAllKbDocuments(collapse){
  document.querySelectorAll('#knowledgeCatalogAdmin .kb-collapsible').forEach(el=>{ el.style.display=collapse?'none':''; });
  document.querySelectorAll('#knowledgeCatalogAdmin .kb-doc-toggle').forEach(btn=>{ btn.textContent=collapse?'▸ Mở rộng':'▾ Thu gọn'; });
}
function renderKnowledgeCatalog(nodes){
  const box=document.getElementById("knowledgeCatalogAdmin");
  if(!nodes||!nodes.length){box.innerHTML='<div class="small">Chưa có tài liệu.</div>';return;}
  box.innerHTML=nodes.map((doc,di)=>{
    const cts=doc.content_types||[];
    const docId=`kb-doc-${di}`;
    return `<div style="border:1px solid #ddd;border-radius:10px;padding:12px;margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap">
        <div style="display:flex;align-items:center;gap:8px">
          <button class="gray kb-toggle kb-doc-toggle" style="padding:5px 10px;min-width:110px;font-weight:700" onclick="toggleKbSection('${docId}',this)">▾ Thu gọn</button>
          <div><b>📄 ${esc(doc.source_file)}</b><div class="small">${esc(doc.subject||"")} · ${esc(doc.namespace||"__default__")}</div></div>
        </div>
        <button class="red" onclick='deleteKnowledgeScope(${JSON.stringify({source_file:doc.source_file})})'>🗑️ Xóa tài liệu</button>
      </div>
      <div id="${docId}" class="kb-collapsible" style="margin-top:8px">${cts.map((ct,ci)=>{
        const ctId=`${docId}-ct-${ci}`;
        return `<div style="border-top:1px solid #eee;margin-top:9px;padding-top:9px">
          <div style="display:flex;align-items:center;gap:7px">
            <button class="gray" style="padding:4px 8px" onclick="toggleKbSection('${ctId}',this)">▾</button>
            <b>Loại nội dung:</b> ${esc(ct.content_type)}
          </div>
          <div id="${ctId}" style="margin-top:5px">${(ct.lessons||[]).map((ls,li)=>{
            const lsId=`${ctId}-ls-${li}`;
            return `<div style="margin:7px 0 7px 18px;background:#f8fafc;border-radius:8px;padding:8px">
              <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap">
                <div style="display:flex;align-items:center;gap:7px">
                  <button class="gray" style="padding:4px 8px" onclick="toggleKbSection('${lsId}',this)">▾</button>
                  <div><b>📘 Bài học:</b> ${esc(ls.lesson)}${ls.lesson_pages?` <span class='small'>[${esc(ls.lesson_pages)}]</span>`:""}</div>
                </div>
                <button class="red" onclick='deleteKnowledgeScope(${JSON.stringify({source_file:doc.source_file,content_type:ct.content_type,lesson:ls.lesson})})'>Xóa bài</button>
              </div>
              <div id="${lsId}" style="margin-top:6px">${(ls.topics||[]).length?`<div class="small" style="margin-left:26px"><b>Chủ đề:</b> ${(ls.topics||[]).map(t=>`<span style='display:inline-flex;align-items:center;gap:4px;border:1px solid #ddd;border-radius:999px;padding:3px 7px;margin:3px 4px 0 0;background:#fff'>${esc(t.topic)} <button class='red' style='padding:2px 6px' onclick='deleteKnowledgeScope(${JSON.stringify({source_file:doc.source_file,content_type:ct.content_type,lesson:ls.lesson,topic:t.topic})})'>×</button></span>`).join("")}</div>`:`<div class="small" style="margin-left:26px;color:#999">Không có chủ đề.</div>`}</div>
            </div>`;
          }).join("")}</div>
        </div>`;
      }).join("")}</div>
    </div>`;
  }).join("");
}
async function loadKnowledgeCatalog(){const box=document.getElementById("knowledgeCatalogAdmin");try{const d=await api("/admin/api/knowledge/catalog?password="+encodeURIComponent(pw)+"&t="+Date.now());renderKnowledgeCatalog(d.documents||[]);}catch(e){box.innerHTML='<span style="color:#c00">Không tải được catalog: '+esc(e.message)+'</span>';}}
async function deleteKnowledgeScope(scope){const what=scope.topic?`chủ đề "${scope.topic}"`:scope.lesson?`bài học "${scope.lesson}"`:`tài liệu "${scope.source_file}"`;if(!confirm(`Xóa ${what}?\n\nSẽ xóa vector/chunk Pinecone và Knowledge Cache liên quan. Thao tác này không thể hoàn tác.`))return;try{const d=await api("/admin/api/knowledge/delete",{method:"POST",body:JSON.stringify({...scope,password:pw})});alert("✅ "+d.message);await loadKnowledgeCatalog();}catch(e){alert("❌ Xóa thất bại: "+e.message);}}

async function probeOrphanKnowledge(){
  const source_file=document.getElementById("orphanSource").value.trim();
  const content_type=document.getElementById("orphanContentType").value.trim();
  const lesson=document.getElementById("orphanLesson").value.trim();
  const topic=document.getElementById("orphanTopic").value.trim();
  if(!source_file){alert("Nhập source_file trước.");return;}
  try{
    const d=await api("/admin/api/knowledge/pinecone-probe",{method:"POST",body:JSON.stringify({password:pw,source_file,content_type,lesson,topic})});
    document.getElementById("orphanProbeResult").textContent=`Pinecone còn ${d.count} vector trong exact scope.`;
  }catch(e){document.getElementById("orphanProbeResult").textContent="❌ "+e.message;}
}
async function deleteOrphanKnowledge(){
  const source_file=document.getElementById("orphanSource").value.trim();
  const content_type=document.getElementById("orphanContentType").value.trim();
  const lesson=document.getElementById("orphanLesson").value.trim();
  const topic=document.getElementById("orphanTopic").value.trim();
  if(!source_file){alert("Nhập source_file trước.");return;}
  const what=topic?`chủ đề "${topic}"`:lesson?`bài học "${lesson}"`:content_type?`loại nội dung "${content_type}"`: `tài liệu "${source_file}"`;
  if(!confirm(`Xóa vector Pinecone mồ côi của ${what}?\n\nServer chỉ xóa exact identity scope và phải verify Pinecone sạch mới báo thành công.`))return;
  try{
    const d=await api("/admin/api/knowledge/delete",{method:"POST",body:JSON.stringify({password:pw,source_file,content_type,lesson,topic})});
    alert("✅ "+d.message);
    await probeOrphanKnowledge();
    await loadKnowledgeCatalog();
  }catch(e){alert("❌ Xóa thất bại: "+e.message);}
}
async function loadPaymentPackages(){
  try{
    const d=await api("/admin/api/payment-packages?password="+encodeURIComponent(pw));
    document.getElementById("paymentPackagesAdmin").innerHTML=d.packages.map(p=>{
      const preview=p.qr_url?`<img src="${esc(p.qr_url)}" style="width:110px;height:110px;object-fit:contain;border:1px solid #ddd;border-radius:6px;margin-top:7px">`:`<div class="small" style="margin-top:7px;color:#999">Chưa có QR</div>`;
      return `<div style="display:grid;grid-template-columns:90px 160px 1fr 130px;gap:10px;align-items:center;border-top:1px solid #eee;padding:10px 0">
        <b>${esc(p.plan_name)}</b>
        <input id="price-${p.months}" type="number" min="0" step="1000" value="${Number(p.price_vnd||0)}" placeholder="Giá VNĐ">
        <div>${preview}<div class="small">${esc(p.qr_key||"")}</div></div>
        <div>
          <input id="qr-${p.months}" type="file" accept="image/png,image/jpeg,image/webp" style="width:100%">
          <button style="margin-top:6px" onclick="savePaymentPackage(${p.months})">Lưu</button>
        </div>
      </div>`;
    }).join("");
  }catch(e){ document.getElementById("paymentPackagesAdmin").textContent="Không tải được cấu hình thanh toán: "+e.message; }
}
async function savePaymentPackage(months){
  const fd=new FormData();
  fd.append("password",pw);
  fd.append("price_vnd",document.getElementById("price-"+months).value||0);
  const file=document.getElementById("qr-"+months).files[0];
  if(file) fd.append("qr_file",file);
  try{
    const r=await fetch("/admin/api/payment-packages/"+months,{method:"POST",body:fd});
    const t=await r.text(); let d={}; try{d=JSON.parse(t)}catch{d={detail:t}}
    if(!r.ok)throw Error(d.detail||("HTTP "+r.status));
    alert("Đã lưu cấu hình "+months+" tháng.");
    loadPaymentPackages();
  }catch(e){alert("Không lưu được: "+e.message);}
}

async function loadUsers(){
  const d=await api("/admin/api/users?password="+encodeURIComponent(pw));
  document.getElementById("count").textContent="  Tổng: "+d.users.length;
  document.getElementById("users").innerHTML=d.users.map(u=>{
    const s=u.subscription||{}, st=u.status||"PENDING";
    const ex=s.expires_at?new Intl.DateTimeFormat("vi-VN",{dateStyle:"short",timeStyle:"short",timeZone:"Asia/Ho_Chi_Minh"}).format(new Date(s.expires_at)):"-";
    return `<div class="user ${selectedUser===u.id?'sel':''}" onclick="selectUser(${u.id},'${esc(u.nickname)}')">
      <b>#${u.id} ${esc(u.nickname)}</b> — ${esc(u.phone)}
      <div><span class="status-${st}"><b>${st}</b></span> · Gói: <b>${esc(s.plan||"Free")}</b> · ${s.plan==='Free' ? `đã hỏi hôm nay: ${Number(s.used_today||0)}/5` : `hết hạn: ${ex}`}</div>
      <div class="small">Bấm để xem lịch sử và chat</div>
      <div style="margin-top:7px">
        <button onclick="event.stopPropagation();act(${u.id},1)">1 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},3)">3 tháng</button>
        <button onclick="event.stopPropagation();act(${u.id},6)">6 tháng</button>
        <button class="gray" onclick="event.stopPropagation();resetFree(${u.id})">Gói Free</button>
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
async function resetFree(id){
  if(!confirm("Đưa user này về gói Free (5 lượt hỏi/ngày)?"))return;
  await api("/admin/api/users/"+id+"/reset-free",{method:"POST",body:JSON.stringify({password:pw})});
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
def _table_explanation_overlaps_chunk(explanation, marker, chunk):
    """Return True when a chunk contains any meaningful part of one table explanation.

    Table explanations can be longer than one RAG chunk. Matching only the
    marker therefore attached the table image to chunk 0 while later chunks
    containing the actual explanation had image_keys=[]. We intentionally map
    the same table image to every chunk that contains a substantial fragment of
    that table's Vision explanation.
    """
    ch = re.sub(r"\s+", " ", str(chunk or "")).strip()
    exp = re.sub(r"\s+", " ", str(explanation or "")).strip()
    mark = re.sub(r"\s+", " ", str(marker or "")).strip()
    if not ch:
        return False
    if mark and mark in ch:
        return True
    if not exp:
        return False
    if exp in ch:
        return True
    # Check several stable 80-character anchors. This catches continuation
    # chunks without requiring the whole explanation to fit inside one chunk.
    if len(exp) >= 80:
        anchors = [exp[:80], exp[len(exp)//2-40:len(exp)//2+40], exp[-80:]]
        if any(a and a in ch for a in anchors):
            return True
    # Final conservative token-window fallback for OCR whitespace differences.
    tokens = exp.split()
    if len(tokens) >= 12:
        for start in (0, max(0, len(tokens)//2 - 6), max(0, len(tokens)-12)):
            window = " ".join(tokens[start:start+12])
            if len(window) >= 40 and window in ch:
                return True
    return False


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

def b2_put_file(key: str, file_path: str, content_type: str):
    """Upload local file without loading the whole PDF into RAM."""
    if not b2_ready():
        raise RuntimeError("Backblaze B2 chưa được cấu hình. Cần B2_ENDPOINT, B2_KEY_ID, B2_APPLICATION_KEY và B2_BUCKET.")
    with open(file_path, "rb") as fh:
        b2.put_object(Bucket=B2_BUCKET, Key=key, Body=fh, ContentType=content_type)
    if B2_PUBLIC_BASE_URL:
        return f"{B2_PUBLIC_BASE_URL}/{key}"
    return None

def b2_delete_key(key: str):
    """Best-effort delete for an object already stored in Backblaze B2."""
    if not key or not b2_ready():
        return False
    try:
        b2.delete_object(Bucket=B2_BUCKET, Key=key)
        return True
    except Exception as exc:
        print(f"[B2 DELETE] key={key!r} skipped: {type(exc).__name__}: {exc}")
        return False


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

def render_pdf_page(pdf_source, page_no: int, dpi: int = 150) -> bytes:
    if fitz is None:
        raise RuntimeError("PyMuPDF chưa được cài đặt. Thêm PyMuPDF vào requirements.")
    doc = fitz.open(pdf_source) if isinstance(pdf_source, (str, os.PathLike)) else fitz.open(stream=pdf_source, filetype="pdf")
    try:
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def _parse_gemini_json(text: str):
    """Parse Gemini JSON robustly, including raw control chars inside string values.

    Gemini occasionally emits literal newlines/tabs inside a JSON string even when
    response_mime_type="application/json" is requested. Python's default json.loads
    rejects those with `Invalid control character`. `strict=False` intentionally allows
    those control characters and is safe here because this parser is only consuming
    model output that we immediately validate structurally.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    if not raw:
        raise ValueError("Gemini không trả về nội dung JSON.")

    last_error = None
    for candidate in (raw,):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            last_error = exc

    # Fallback: extract the outermost JSON object when Gemini adds prose around it.
    start = raw.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        end = None
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None:
            candidate = raw[start:end]
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError as exc:
                last_error = exc

    detail = f"; {last_error}" if last_error else ""
    raise ValueError(f"Gemini không trả về JSON hợp lệ{detail}")

def gemini_ocr_page(page_png: bytes, page_no: int, source_file: str = ""):
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
    _log_gemini_usage(response, operation=f"vision_page_images:{source_file}:page_{page_no}")
    data = _parse_gemini_json(response.text or "{}")
    text = str(data.get("text") or "").strip()
    images = data.get("images") if isinstance(data.get("images"), list) else []
    return text, images

def _detect_long_grid_lines(page_png: bytes):
    """Cheap local detector used ONLY to decide whether a page contains a table.

    This never reconstructs/reads table cells and never calls Gemini.
    """
    if Image is None:
        return False
    try:
        im = Image.open(io.BytesIO(page_png)).convert("L")
        max_w = 1000
        if im.width > max_w:
            ratio = max_w / float(im.width)
            im = im.resize((max_w, max(1, int(im.height * ratio))))
        w, h = im.size
        if w < 120 or h < 120:
            return False
        px = im.load()
        threshold = 165

        def clustered_hits(axis="h"):
            hits=[]
            if axis == "h":
                min_run=max(45, int(w*0.28))
                for y in range(h):
                    run=longest=dark=0
                    for x in range(w):
                        if px[x,y] < threshold:
                            run += 1; dark += 1; longest=max(longest,run)
                        else:
                            run=0
                    if longest >= min_run and dark/float(w) >= 0.22:
                        hits.append(y)
            else:
                min_run=max(45, int(h*0.25))
                for x in range(w):
                    run=longest=dark=0
                    for y in range(h):
                        if px[x,y] < threshold:
                            run += 1; dark += 1; longest=max(longest,run)
                        else:
                            run=0
                    if longest >= min_run and dark/float(h) >= 0.18:
                        hits.append(x)
            if not hits:
                return []
            groups=[[hits[0]]]
            for v in hits[1:]:
                if v-groups[-1][-1] <= 4:
                    groups[-1].append(v)
                else:
                    groups.append([v])
            return [int(sum(g)/len(g)) for g in groups]

        hs=clustered_hits("h")
        vs=clustered_hits("v")
        return (len(hs) >= 2 and len(vs) >= 2) or (len(hs) >= 4 and len(vs) >= 1)
    except Exception as exc:
        print("[TABLE DETECTOR] skipped:", type(exc).__name__, str(exc))
        return False


def _text_looks_like_table(extracted: str) -> bool:
    """Detect table-like text emitted by PDF text extraction, without OCR.

    Many textbook PDFs expose tables as text/Markdown-like rows even when
    their vector borders are not detectable at preview DPI. This detector is
    deliberately conservative: it only decides whether to enter the existing
    Table Visual pipeline; it never reconstructs rows/columns.
    """
    text = str(extracted or "").strip()
    if not text:
        return False
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    lines = [x for x in lines if x]
    if len(lines) < 3:
        return False

    pipe_lines = sum(1 for x in lines if x.count("|") >= 2)
    separator_lines = sum(1 for x in lines if re.search(r"\|\s*:?-{2,}:?\s*(?:\||$)", x))
    # Markdown-like table extraction: several pipe rows plus at least one
    # separator/header row.
    if pipe_lines >= 3 and separator_lines >= 1:
        return True

    # Do NOT classify ordinary prose/exercises as a table merely because they
    # contain several times/prices. For non-pipe text we rely on the visual
    # grid detector below. This keeps the old non-table exercise pipeline
    # isolated from the new table pipeline.
    return False


def _page_has_table_grid(page, page_png=None, extracted_text: str = ""):
    """Detect table-like borders without doing table OCR.

    Text structure is checked first because many textbook tables have a valid
    PDF text layer but their borders disappear at low preview DPI.
    """
    if _text_looks_like_table(extracted_text):
        return True
    try:
        drawings = page.get_drawings() if hasattr(page, "get_drawings") else []
        h = v = 0
        for d in drawings:
            for item in d.get("items", []):
                if not item or item[0] != "l" or len(item) < 3:
                    continue
                p1, p2 = item[1], item[2]
                dx = abs(float(p2.x) - float(p1.x))
                dy = abs(float(p2.y) - float(p1.y))
                if dx >= 80 and dy <= 2.5:
                    h += 1
                elif dy >= 50 and dx <= 2.5:
                    v += 1
        if h >= 2 and v >= 2:
            return True
    except Exception:
        pass
    if page_png is None:
        return False
    return _detect_long_grid_lines(page_png)


def gemini_explain_table_page(page_png: bytes, page_no: int, extracted_text: str = "", source_file: str = ""):
    """Vision extraction for original tables.

    The table image remains the visual source of truth. Vision also emits
    structured, source-grounded facts so downstream chat can reason over empty
    cells and time/day relationships instead of relying on flattened OCR.
    """
    if not gemini:
        raise RuntimeError("Gemini chưa được khởi tạo.")
    prompt = f"""Đây là trang {page_no} của tài liệu học tiếng Nhật. Trang có thể có một hoặc nhiều BẢNG.

Hãy nhìn TRỰC TIẾP vào ẢNH và tạo dữ liệu cho TỪNG BẢNG GỐC trên trang để hệ thống RAG có thể tìm kiếm, hiển thị đúng ảnh và để Doraemon có thể suy luận khi học sinh hỏi hoặc làm bài.

ĐÂY KHÔNG PHẢI TABLE OCR. Không tái tạo toàn bộ grid thành JSON row/column. ẢNH là nguồn sự thật.

Mỗi bảng trả về:
- box: vùng bao quanh CHÍNH bảng đó, [ymin, xmin, ymax, xmax] chuẩn hóa 0-1000;
- explanation: giải thích ngắn, chính xác nội dung bảng;
- facts: danh sách các FACT quan trọng đọc được trực tiếp từ bảng. Mỗi fact là một chuỗi độc lập, có đủ chủ thể/ngày/khung giờ/giá trị khi có thể xác định.

QUY TẮC BẮT BUỘC:
1. Không dùng kiến thức bên ngoài để sửa/đoán dữ liệu.
2. Giữ chính xác quan hệ giữa hàng, cột, ngày, giờ, hoạt động, ký hiệu ○/／ và các ô gộp.
3. Ô TRỐNG phải được ghi nhận khi nó có ý nghĩa trong ngữ cảnh bảng. Với bảng lịch, hãy diễn đạt rõ "Thứ X, khung giờ Y: trống/không có hoạt động ghi trong bảng" CHỈ KHI nhìn từ bố cục ảnh xác định được.
4. Không được biến ô "／" thành "trống/rảnh". "／" là ký hiệu nghỉ/không mở hoặc giá trị được thể hiện bằng chính ký hiệu đó.
5. Không nhân bản một hoạt động sang ngày/khung giờ khác chỉ vì ngữ nghĩa có vẻ hợp lý.
6. Với bảng lịch, phải gắn hoạt động vào đúng ngày và đúng khung giờ dựa trên vị trí ô; không suy ra chỉ từ thứ tự chữ OCR.
7. Với bảng giờ mở cửa, giữ nguyên ○ và ／ và nêu rõ ngày/ca tương ứng.
8. Với bảng giá/bài tập, giữ nguyên từng món, giá, số lượng hoặc lựa chọn nếu nhìn thấy.
9. Nếu một chi tiết không đọc chắc, ghi "không rõ" hoặc bỏ qua; tuyệt đối không đoán.
10. facts phải là dữ kiện nguồn, KHÔNG tự tạo đáp án cho câu hỏi của học sinh. Việc suy luận đáp án sẽ do Doraemon thực hiện từ các facts.
11. Nếu trang không có bảng rõ ràng, trả về tables=[]; không tự coi toàn bộ trang là một bảng.

TEXT TỪ PDF (chỉ hỗ trợ tìm chữ; có thể mất vị trí, KHÔNG được ưu tiên hơn ảnh):
{extracted_text[:6000]}

Chỉ trả JSON đúng schema:
{{
  "tables": [
    {{
      "box": [0,0,1000,1000],
      "explanation": "...",
      "facts": ["...", "..."]
    }}
  ]
}}"""
    part = types.Part.from_bytes(data=page_png, mime_type="image/png")
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
            response_mime_type="application/json",
        )
    )
    _log_gemini_usage(response, operation=f"vision_table_page:{source_file}:page_{page_no}")
    data = _parse_gemini_json(response.text or "{}")
    tables = data.get("tables") if isinstance(data.get("tables"), list) else []
    out=[]
    for item in tables:
        if not isinstance(item, dict):
            continue
        box=item.get("box")
        explanation=str(item.get("explanation") or "").strip()
        facts=item.get("facts") if isinstance(item.get("facts"), list) else []
        facts=[str(x).strip() for x in facts if str(x).strip()]
        if not isinstance(box, (list, tuple)) or len(box)!=4 or not explanation:
            continue
        try:
            vals=[max(0,min(1000,int(float(x)))) for x in box]
        except Exception:
            continue
        if vals[2] <= vals[0] or vals[3] <= vals[1]:
            continue
        out.append({"box": vals, "explanation": explanation, "facts": facts})
    return out

def _store_table_source_image(source_file: str, subject: str, page_meta, page_no: int, table_index: int, image_bytes: bytes, size):
    """Store ONE original table image as an independent knowledge image."""
    if not b2_ready():
        return None
    try:
        width, height = size
        key = f"images/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}/page_{page_no:04d}/table_{table_index:02d}.jpg"
        b2_put_bytes(key, image_bytes, "image/jpeg")
        primary = page_meta[0] if page_meta else {}
        conn=db()
        try:
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO knowledge_images
                    (source_file,subject,content_type,lesson,topic,page,image_key,image_url,description,width,height)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (source_file,subject,primary.get("content_type","Từ vựng"),primary.get("lesson"),primary.get("topic"),
                     page_no,key,b2_url(key),f"Ảnh gốc bảng {table_index} trong giáo trình",width,height))
            conn.commit()
        finally:
            conn.close()
        return {
            "key":key,
            "description":f"Ảnh gốc bảng {table_index} trong giáo trình",
            "page":page_no,
            "kind":"table_source",
            "table_index":table_index,
        }
    except Exception as exc:
        print(f"[TABLE IMAGE] page={page_no} table={table_index} skipped:", type(exc).__name__, str(exc))
        return None

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

def _lesson_image_is_meaningful(doc, page, xref, info, data):
    """Reject PDF image resources that are masks/backgrounds/blank white assets.

    Some educational PDFs contain many internal image XObjects that are not
    visible content (white backgrounds, masks, clipping assets, etc.).  The old
    extractor stored all of them in B2, which made one real table page produce
    many useless unused internal PDF image objects.
    """
    try:
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if width < 80 or height < 60:
            return False

        # Ignore resources that have no visible placement on this page.
        rects = page.get_image_rects(xref) if hasattr(page, "get_image_rects") else []
        if not rects:
            return False
        page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
        visible_area = max((float(r.width) * float(r.height) for r in rects), default=0.0)
        if visible_area / page_area < 0.0008:
            return False

        if Image is None:
            return True

        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((96, 96), Image.Resampling.BILINEAR)
        pixels = list(im.getdata())
        if not pixels:
            return False

        # A nearly uniform white/transparent resource is not an educational
        # image. Keep mostly-white images when they contain enough non-white
        # pixels or visible contrast (e.g. diagrams with white background).
        nonwhite = sum(1 for r,g,b in pixels if min(r,g,b) < 245) / len(pixels)
        mean = tuple(sum(px[i] for px in pixels) / len(pixels) for i in range(3))
        variance = sum(sum((px[i] - mean[i]) ** 2 for i in range(3)) for px in pixels) / (len(pixels) * 3)
        if nonwhite < 0.003 and variance < 8.0:
            return False
        return True
    except Exception:
        # Never make extraction fail because a single exotic PDF image cannot
        # be inspected. Fall back to the legacy size/data checks.
        return True


def extract_lesson_images(pdf_source, page_no: int, source_file: str, subject: str, page_meta, exclude_boxes=None):
    """Extract meaningful native lesson images as img_XX.jpg (never embedded_XX.*).

    Native PDF image resources are still used because they preserve the original
    lesson illustration quality, but they are stored under the normal img_XX.jpg
    naming convention. On table pages, exclude_boxes contains Vision-detected
    table regions so table source images are left to the table pipeline.
    """
    if fitz is None:
        return []
    if not b2_ready():
        return []
    doc = fitz.open(pdf_source) if isinstance(pdf_source, (str, os.PathLike)) else fitz.open(stream=pdf_source, filetype="pdf")
    stored=[]
    try:
        page=doc.load_page(page_no-1)
        seen=set()
        lesson_idx=0
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

            # If this native image lies inside a detected table box, do not
            # store it here. The table pipeline owns that image as table_XX.jpg.
            if exclude_boxes:
                try:
                    rects = page.get_image_rects(xref) if hasattr(page, "get_image_rects") else []
                    excluded = False
                    for rect in rects or []:
                        if float(rect.width) <= 0 or float(rect.height) <= 0:
                            continue
                        rb = [
                            max(0, min(1000, int(round(1000 * float(rect.y0) / float(page.rect.height))))),
                            max(0, min(1000, int(round(1000 * float(rect.x0) / float(page.rect.width))))),
                            max(0, min(1000, int(round(1000 * float(rect.y1) / float(page.rect.height))))),
                            max(0, min(1000, int(round(1000 * float(rect.x1) / float(page.rect.width))))) ,
                        ]
                        for box in exclude_boxes:
                            if not isinstance(box, (list, tuple)) or len(box) != 4:
                                continue
                            ymin, xmin, ymax, xmax = [max(0, min(1000, float(v))) for v in box]
                            inter_w = max(0.0, min(rb[3], xmax) - max(rb[1], xmin))
                            inter_h = max(0.0, min(rb[2], ymax) - max(rb[0], ymin))
                            inter = inter_w * inter_h
                            img_area = max(1.0, (rb[3]-rb[1]) * (rb[2]-rb[0]))
                            if inter / img_area >= 0.45:
                                excluded = True
                                break
                        if excluded:
                            break
                    if excluded:
                        print(f"[LESSON IMAGE skip-table] page={page_no} xref={xref}")
                        continue
                except Exception as exc:
                    print(f"[LESSON IMAGE bbox skip] page={page_no} xref={xref}: {type(exc).__name__} {exc}")

            if not _lesson_image_is_meaningful(doc, page, xref, info, data):
                print(f"[LESSON IMAGE skip] page={page_no} xref={xref} size={info.get('width')}x{info.get('height')}")
                continue

            # Normalize every lesson illustration to JPEG. We deliberately do
            # not create any embedded_XX.* objects.
            width, height = int(info.get("width") or 0), int(info.get("height") or 0)
            if Image is not None:
                try:
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    out = io.BytesIO()
                    im.save(out, format="JPEG", quality=92, optimize=True)
                    data = out.getvalue()
                    width, height = im.size
                except Exception:
                    pass
            lesson_idx += 1
            key=f"images/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}/page_{page_no:04d}/img_{lesson_idx:02d}.jpg"
            b2_put_bytes(key,data,"image/jpeg")
            primary=page_meta[0] if page_meta else {}
            conn=db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO knowledge_images
                        (source_file,subject,content_type,lesson,topic,page,image_key,image_url,description,width,height)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (source_file,subject,primary.get("content_type","Từ vựng"),primary.get("lesson"),primary.get("topic"),
                         page_no,key,b2_url(key),"Lesson image",width,height))
                conn.commit()
            finally:
                conn.close()
            stored.append({"key":key,"description":"Lesson image","page":page_no,"width":width,"height":height})
    finally:
        doc.close()
    return stored

def process_pdf_pages(pdf_source, reader, records_meta, source_file: str, subject: str, selected_pages=None):
    """Extract text/images using the V16 baseline, plus semantic Vision text for table pages.

    IMPORTANT: no table OCR/grid reconstruction is performed here. For a page
    that is visually a table, the original rendered page is stored as an image,
    and one additional Vision call creates a natural-language explanation that
    is appended to the page text before chunking/embedding.
    """
    page_texts = {}
    page_images = {}
    page_units = {}
    if selected_pages is not None:
        selected = {int(x) for x in (selected_pages or [])}
        if not selected:
            raise ValueError('selected_pages không được rỗng khi tạo AI Draft.')
    else:
        selected = None
    for page_no, page in enumerate(reader.pages, 1):
        if selected is not None and page_no not in selected:
            continue
        page_meta = metadata_for_page(records_meta, page_no)
        extracted = (page.extract_text() or "").strip()
        text_len = len(re.sub(r"\s+", "", extracted))

        # Keep the original V16 fast path for ordinary text pages.
        if text_len >= 30:
            # A low-cost local visual check is used only to detect whether an
            # actual table is present. It does NOT perform OCR. PdfReader's
            # PageObject has no drawing API, so use a tiny rendered preview.
            preview = render_pdf_page(pdf_source, page_no, dpi=72)
            table_page = _page_has_table_grid(page, preview, extracted)
            if not table_page:
                page_texts[page_no] = extracted
                lesson_images = extract_lesson_images(pdf_source, page_no, source_file, subject, page_meta)
                if lesson_images:
                    page_images[page_no] = lesson_images
                continue
        else:
            table_page = False

        # Scan/low-text pages retain the old Gemini OCR behavior.
        png = render_pdf_page(pdf_source, page_no, dpi=140 if table_page else 120)
        ocr_text = extracted
        if not table_page:
            table_page = _page_has_table_grid(page, png, ocr_text or extracted)

        stored = []
        if text_len < 30:
            ocr_text, detected = gemini_ocr_page(png, page_no, source_file=source_file)
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

        # For table pages, keep EACH ORIGINAL TABLE as its own image and add
        # one semantic Vision explanation per table to the text that is embedded in the RAG chunk.
        # IMPORTANT: native lesson illustrations must be extracted
        # BEFORE building page_units. V16.6.3 computed lesson_image_keys first
        # and only then appended lesson images, so the normal text unit got
        # image_keys=[] even though the images were already stored in B2.
        if table_page:
            table_items = gemini_explain_table_page(png, page_no, extracted_text=ocr_text or extracted, source_file=source_file)

            # Extract only non-table native illustrations as normal lesson images.
            # Table regions are excluded by their Vision boxes; table images are
            # created separately by _store_table_source_image below.
            table_boxes = [item.get("box") for item in table_items if isinstance(item, dict) and isinstance(item.get("box"), (list, tuple))]
            lesson_images = extract_lesson_images(
                pdf_source, page_no, source_file, subject, page_meta, exclude_boxes=table_boxes
            )
            if lesson_images:
                for emb in lesson_images:
                    emb["image_scope"] = "lesson"
                stored.extend(lesson_images)

            table_parts=[]
            for table_index, item in enumerate(table_items, 1):
                marker = f"【GIẢI THÍCH BẢNG TRANG {page_no} #{table_index}】"
                table_parts.append(marker + "\n" + str(item.get("explanation") or "").strip())
                cropped = crop_image_from_page(png, item.get("box"))
                if cropped:
                    image_bytes, size = cropped
                else:
                    # If Vision could explain a table but its box is unusable,
                    # do not lose the table explanation; keep the whole page as
                    # the visual fallback for that table.
                    im = Image.open(io.BytesIO(png)).convert("RGB") if Image is not None else None
                    if im is None:
                        continue
                    outbuf=io.BytesIO(); im.save(outbuf, format="JPEG", quality=92, optimize=True)
                    image_bytes, size = outbuf.getvalue(), im.size
                table_image = _store_table_source_image(
                    source_file, subject, page_meta, page_no, table_index, image_bytes, size
                )
                if table_image:
                    table_image["marker"] = marker
                    table_image["explanation"] = str(item.get("explanation") or "").strip()
                    table_image["facts"] = list(item.get("facts") or [])
                    table_image["unit_id"] = f"table:{page_no}:{table_index}"
                    stored.append(table_image)

            base_text=(ocr_text or extracted).strip()
            # Provenance-bearing content units. Table explanation and its source
            # image stay together before chunking; no post-chunk text matching.
            units=[]
            # Ordinary lesson illustrations on a table page belong to the
            # normal page/lesson context, not to any individual table.
            # Build this list from BOTH the in-memory extraction result and
            # the persisted knowledge_images rows. The latter is an important
            # recovery path when the same PDF/source_file was uploaded before
            # and the B2 object already exists. Presence in B2 alone must not
            # be treated as provenance; the DB row is the authoritative key.
            lesson_image_keys = [
                str(x.get("key") or "").strip()
                for x in stored
                if str(x.get("image_scope") or "").strip().lower() == "lesson"
                and str(x.get("key") or "").strip()
            ]
            try:
                conn=db()
                try:
                    with conn.cursor() as cur:
                        cur.execute("""SELECT image_key FROM knowledge_images
                            WHERE source_file=%s AND page=%s
                              AND (description='Lesson image' OR image_key LIKE %s)""",
                            (source_file, page_no, f"images/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}/page_{page_no:04d}/img_%%"))
                        for row in cur.fetchall() or []:
                            key=str(row[0] or "").strip()
                            if key:
                                lesson_image_keys.append(key)
                finally:
                    conn.close()
            except Exception as e:
                print(f"[LESSON IMAGE DB RECOVERY skip] page={page_no}: {e}")
            lesson_image_keys = list(dict.fromkeys(lesson_image_keys))
            print(f"[LESSON IMAGE LINK] page={page_no} keys={lesson_image_keys}")
            if base_text:
                units.append({
                    "type":"normal",
                    "unit_id":f"page:{page_no}:text",
                    "text":base_text,
                    "image_keys":lesson_image_keys,
                })
            for table_image in [x for x in stored if str(x.get("kind") or "") == "table_source"]:
                explanation=str(table_image.get("explanation") or "").strip()
                facts=[str(x).strip() for x in (table_image.get("facts") or []) if str(x).strip()]
                marker=str(table_image.get("marker") or "").strip()
                key=str(table_image.get("key") or "").strip()
                if explanation and key:
                    fact_text = "\n".join(f"- {x}" for x in facts)
                    unit_text = marker + "\n" + explanation
                    if fact_text:
                        unit_text += "\nFACTS NGUỒN CỦA BẢNG:\n" + fact_text
                    units.append({
                        "type":"table",
                        "unit_id":str(table_image.get("unit_id") or ""),
                        "text":unit_text.strip(),
                        "image_keys":[key],
                    })
            page_units[page_no]=units
            page_texts[page_no] = "\n\n".join(u["text"] for u in units).strip()

            print(f"[TABLE VISUAL] page={page_no} tables={len(table_items)} source_images={sum(1 for x in stored if x.get('kind')=='table_source')} lesson_images={sum(1 for x in stored if x.get('image_scope')=='lesson')} lesson_image_keys={lesson_image_keys}")
        else:
            page_texts[page_no] = (ocr_text or extracted).strip()

        if stored:
            page_images[page_no] = stored
        # Release large rendered buffers before processing the next page.
        try:
            del preview
        except Exception:
            pass
        try:
            del png
        except Exception:
            pass
        try:
            del page
        except Exception:
            pass
        gc.collect()
    return page_texts, page_images, page_units

@app.post("/admin/api/knowledge/upload")
async def admin_knowledge_upload(
    background_tasks: BackgroundTasks,
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

    source_file=os.path.basename(file.filename)
    namespace="__default__"
    temp_pdf_path = None
    raw_size = 0
    sha = hashlib.sha256()
    try:
        # Stream the upload to disk instead of keeping the whole PDF in RAM.
        with tempfile.NamedTemporaryFile(prefix="doraemon_upload_", suffix=".pdf", delete=False) as tf:
            temp_pdf_path = tf.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                raw_size += len(chunk)
                if raw_size > 50 * 1024 * 1024:
                    raise HTTPException(400, "File quá lớn. Giới hạn 50 MB.")
                sha.update(chunk)
                tf.write(chunk)
        source_hash = sha.hexdigest()
        if not temp_pdf_path:
            raise HTTPException(400, "Không nhận được file PDF.")
        reader=PdfReader(temp_pdf_path)
        records_meta=normalize_kb_records(metadata_json,len(reader.pages))
    except HTTPException:
        if temp_pdf_path:
            try: os.unlink(temp_pdf_path)
            except Exception: pass
        raise
    except ValueError as e:
        if temp_pdf_path:
            try: os.unlink(temp_pdf_path)
            except Exception: pass
        raise HTTPException(400,str(e))
    except Exception as e:
        if temp_pdf_path:
            try: os.unlink(temp_pdf_path)
            except Exception: pass
        raise HTTPException(400,f"Không đọc được PDF: {e}")

    # Save original PDF to B2 without copying the full PDF into RAM.
    pdf_key = f"pdf/{re.sub(r'[^A-Za-z0-9_.-]+','_',source_file)}"
    pdf_url = None
    if b2_ready():
        try:
            pdf_url = b2_put_file(pdf_key, temp_pdf_path, "application/pdf")
        except Exception as e:
            try: os.unlink(temp_pdf_path)
            except Exception: pass
            raise HTTPException(500, f"Không lưu được PDF vào B2: {e}")

    # Extract text normally; scanned/empty pages go through Gemini OCR.
    try:
        page_texts, page_images, page_units = process_pdf_pages(temp_pdf_path, reader, records_meta, source_file, subject)
    except Exception as e:
        try: os.unlink(temp_pdf_path)
        except Exception: pass
        raise HTTPException(500, f"OCR Gemini/xử lý ảnh thất bại: {e}")

    vectors=[]
    total=0
    image_vectors_total=0
    knowledge_cache_text_records=[]
    knowledge_cache_image_records=[]
    try:
        # Khi re-upload cùng source_file, xóa các vector cũ của tài liệu này
        # để tránh record V1 cũ tiếp tục cạnh tranh với record V2 mới.
        try:
            index.delete(filter={"source_file": source_file}, namespace=namespace)
        except Exception as e:
            print("Pinecone old-source cleanup skipped:", type(e).__name__, str(e))

        lesson_chunk_counters = {}
        for page_no in range(1, len(reader.pages)+1):
            text = page_texts.get(page_no, "")
            page_meta=metadata_for_page(records_meta,page_no)
            primary=page_meta[0] if page_meta else None
            content_type=primary["content_type"] if primary else (records_meta[0]["content_type"] if records_meta else "Từ vựng")
            units = page_units.get(page_no) or []

            # Table pages use provenance-bearing units. Each table explanation is
            # chunked independently and every resulting chunk inherits the exact
            # table image key. Normal text on the same page remains a separate
            # unit and does not inherit table images.
            if units and any(u.get("type") == "table" for u in units):
                chunk_records=[]
                for unit in units:
                    unit_text=str(unit.get("text") or "").strip()
                    if not unit_text:
                        continue
                    unit_chunks=kb_chunk_text(unit_text,chunk_size,overlap)
                    unit_id=str(unit.get("unit_id") or "")
                    unit_image_keys=list(dict.fromkeys(str(k).strip() for k in (unit.get("image_keys") or []) if str(k).strip()))
                    # Hard fallback for the normal text unit on mixed pages:
                    # if page_units was serialized without lesson keys, derive
                    # them again from the page image records before embedding.
                    if str(unit.get("type") or "").strip().lower() == "normal" and not unit_image_keys:
                        unit_image_keys = list(dict.fromkeys(
                            str(x.get("key") or "").strip()
                            for x in (page_images.get(page_no, []) or [])
                            if str(x.get("image_scope") or "").strip().lower() == "lesson"
                            and str(x.get("key") or "").strip()
                        ))
                        if unit_image_keys:
                            print(f"[LESSON IMAGE CHUNK FALLBACK] page={page_no} keys={unit_image_keys}")
                    lesson_key_for_index = _canonical_lesson_key((page_meta[0] if page_meta else {}).get("lesson") if page_meta else "")
                    if not lesson_key_for_index:
                        lesson_key_for_index = "__unassigned__"
                    for local_no, chunk in enumerate(unit_chunks):
                        global_idx = lesson_chunk_counters.get(lesson_key_for_index, 0)
                        lesson_chunk_counters[lesson_key_for_index] = global_idx + 1
                        chunk_records.append({
                            "chunk_index": global_idx,
                            "page_chunk_index": len(chunk_records),
                            "local_index": local_no,
                            "unit_id": unit_id,
                            "text": chunk,
                            "image_keys": unit_image_keys,
                        })

                for rec in chunk_records:
                    chunk=rec["text"]
                    md_list=[{
                        "content_type":r["content_type"],"lesson":r["lesson"],"lesson_pages":r["lesson_pages"],
                        "topic":r["topic"],"topic_pages":r["topic_pages"],
                        "question_pages":r["question_pages"],"answer_pages":r["answer_pages"]
                    } for r in page_meta]
                    md={
                        "record_type":"text",
                        "text":chunk,"course":subject,"subject":subject,"content_type":content_type,
                        "source_file":source_file,"page":page_no,"chunk_index":rec["chunk_index"],
                        "metadata_records":json.dumps(md_list,ensure_ascii=False),
                        "image_keys":json.dumps(rec["image_keys"],ensure_ascii=False),
                        "content_unit_id":rec["unit_id"],
                    }
                    if primary:
                        md.update({
                            "lesson":primary["lesson"],"lesson_pages":primary["lesson_pages"],
                            "topic":primary["topic"],"topic_pages":primary["topic_pages"],
                            "question_pages":primary["question_pages"],"answer_pages":primary["answer_pages"]
                        })
                    if rec["unit_id"].startswith("page:"):
                        print(f"[PINECONE PAGE CHUNK] page={page_no} chunk={rec['chunk_index']} image_keys={rec['image_keys']}")
                    knowledge_cache_text_records.append({"text": chunk, "metadata": dict(md), "image_keys": list(rec["image_keys"])})
                    vectors.append({"id":uuid.uuid4().hex,"values":embed_text(chunk),"metadata":md})
                    total+=1

                # Build exact chunk provenance for each image record.
                unit_chunk_map={}
                for rec in chunk_records:
                    unit_chunk_map.setdefault(rec["unit_id"], []).append(int(rec["chunk_index"]))

                for img in page_images.get(page_no, []):
                    key=str(img.get("key") or "").strip()
                    if not key:
                        continue
                    term=str(img.get("term") or "").strip()
                    reading=str(img.get("reading") or "").strip()
                    meaning=str(img.get("meaning") or "").strip()
                    associated_text=str(img.get("associated_text") or "").strip()
                    description=str(img.get("description") or "").strip()
                    unit_id=str(img.get("unit_id") or "")
                    table_explanation=str(img.get("explanation") or "").strip()
                    table_facts=" | ".join(str(x).strip() for x in (img.get("facts") or []) if str(x).strip())
                    search_text=" | ".join(x for x in [term,reading,meaning,associated_text,description,table_explanation,table_facts,f"Trang {page_no}"] if x)
                    if not search_text:
                        search_text=f"Hình minh họa trang {page_no}"

                    image_md={
                        "record_type":"image",
                        "text":search_text,
                        "course":subject,"subject":subject,"content_type":content_type,
                        "source_file":source_file,"page":page_no,
                        "image_key":key,"image_url":b2_url(key),
                        "term":term,"reading":reading,"meaning":meaning,
                        "associated_text":associated_text,"description":description,
                        "bbox":str(img.get("bbox") or ""),
                        "image_kind":str(img.get("kind") or "educational_image"),
                        "image_scope":str(img.get("image_scope") or ("table" if img.get("kind") == "table_source" else "chunk")),
                        "table_facts":json.dumps(img.get("facts") or [],ensure_ascii=False),
                    }

                    if unit_id and unit_id in unit_chunk_map:
                        matched_chunks=unit_chunk_map[unit_id]
                        if matched_chunks:
                            image_md["chunk_index"]=int(matched_chunks[0])
                            image_md["chunk_indices"]=json.dumps(matched_chunks,ensure_ascii=False)
                            image_md["content_unit_id"]=unit_id
                    elif len(chunk_records) == 1 and str(img.get("image_scope") or "").strip().lower() != "lesson":
                        image_md["chunk_index"]=0

                    if img.get("chunk_index") not in (None,"") and str(img.get("image_scope") or "").strip().lower() != "lesson":
                        try:
                            image_md["chunk_index"]=int(img.get("chunk_index"))
                        except Exception:
                            image_md["chunk_index"]=str(img.get("chunk_index")).strip()

                    if primary:
                        image_md.update({
                            "lesson":primary["lesson"],"topic":primary["topic"],
                            "lesson_pages":primary["lesson_pages"],"topic_pages":primary["topic_pages"]
                        })
                    knowledge_cache_image_records.append({"metadata": dict(image_md)})
                    vectors.append({"id":uuid.uuid4().hex,"values":embed_text(search_text),"metadata":image_md})
                    if image_md.get("image_scope") == "lesson":
                        print("[IMAGE UPSERT lesson]", {
                            "source_file": source_file, "page": page_no,
                            "lesson": image_md.get("lesson"),
                            "key": key, "chunk_index": image_md.get("chunk_index"),
                            "image_scope": image_md.get("image_scope")
                        })
                    image_vectors_total+=1

                if len(vectors)>=50:
                    index.upsert(vectors=vectors,namespace=namespace)
                    vectors=[]
                continue

            # Non-table pages: keep the original V16/V16.3 chunk and image
            # mapping path unchanged.
            chunks=kb_chunk_text(text,chunk_size,overlap)
            lesson_key_for_index = _canonical_lesson_key((primary or {}).get("lesson") if primary else "")
            if not lesson_key_for_index:
                lesson_key_for_index = "__unassigned__"
            for page_chunk_no,chunk in enumerate(chunks):
                chunk_no = lesson_chunk_counters.get(lesson_key_for_index, 0)
                lesson_chunk_counters[lesson_key_for_index] = chunk_no + 1
                md_list=[{
                    "content_type":r["content_type"],"lesson":r["lesson"],"lesson_pages":r["lesson_pages"],
                    "topic":r["topic"],"topic_pages":r["topic_pages"],
                    "question_pages":r["question_pages"],"answer_pages":r["answer_pages"]
                } for r in page_meta]
                # Preserve lesson-scope illustration provenance on non-table text chunks.
                # The previous path hard-coded image_keys=[] and dropped cached lesson images.
                lesson_image_keys = list(dict.fromkeys(
                    str(x.get("key") or "").strip()
                    for x in (page_images.get(page_no, []) or [])
                    if str(x.get("image_scope") or "").strip().lower() == "lesson"
                    and str(x.get("key") or "").strip()
                ))
                md={
                    "record_type":"text",
                    "text":chunk,"course":subject,"subject":subject,"content_type":content_type,
                    "source_file":source_file,"page":page_no,"chunk_index":chunk_no,"page_chunk_index":page_chunk_no,
                    "metadata_records":json.dumps(md_list,ensure_ascii=False),
                    "image_keys":json.dumps(lesson_image_keys,ensure_ascii=False)
                }
                if primary:
                    md.update({
                        "lesson":primary["lesson"],"lesson_pages":primary["lesson_pages"],
                        "topic":primary["topic"],"topic_pages":primary["topic_pages"],
                        "question_pages":primary["question_pages"],"answer_pages":primary["answer_pages"]
                    })
                knowledge_cache_text_records.append({"text": chunk, "metadata": dict(md), "image_keys": lesson_image_keys})
                vectors.append({"id":uuid.uuid4().hex,"values":embed_text(chunk),"metadata":md})
                total+=1

            page_chunk_count=len(chunks)
            for img in page_images.get(page_no, []):
                key=str(img.get("key") or "").strip()
                if not key:
                    continue
                term=str(img.get("term") or "").strip()
                reading=str(img.get("reading") or "").strip()
                meaning=str(img.get("meaning") or "").strip()
                associated_text=str(img.get("associated_text") or "").strip()
                description=str(img.get("description") or "").strip()
                search_text=" | ".join(x for x in [term,reading,meaning,associated_text,description,f"Trang {page_no}"] if x)
                if not search_text:
                    search_text=f"Hình minh họa trang {page_no}"
                image_md={
                    "record_type":"image","text":search_text,
                    "course":subject,"subject":subject,"content_type":content_type,
                    "source_file":source_file,"page":page_no,
                    "image_key":key,"image_url":b2_url(key),
                    "term":term,"reading":reading,"meaning":meaning,
                    "associated_text":associated_text,"description":description,
                    "bbox":str(img.get("bbox") or ""),
                    "image_kind":str(img.get("kind") or "educational_image"),
                    "image_scope":str(img.get("image_scope") or "chunk"),
                }
                if page_chunk_count==1 and str(img.get("image_scope") or "").strip().lower() != "lesson":
                    image_md["chunk_index"]=0
                if img.get("chunk_index") not in (None,"") and str(img.get("image_scope") or "").strip().lower() != "lesson":
                    try:
                        image_md["chunk_index"]=int(img.get("chunk_index"))
                    except Exception:
                        image_md["chunk_index"]=str(img.get("chunk_index")).strip()
                if primary:
                    image_md.update({
                        "lesson":primary["lesson"],"topic":primary["topic"],
                        "lesson_pages":primary["lesson_pages"],"topic_pages":primary["topic_pages"]
                    })
                knowledge_cache_image_records.append({"metadata": dict(image_md)})
                vectors.append({"id":uuid.uuid4().hex,"values":embed_text(search_text),"metadata":image_md})
                image_vectors_total+=1

            if len(vectors)>=50:
                index.upsert(vectors=vectors,namespace=namespace)
                vectors=[]
        if vectors:
            index.upsert(vectors=vectors,namespace=namespace)

    except Exception as e:
        try: os.unlink(temp_pdf_path)
        except Exception: pass
        raise HTTPException(500,f"Lỗi embedding/Pinecone: {e}")

    conn=db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_documents WHERE source_file=%s", (source_file,))
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
    try:
        cache_lessons = _upsert_upload_knowledge_cache(
            source_file, source_hash, subject, len(reader.pages),
            knowledge_cache_text_records, knowledge_cache_image_records,
            defer_curriculum_plan=True
        )
    except Exception as exc:
        print("[KNOWLEDGE CACHE] build failed:", type(exc).__name__, str(exc))
        try: os.unlink(temp_pdf_path)
        except Exception: pass
        raise HTTPException(500, f"Tạo Knowledge Cache thất bại: {exc}")

    image_count=sum(len(v) for v in page_images.values())
    scanned_pages=sum(1 for p in range(1,len(reader.pages)+1) if len(re.sub(r"\s+","",(reader.pages[p-1].extract_text() or ""))) < 30)
    table_source_images=sum(1 for imgs in page_images.values() for img in imgs if str(img.get("kind") or "") == "table_source")
    if cache_lessons:
        print(f"[CURRICULUM PLAN DEFERRED] source={source_file!r} lessons={cache_lessons}; HIGH planner will run lazily on first curriculum use")

    page_count = len(reader.pages)
    record_count = len(records_meta)
    # Explicitly release upload-time working sets before returning to Render.
    try:
        del page_texts, page_images, page_units, knowledge_cache_text_records, knowledge_cache_image_records, vectors, reader, records_meta
    except Exception:
        pass
    gc.collect()
    try:
        if temp_pdf_path:
            os.unlink(temp_pdf_path)
    except Exception:
        pass

    return {"success":True,"filename":source_file,"subject":subject,
            "pages":page_count,"scanned_pages_ocr":scanned_pages,"chunks":total,"records":record_count,
            "images":image_count,"image_vectors":image_vectors_total,"table_source_images":table_source_images,
            "knowledge_cache_lessons": cache_lessons, "knowledge_cache_hash": source_hash,
            "curriculum_plan_status": "PLANNING" if cache_lessons else "READY",
            "pdf_url":pdf_url,"dimension":768,"index":PINECONE_INDEX,"namespace":namespace}

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

@app.get("/admin/api/payment-packages")
def admin_payment_packages(password: str):
    check_admin(password)
    conn = db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT months,plan_name,price_vnd,qr_key,updated_at FROM payment_packages WHERE months IN (1,3,6) ORDER BY months")
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return {"packages":[{
        "months": int(r["months"]),
        "plan_name": r["plan_name"],
        "price_vnd": int(r["price_vnd"] or 0),
        "qr_key": r.get("qr_key"),
        "qr_url": b2_url(r.get("qr_key")) if r.get("qr_key") else None,
        "updated_at": r.get("updated_at")
    } for r in rows]}


@app.post("/admin/api/payment-packages/{months}")
async def admin_payment_package(months: int, password: str = Form(...), price_vnd: int = Form(0), qr_file: UploadFile | None = File(None)):
    check_admin(password)
    if months not in (1,3,6):
        raise HTTPException(400, "Chỉ hỗ trợ gói 1, 3 hoặc 6 tháng.")
    if price_vnd < 0:
        raise HTTPException(400, "Giá gói không được âm.")
    qr_key = None
    if qr_file is not None and qr_file.filename:
        content_type = (qr_file.content_type or "").lower()
        if content_type not in {"image/png","image/jpeg","image/webp"}:
            raise HTTPException(400, "QR code phải là PNG, JPG hoặc WEBP.")
        data = await qr_file.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(400, "Ảnh QR tối đa 5MB.")
        if not b2_ready():
            raise HTTPException(500, "Backblaze B2 chưa được cấu hình nên chưa thể lưu QR.")
        ext = {"image/png":"png","image/jpeg":"jpg","image/webp":"webp"}.get(content_type, "png")
        qr_key = f"payments/qr_{months}_month.{ext}"
        b2_put_bytes(qr_key, data, content_type)
    conn = db()
    try:
        with conn.cursor() as cur:
            if qr_key:
                cur.execute("UPDATE payment_packages SET price_vnd=%s,qr_key=%s,updated_at=NOW() WHERE months=%s", (price_vnd,qr_key,months))
            else:
                cur.execute("UPDATE payment_packages SET price_vnd=%s,updated_at=NOW() WHERE months=%s", (price_vnd,months))
            if cur.rowcount == 0:
                cur.execute("INSERT INTO payment_packages(months,plan_name,price_vnd,qr_key) VALUES(%s,%s,%s,%s)", (months,f"{months} tháng",price_vnd,qr_key))
        conn.commit()
    finally:
        conn.close()
    return {"success":True,"months":months,"price_vnd":price_vnd,"qr_key":qr_key}


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
                "expires_at_vn": _vn_display(raw_exp) if paid_active else None,
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
            old=cur.fetchone(); now=_now_local()
            old_plan = str(old.get("plan") or "Free") if old else "Free"
            start = old["expires_at"] if old and old_plan != "Free" and old["expires_at"] and old["expires_at"]>now else now
            exp=_add_calendar_months(start, months)
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
    return {"success":True,"expires_at":exp,"expires_at_vn":_vn_display(exp),"timezone":"Asia/Ho_Chi_Minh"}



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

@app.post("/admin/api/users/{user_id}/reset-free")
def admin_reset_free(user_id:int,data:dict):
    """Return a user to a clean Free plan immediately.

    The operation is authoritative: any existing paid subscription rows are
    marked EXPIRED, then a fresh Free subscription row is inserted.
    Today's Free quota is also reset to 0 so the user gets a fresh 5/5 today.
    Learning progress and admin chat are intentionally preserved.
    """
    check_admin(str(data.get("password","")))
    now = _now_local()
    today = now.date()
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE id=%s", (user_id,))
            if not cur.fetchone():
                raise HTTPException(404,"Không tìm thấy user.")

            # Close every older subscription so there is no ambiguity about
            # which package is active. Keep rows for audit/history.
            cur.execute(
                "UPDATE subscriptions SET status='EXPIRED', expires_at=COALESCE(expires_at,%s) WHERE user_id=%s AND status <> 'EXPIRED'",
                (now, user_id)
            )

            # Insert a new authoritative Free subscription as the latest row.
            cur.execute(
                """INSERT INTO subscriptions(user_id,plan,started_at,expires_at,status)
                   VALUES(%s,'Free',%s,NULL,'ACTIVE')
                   RETURNING id,plan,started_at,expires_at,status""",
                (user_id, now)
            )
            sub_row = cur.fetchone()

            # A reset-to-Free should start the current Vietnam day with the
            # full 5 questions available. Do not touch learning progress/chat.
            cur.execute(
                """INSERT INTO daily_question_usage(user_id,usage_date,question_count)
                   VALUES(%s,%s,0)
                   ON CONFLICT(user_id,usage_date) DO UPDATE
                   SET question_count=0""",
                (user_id, today)
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "success": True,
        "plan": "Free",
        "daily_limit": 5,
        "used_today": 0,
        "remaining_today": 5,
        "expires_at": None,
        "timezone": "Asia/Ho_Chi_Minh",
        "subscription_id": sub_row[0] if sub_row else None,
    }


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
    return {
        "status": "ok",
        "pinecone": index is not None,
        "gemini": gemini is not None,
        "openai": openai_client is not None,
        "llm_provider": LLM_PROVIDER,
        "database": bool(DATABASE_URL),
        "learning_engine": True,
        "content_types": sorted(CONTENT_TYPES),
        "gemini_model": GEMINI_MODEL,
        "openai_model_low": OPENAI_MODEL_LOW,
        "openai_model_medium": OPENAI_MODEL_MEDIUM,
    }
