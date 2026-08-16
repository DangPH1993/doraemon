import ctypes
import json
import os
import random
import re
import sys
import threading
import websocket
import time
import urllib.parse
import webbrowser
import winreg
import io
import base64
import requests
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

SERVER_BASE_URL = "https://doraemon-pro.onrender.com"
PROXY_URL = SERVER_BASE_URL + "/api/proxy-chat"
REGISTER_URL = SERVER_BASE_URL + "/auth/register"
LOGIN_URL = SERVER_BASE_URL + "/auth/login"
ME_URL = SERVER_BASE_URL + "/auth/me"
ADMIN_HISTORY_URL = SERVER_BASE_URL + "/admin-chat/history"
ADMIN_SEND_URL = SERVER_BASE_URL + "/admin-chat/send"
WELCOME_URL = SERVER_BASE_URL + "/session/welcome"
RESET_LEARNING_URL = SERVER_BASE_URL + "/learning/reset"
WS_USER_URL = SERVER_BASE_URL.replace("https://", "wss://") + "/ws/user"
CONFIG_FILE = "config.json"
KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "DoraemonDesktopPet"
APP_VERSION = "2026-08-16-doraemon-baseline-v5.2-admin-chat-stability"

def set_autostart(enable):
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY_PATH, 0, winreg.KEY_SET_VALUE | winreg.KEY_READ)
        if enable:
            if getattr(sys, 'frozen', False):
                app_path = f'"{sys.executable}"'
            else:
                app_path = f'"{sys.executable}" "{os.path.abspath(sys.argv[0])}"'
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, app_path)
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass
        winreg.CloseKey(key)
    except: pass

def load_config():
    default_config = {
        "client_token": "",
        "access_token": "",
        "phone": "",
        "nickname": "",
        "auto_start": False,
        "auto_chat": False,
        "proactive_last_at": 0,
        "proactive_day": "",
        "proactive_count": 0
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return default_config

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

class SmoothVectorSpeechBubble(tk.Frame):
    def __init__(self, parent, width=340, height=360, bg="#FFFFFF", **kwargs):
        super().__init__(parent, bg=parent['bg'])
        self.width = width
        self.height = height
        
        self.bg_label = tk.Label(self, bg=parent['bg'], bd=0, highlightthickness=0)
        self.bg_label.pack(fill="both", expand=True)
        
        self.update_bubble_image()
        
        self.text_frame = tk.Frame(self, bg=bg)
        # Fix tràn chữ: Giới hạn chiều cao text_frame nhỏ lại, để khoảng trống an toàn cho đuôi bong bóng
        self.text_frame.place(x=16, y=12, width=width-32, height=max(120, height-42))

        # Smooth scrolling state. Accumulate wheel input into a target and
        # animate toward it without blocking Tkinter.
        self._scroll_animating = False
        self._scroll_after_id = None
        self._scroll_target = 0.0

        self.text_widget = tk.Text(
            self.text_frame, bg=bg, fg="#1F2937", bd=0, highlightthickness=0,
            wrap="word",
            font=('Segoe UI', 10),
            # Nội dung dài/ảnh được cuộn bằng scrollbar dọc.
            **kwargs
        )
        self.scrollbar = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)
        
        self.text_widget.pack(side="left", fill="both", expand=True, padx=(2, 2), pady=0)
        self.scrollbar.pack(side="right", fill="y", pady=0)
        self.image_refs = []

        self.text_widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.text_frame.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.text_widget.bind("<Button-4>", self._on_mousewheel, add="+")
        self.text_widget.bind("<Button-5>", self._on_mousewheel, add="+")
        self.text_frame.bind("<Button-4>", self._on_mousewheel, add="+")
        self.text_frame.bind("<Button-5>", self._on_mousewheel, add="+")

    def bind_scroll_widget(self, widget):
        try:
            widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_mousewheel, add="+")
            widget.bind("<Button-5>", self._on_mousewheel, add="+")
        except Exception:
            pass

    def _on_mousewheel(self, event):
        try:
            first, last = self.text_widget.yview()
            if last <= first:
                return "break"

            if getattr(event, "num", None) == 4:
                direction = -1
            elif getattr(event, "num", None) == 5:
                direction = 1
            else:
                delta = getattr(event, "delta", 0)
                if not delta:
                    return "break"
                direction = -1 if delta > 0 else 1

            current = float(self.text_widget.yview()[0])
            if not self._scroll_animating:
                self._scroll_target = current

            visible = max(0.05, float(last - first))
            step = max(0.025, min(0.14, visible * 0.40))
            self._scroll_target += direction * step
            self._scroll_target = max(
                0.0, min(1.0 - visible, self._scroll_target)
            )

            if not self._scroll_animating:
                self._scroll_animating = True
                self._animate_scroll()

            return "break"
        except Exception:
            return "break"

    def _animate_scroll(self):
        try:
            first, _ = self.text_widget.yview()
            current = float(first)
            target = float(self._scroll_target)
            distance = target - current

            if abs(distance) < 0.002:
                self.text_widget.yview_moveto(target)
                self._scroll_animating = False
                self._scroll_after_id = None
                return

            next_pos = current + distance * 0.30
            self.text_widget.yview_moveto(next_pos)
            self._scroll_after_id = self.after(12, self._animate_scroll)
        except Exception:
            self._scroll_animating = False
            self._scroll_after_id = None

    def stop_scroll_animation(self):
        try:
            if self._scroll_after_id is not None:
                self.after_cancel(self._scroll_after_id)
        except Exception:
            pass
        self._scroll_after_id = None
        self._scroll_animating = False

    def set_height(self, height):
        """Resize the bubble, redraw its border/tail, and resize the text area to stay inside the border."""
        self.height = max(240, min(int(height), 620))
        # Keep text/content strictly inside the rounded rectangle, leaving room for the tail.
        content_h = max(120, self.height - 42)
        self.text_frame.place(x=16, y=12, width=self.width-32, height=content_h)
        self.update_bubble_image()
        self.update_idletasks()

    def update_bubble_image(self):
        scale = 3
        w = self.width * scale
        h = self.height * scale
        
        # Vẽ trên nền trong suốt hoàn toàn (Alpha = 0) để không bị lai màu
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        margin = 4 * scale
        box_w = w - margin * 2
        tail_h = 16 * scale
        box_h = h - margin * 2 - tail_h
        r = 16 * scale
        
        draw.rounded_rectangle(
            [margin, margin, margin + box_w, margin + box_h],
            radius=r,
            fill="#FFFFFF",
            outline="#1F3A60",
            width=2 * scale
        )
        
        tail_x = int(box_w * 0.35) + margin
        tail_w = 9 * scale
        
        tail_points = [
            (tail_x - tail_w, margin + box_h - scale),
            (tail_x, margin + box_h + tail_h),
            (tail_x + tail_w, margin + box_h - scale)
        ]
        draw.polygon(tail_points, fill="#FFFFFF", outline="#1F3A60", width=2 * scale)
        
        # Xóa đường gạch ngang ở cuống đuôi bong bóng
        draw.polygon([
            (tail_x - tail_w + 2*scale, margin + box_h - 2*scale),
            (tail_x, margin + box_h + tail_h - 3*scale),
            (tail_x + tail_w - 2*scale, margin + box_h - 2*scale)
        ], fill="#FFFFFF")
        
        img_resized = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Kỹ thuật Alpha Thresholding: Cắt viền sắc nét áp lên nền Tím để loại bỏ 100% viền màu tím thừa
        final_img = Image.new("RGBA", (self.width, self.height), (255, 0, 255, 255))
        pixels = img_resized.load()
        final_pixels = final_img.load()
        
        for x in range(self.width):
            for y in range(self.height):
                r_p, g_p, b_p, a_p = pixels[x, y]
                if a_p > 100:  # Giữ lại các pixel thuộc về bong bóng
                    final_pixels[x, y] = (r_p, g_p, b_p, 255)
                    
        self.bubble_photo = ImageTk.PhotoImage(final_img)
        self.bg_label.config(image=self.bubble_photo)

class DoraemonPet:
    def __init__(self, root):
        self.root = root
        self.config = load_config()
        self.chat_history = []
        self.is_jumping = False
        self.is_box_open = False
        self.is_entry_open = False
        self.authenticated = bool(self.config.get("access_token"))
        self.admin_ws = None
        self.admin_ws_thread = None
        self.admin_chat_open = False

        set_autostart(self.config.get("auto_start", False))
            
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        
        self.transparent_color = "#FF00FF"
        self.root.wm_attributes("-transparentcolor", self.transparent_color)
        self.root.configure(bg=self.transparent_color)

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.window_width = 350
        self.update_window_geometry()

        self.chat_box = SmoothVectorSpeechBubble(root, width=340, height=320)
        self.chat_box.pack_forget() 

        self.load_doraemon_image()
        self.img_label = tk.Label(root, image=self.tk_image, bg=self.transparent_color, cursor="hand2")
        self.img_label.pack(side="bottom", pady=(0, 5))
        
        self.img_label.bind("<Button-1>", self.toggle_chat_entry)

        self.chat_entry = tk.Entry(root, font=('Segoe UI', 10), width=30, justify='center', bd=2, relief="groove")
        self.chat_entry.bind("<Return>", self.handle_chat_command)
        self.chat_entry.insert(0, "Nhập câu hỏi rồi bấm Enter...")
        self.chat_entry.bind("<FocusIn>", lambda e: self.chat_entry.delete(0, tk.END) if self.chat_entry.get() == "Nhập câu hỏi rồi bấm Enter..." else None)

        self.context_menu = tk.Menu(self.root, tearoff=0, font=('Segoe UI', 10))
        self.context_menu.add_command(label="👤 Đăng nhập / Đăng ký", command=self.open_auth_window)
        self.context_menu.add_command(label="💬 Chat với Admin", command=self.open_admin_chat)
        self.context_menu.add_command(label="⚙️ Cấu hình Client Token & Khởi động", command=self.open_settings)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="❌ Thoát ứng dụng", command=self.close_app)

        self.img_label.bind("<Button-3>", lambda e: self.context_menu.post(e.x_root, e.y_root))
        
        self.root.after(5000, self.idle_jump_loop)

        # Chào mừng/tiếp tục học khi phiên đăng nhập đã có sẵn.
        if self.authenticated:
            self.root.after(1200, self.refresh_session_welcome)

        # Tự động bắt chuyện: 2–3 lần/ngày, mỗi lần cách nhau khoảng 4–6 giờ.
        self.schedule_proactive_chat()

    def schedule_proactive_chat(self):
        """Lên lịch 2–3 lời bắt chuyện/ngày, cách nhau khoảng 4–6 giờ."""
        try:
            if not self.config.get("auto_chat", False):
                return

            now = time.time()
            day = time.strftime("%Y-%m-%d")
            saved_day = self.config.get("proactive_day", "")
            count = int(self.config.get("proactive_count", 0) or 0)
            last_at = float(self.config.get("proactive_last_at", 0) or 0)

            if saved_day != day:
                count = 0
                last_at = 0
                self.config["proactive_day"] = day
                self.config["proactive_count"] = 0
                self.config["proactive_last_at"] = 0
                save_config(self.config)

            # Đã đủ 3 lượt hôm nay: kiểm tra lại vào đầu ngày kế tiếp.
            if count >= 3:
                next_day = time.localtime(now + 24 * 3600)
                midnight = time.mktime((
                    next_day.tm_year, next_day.tm_mon, next_day.tm_mday,
                    0, 5, 0, next_day.tm_wday, next_day.tm_yday, next_day.tm_isdst
                ))
                self.root.after(max(60_000, int((midnight - now) * 1000)), self.schedule_proactive_chat)
                return

            # Nếu vừa gửi một lời bắt chuyện, đảm bảo tối thiểu 4 giờ.
            min_wait = 4 * 3600
            if last_at > 0:
                elapsed = now - last_at
                remaining = max(0, min_wait - elapsed)
            else:
                remaining = 0

            # Sau khoảng cách tối thiểu, chọn thời điểm ngẫu nhiên trong 4–6 giờ.
            delay_seconds = max(remaining, random.randint(4 * 3600, 6 * 3600))
            self.root.after(int(delay_seconds * 1000), self.proactive_chat_if_needed)
        except Exception as exc:
            print("Proactive schedule error:", exc)

    def proactive_chat_if_needed(self):
        """Gửi một lời hỏi han ngắn, không biến thành một bài học bắt buộc."""
        try:
            if not self.config.get("auto_chat", False):
                return

            token = self.config.get("access_token", "") or self.config.get("client_token", "")
            if not token:
                self.schedule_proactive_chat()
                return

            if self.is_entry_open or self.admin_chat_open:
                # Không chen vào lúc user đang tương tác; thử lại sau 30 phút.
                self.root.after(30 * 60 * 1000, self.proactive_chat_if_needed)
                return

            day = time.strftime("%Y-%m-%d")
            count = int(self.config.get("proactive_count", 0) or 0)
            if self.config.get("proactive_day") != day:
                count = 0

            if count >= 3:
                self.schedule_proactive_chat()
                return

            prompt = (
                "Đây là một lời bắt chuyện ngắn của Doraemon với người học. "
                "KHÔNG dạy bài, KHÔNG hỏi menu học gì, KHÔNG tạo bài tập và KHÔNG đưa đáp án. "
                "Chỉ chọn một câu thân thiện, tự nhiên trong tinh thần như: "
                "\"Chào bạn, mình học cùng nhau nhé!\", "
                "\"Bạn đang làm rất tốt!\", "
                "\"Cố lên, bạn sắp thành công rồi!\", "
                "\"Chúc bạn một ngày tốt lành!\", "
                "\"Hôm nay bạn khỏe không?\". "
                "Có thể thay đổi cách diễn đạt một chút để tránh lặp lại. "
                "Chỉ trả lời 1 câu ngắn."
            )

            self.config["proactive_last_at"] = time.time()
            self.config["proactive_day"] = day
            self.config["proactive_count"] = count + 1
            save_config(self.config)

            threading.Thread(
                target=self.call_proxy_server,
                args=(prompt, None, True),
                daemon=True
            ).start()
        except Exception as exc:
            print("Proactive chat error:", exc)
        finally:
            self.schedule_proactive_chat()

    def update_window_geometry(self):
        """Keep the whole Doraemon stack visible inside the screen.

        The previous layout used a hard-coded total height. On some Windows
        setups Tkinter's requested heights (especially the image label and
        text widget windows) can differ from that estimate, which may leave
        the Doraemon image below the visible toplevel area.
        """
        try:
            # Let pack/place calculate the real requested sizes first.
            self.root.update_idletasks()

            # Doraemon image row: requested image height + its vertical padding.
            image_h = 0
            if hasattr(self, "img_label") and self.img_label.winfo_manager():
                image_h = max(0, int(self.img_label.winfo_reqheight()))

            # Chat box uses an explicit fixed height. Include its outer pack padding.
            chat_h = 0
            if self.is_box_open:
                chat_h = max(240, int(self.chat_box.height)) + 4

            # Entry row is only present while typing.
            entry_h = 0
            if self.is_entry_open:
                entry_h = max(30, int(self.chat_entry.winfo_reqheight())) + 7

            # Keep a small safety margin between Doraemon and the bottom edge.
            base_h = max(125, image_h) + chat_h + entry_h + 6

            # Never allow the top of the window to move off-screen.
            bottom_margin = 12
            max_h = max(240, self.screen_height - bottom_margin)
            base_h = min(base_h, max_h)

            bottom_y = self.screen_height - bottom_margin
            new_y = max(0, bottom_y - base_h)
            home_x = max(0, self.screen_width - self.window_width - 40)
            self.root.geometry(
                f"{self.window_width}x{int(base_h)}+{int(home_x)}+{int(new_y)}"
            )
            self.root.update_idletasks()
        except Exception as exc:
            print("Window geometry error:", exc)

    def load_doraemon_image(self):
        img_path = "Doraemon.png.jpg"
        # Tăng kích thước Doraemon
        target_size = 125 
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGBA")
                w, h = img.size
                pixels = img.load()
                
                # BƯỚC 1: Xóa nền ngay từ ảnh gốc (chưa resize) để viền không bị đứt đoạn
                queue = deque([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])
                visited = set(queue)
                
                while queue:
                    x, y = queue.popleft()
                    r, g, b, a = pixels[x, y]
                    # Bắt các pixel màu đen/tối của nền
                    if r < 55 and g < 55 and b < 55:
                        pixels[x, y] = (0, 0, 0, 0) # Xóa thành trong suốt hoàn toàn
                        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
                                
                # BƯỚC 2: Resize mịn màng bằng LANCZOS
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # BƯỚC 3: Khử viền dơ bằng Alpha Thresholding
                final_img = Image.new("RGBA", (target_size, target_size), (255, 0, 255, 255))
                pixels_resized = img.load()
                final_pixels = final_img.load()
                
                for x in range(target_size):
                    for y in range(target_size):
                        r, g, b, a = pixels_resized[x, y]
                        if a > 100: # Pixel thuộc về Doraemon
                            final_pixels[x, y] = (r, g, b, 255)
                            
                self.tk_image = ImageTk.PhotoImage(final_img)
                return
            except Exception as e:
                print("Lỗi tải ảnh Doraemon:", e)
        
        fallback_img = Image.new('RGBA', (target_size, target_size), (255, 0, 255, 255))
        self.tk_image = ImageTk.PhotoImage(fallback_img)

    def idle_jump_loop(self):
        if not self.is_jumping:
            self.is_jumping = True
            try:
                current_x = self.root.winfo_x()
                current_y = self.root.winfo_y()
                self.root.geometry(f"{self.window_width}x{self.root.winfo_height()}+{current_x}+{current_y - 8}")
                self.root.after(150, lambda: self.root.geometry(f"{self.window_width}x{self.root.winfo_height()}+{current_x}+{current_y}"))
                self.root.after(150, lambda: setattr(self, 'is_jumping', False))
            except:
                self.is_jumping = False
        self.root.after(random.randint(8000, 15000), self.idle_jump_loop)

    def toggle_chat_entry(self, event):
        if self.is_entry_open:
            self.chat_entry.pack_forget()
            self.is_entry_open = False
        else:
            self.chat_entry.pack(side="bottom", pady=(2, 5))
            self.is_entry_open = True
            self.chat_entry.focus_set()
        self.update_window_geometry()

    def handle_chat_command(self, event):
        user_input = self.chat_entry.get().strip()
        
        if self.is_box_open and (not user_input or user_input == "Nhập câu hỏi rồi bấm Enter..."):
            self.close_chat_box()
            return

        if not user_input or user_input == "Nhập câu hỏi rồi bấm Enter...":
            return
        
        self.chat_entry.delete(0, tk.END)
        self.show_response("Doraemon đang suy nghĩ... 💭")
        
        final_prompt = user_input
        if "chỗ này" in user_input.lower() or "đoạn này" in user_input.lower():
            try:
                clipboard_text = self.root.clipboard_get()
                if clipboard_text:
                    final_prompt = f"Dựa vào đoạn văn bản đã copy sau đây:\n---\n{clipboard_text}\n---\nNgười dùng hỏi: {user_input}"
            except:
                pass

        image_base64 = None
        if "màn hình" in user_input.lower():
            try:
                screenshot = ImageGrab.grab()
                buffered = io.BytesIO()
                screenshot.save(buffered, format="JPEG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                print("Lỗi chụp màn hình:", e)

        threading.Thread(target=self.call_proxy_server, args=(final_prompt, image_base64), daemon=True).start()

    def call_proxy_server(self, prompt, image_base64, proactive=False):
        token = self.config.get("access_token", "") or self.config.get("client_token", "")
        if not token:
            self.root.after(0, lambda: self.show_response("Bạn chưa đăng nhập. Bấm chuột phải → Đăng nhập / Đăng ký. 👤"))
            return

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = {
            "prompt": prompt,
            "chat_history": self.chat_history,
            "image_base64": image_base64,
            "use_knowledge_base": True,
            "knowledge_namespace": "__default__"
        }

        try:
            response = requests.post(PROXY_URL, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                data = response.json()
                answer = data.get("reply", data.get("response", "Không có phản hồi từ AI."))
                images = data.get("images") or []
                if isinstance(images, dict):
                    images = [images]
                if not isinstance(images, list):
                    images = []

                
                if proactive:
                    # Không giả lập một tin nhắn user khi Doraemon tự mở lời.
                    self.chat_history.append({"role": "model", "parts": [{"text": answer}]})
                else:
                    self.chat_history.append({"role": "user", "parts": [{"text": prompt}]})
                    self.chat_history.append({"role": "model", "parts": [{"text": answer}]})
                
                content_blocks = data.get("content_blocks") or []
                self.root.after(
                    0,
                    lambda answer=answer, images=images, content_blocks=content_blocks:
                    self.show_response(answer, images, content_blocks)
                )
            else:
                try:
                    detail = response.json().get("detail", response.text)
                    if isinstance(detail, dict):
                        code = detail.get("code", "")
                        msg = detail.get("message", "Lỗi Server")
                        if code == "TOKEN_EXPIRED":
                            expiry = detail.get("expires_at", "")
                            text = f"🔒 Client Token đã hết hạn.\nHạn sử dụng: {expiry}\nVui lòng liên hệ quản trị viên để gia hạn."
                        elif code == "CLIENT_DISABLED":
                            text = "🔒 Client này đã bị khóa. Vui lòng liên hệ quản trị viên."
                        elif code == "INVALID_TOKEN":
                            text = "🔑 Client Token không hợp lệ. Bấm chuột phải → Cấu hình Client Token để kiểm tra lại."
                        else:
                            text = f"Lỗi Server: {msg}"
                    else:
                        text = f"Lỗi Server: {detail}"
                except Exception:
                    text = f"Lỗi Server: {response.text}"
                self.root.after(0, lambda text=text: self.show_response(text))
        except Exception as e:
            self.root.after(0, lambda: self.show_response(f"Lỗi kết nối: {str(e)[:50]}"))

    def show_response(self, text, images=None, content_blocks=None):
        images = images or []

        if not self.is_box_open:
            self.chat_box.pack(side="top", fill="x", padx=10, pady=(2, 2), before=self.img_label)
            self.is_box_open = True
            # Recalculate after the chatbox is actually packed so Doraemon
            # stays visible instead of being pushed outside the toplevel.
            self.root.update_idletasks()
            self.update_window_geometry()

        # Xóa nội dung/ảnh cũ.
        for widget in getattr(self.chat_box, "_image_widgets", []):
            try:
                widget.destroy()
            except Exception:
                pass
        self.chat_box._image_widgets = []
        self.chat_box.image_refs = []

        t_widget = self.chat_box.text_widget
        t_widget.config(state="normal")
        t_widget.delete("1.0", tk.END)

        # Server V3.1 trả content_blocks theo thứ tự:
        # text -> image -> text -> image ...
        # Fallback để vẫn tương thích server cũ.
        blocks = content_blocks or []
        if not blocks:
            blocks = [{"type": "text", "text": text or ""}]
            for item in images[:3]:
                if isinstance(item, str):
                    url, key = item, ""
                elif isinstance(item, dict):
                    url = item.get("url") or item.get("image_url")
                    key = item.get("key", "")
                else:
                    url, key = None, ""
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    blocks.append({"type": "image", "key": key, "url": url})

        image_jobs = []

        # Render text và tạo placeholder window cho ảnh NGAY LẬP TỨC.
        # Khi ảnh tải xong chỉ cần thay image của label, không phải chèn lại
        # Text widget => vị trí xen kẽ luôn được giữ chính xác.
        for block_index, block in enumerate(blocks):
            if block.get("type") == "image":
                url = block.get("url") or block.get("image_url")
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    continue

                label = tk.Label(
                    t_widget,
                    text="  Đang tải ảnh...  ",
                    bg="#FFFFFF",
                    fg="#777777",
                    bd=0,
                    padx=2,
                    pady=4
                )
                self.chat_box._image_widgets.append(label)
                self.chat_box.bind_scroll_widget(label)
                t_widget.insert(tk.END, "\n")
                t_widget.window_create(tk.END, window=label)
                t_widget.insert(tk.END, "\n")
                image_jobs.append((block_index, block, label))
                continue

            part = block.get("text", "")
            if not part:
                continue

            url_pattern = re.compile(r'(https?://[^\s]+)')
            parts = url_pattern.split(part)
            for p in parts:
                if not p:
                    continue
                if url_pattern.fullmatch(p):
                    tag_name = f"link_{random.randint(1000,9999)}"
                    t_widget.insert(tk.END, p, ("hyperlink", tag_name))
                    t_widget.tag_config("hyperlink", foreground="blue", underline=True)
                    t_widget.tag_bind(
                        tag_name, "<Button-1>",
                        lambda e, url=p: webbrowser.open(url)
                    )
                    t_widget.tag_bind(
                        tag_name, "<Enter>",
                        lambda e: t_widget.config(cursor="hand2")
                    )
                    t_widget.tag_bind(
                        tag_name, "<Leave>",
                        lambda e: t_widget.config(cursor="")
                    )
                else:
                    t_widget.insert(tk.END, p)

        t_widget.config(state="disabled")

        if image_jobs:
            threading.Thread(
                target=self._load_response_images_inline,
                args=(image_jobs,),
                daemon=True
            ).start()

        self._resize_bubble_to_content()

    def _resize_bubble_to_content(self):
        # Chatbox luôn giữ nguyên chiều cao. Nội dung dài sẽ được cuộn
        # bằng scrollbar dọc của Text widget thay vì làm bubble phình ra.
        try:
            fixed_height = 320
            if self.chat_box.height != fixed_height:
                self.chat_box.set_height(fixed_height)
            self.update_window_geometry()
        except Exception as e:
            print("Bubble resize error:", e)

    def _load_response_images_inline(self, image_jobs):
        """
        Tải ảnh độc lập bằng ThreadPoolExecutor.
        Một ảnh lỗi/kẹt không được chặn các ảnh còn lại.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Do not silently drop images after the first six. The old slice
        # left later placeholders stuck at "Đang tải ảnh..." forever.
        jobs = list(image_jobs)
        if not jobs:
            return

        def load_one(job):
            block_index, block, label = job
            url = block.get("url") or block.get("image_url")

            try:
                # Mỗi worker có Session riêng để tránh dùng chung connection
                # giữa các request song song.
                session = requests.Session()
                session.headers.update({
                    "User-Agent": "DoraemonStudyAssistant/3.1"
                })

                response = None
                last_error = None

                # Chỉ retry lỗi mạng tạm thời. 4xx không retry vì URL/quyền
                # truy cập sai thì retry không giúp ích.
                for attempt in range(2):
                    try:
                        response = session.get(
                            url,
                            timeout=(4, 8),
                            allow_redirects=True
                        )

                        status = response.status_code
                        if 200 <= status < 300:
                            break

                        if 400 <= status < 500:
                            response.raise_for_status()

                        response.raise_for_status()

                    except (requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError) as exc:
                        last_error = exc
                        response = None
                        if attempt == 0:
                            time.sleep(0.3)
                            continue
                        raise

                if response is None:
                    raise RuntimeError(
                        str(last_error) if last_error else "Không nhận được response"
                    )

                content_type = (response.headers.get("Content-Type") or "").lower()
                content = response.content

                # Backblaze/object storage phải trả bytes ảnh.
                if "image" not in content_type and not content.startswith(
                    (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF")
                ):
                    raise RuntimeError(
                        f"Không phải file ảnh: HTTP {response.status_code}, "
                        f"Content-Type={content_type}"
                    )

                img = Image.open(io.BytesIO(content))
                img.load()
                img = img.convert("RGB")

                # Ảnh nhỏ giữ nguyên, không upscale.
                # Chỉ ảnh rộng hơn chatbox mới downscale.
                max_w = max(160, self.chat_box.width - 38)
                if img.width > max_w:
                    ratio = max_w / float(img.width)
                    img = img.resize(
                        (
                            max(1, int(img.width * ratio)),
                            max(1, int(img.height * ratio))
                        ),
                        Image.Resampling.LANCZOS
                    )

                photo = ImageTk.PhotoImage(img)

                return {
                    "ok": True,
                    "label": label,
                    "photo": photo,
                    "url": url,
                }

            except Exception as exc:
                return {
                    "ok": False,
                    "label": label,
                    "error": str(exc),
                    "url": url,
                }

        def finish_one(result):
            label = result["label"]
            try:
                if result["ok"]:
                    photo = result["photo"]
                    label.config(image=photo, text="", fg="#000000")
                    # Giữ reference để Tkinter không garbage-collect ảnh.
                    label.image = photo
                else:
                    label.config(
                        image="",
                        text="Không tải được ảnh",
                        fg="#b00020"
                    )
                    label.image = None
                    print(
                        f"[IMAGE LOAD ERROR] {result['url']}: "
                        f"{result['error']}"
                    )
            except Exception as exc:
                print("[IMAGE RENDER ERROR]", exc)

        # Tối đa 6 ảnh đồng thời. Kết quả ảnh nào xong trước thì render trước.
        executor = ThreadPoolExecutor(max_workers=min(6, len(jobs)))

        futures = [executor.submit(load_one, job) for job in jobs]

        def collect_results():
            try:
                # Không gọi future.result() trên UI thread.
                # Poll từng future để ảnh hoàn thành độc lập.
                pending = []
                for future in futures:
                    if future.done():
                        try:
                            result = future.result()
                            finish_one(result)
                        except Exception as exc:
                            print("[IMAGE WORKER ERROR]", exc)
                    else:
                        pending.append(future)

                if pending:
                    self.root.after(80, collect_results)
                else:
                    executor.shutdown(wait=False)
            except Exception as exc:
                print("[IMAGE COLLECT ERROR]", exc)
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

        self.root.after(0, collect_results)

    def close_chat_box(self):
        if self.is_box_open:
            self.chat_box.pack_forget()
            self.is_box_open = False
            self.update_window_geometry()

    def open_auth_window(self):
        top = tk.Toplevel(self.root)
        top.title("Doraemon - Đăng nhập / Đăng ký")
        top.geometry("390x330")
        top.resizable(False, False)
        top.wm_attributes("-topmost", True)

        tk.Label(top, text="Tài khoản Doraemon", font=("Segoe UI", 14, "bold")).pack(pady=(18, 12))
        form = tk.Frame(top); form.pack(fill="x", padx=25)

        tk.Label(form, text="Số điện thoại").pack(anchor="w")
        phone = tk.Entry(form, font=("Segoe UI", 10)); phone.pack(fill="x", pady=(2, 8))
        phone.insert(0, self.config.get("phone", ""))

        tk.Label(form, text="Nickname (chỉ dùng khi đăng ký)").pack(anchor="w")
        nickname = tk.Entry(form, font=("Segoe UI", 10)); nickname.pack(fill="x", pady=(2, 8))
        nickname.insert(0, self.config.get("nickname", ""))

        tk.Label(form, text="Mật khẩu").pack(anchor="w")
        password = tk.Entry(form, show="*", font=("Segoe UI", 10)); password.pack(fill="x", pady=(2, 12))

        status = tk.Label(top, text="", fg="#555555", wraplength=340)
        status.pack(pady=5)

        def do_register():
            payload = {"phone": phone.get().strip(), "nickname": nickname.get().strip(), "password": password.get()}
            try:
                r = requests.post(REGISTER_URL, json=payload, timeout=20)
                try:
                    data = r.json()
                except ValueError:
                    body = (r.text or "").strip()
                    preview = body[:300] if body else "(response body rỗng)"
                    data = {
                        "detail": f"Server trả response không phải JSON (HTTP {r.status_code}): {preview}"
                    }

                if r.status_code == 200:
                    status.config(text="Đăng ký thành công. Chờ Admin kích hoạt.", fg="green")
                else:
                    status.config(text=str(data.get("detail", data)), fg="red")

            except requests.RequestException as e:
                status.config(text=f"Lỗi kết nối server: {e}", fg="red")
            except Exception as e:
                status.config(text=f"Lỗi: {e}", fg="red")


        def do_login():
            payload = {"phone": phone.get().strip(), "password": password.get()}
            try:
                r = requests.post(LOGIN_URL, json=payload, timeout=20)
                try:
                    data = r.json()
                except ValueError:
                    body = (r.text or "").strip()
                    preview = body[:300] if body else "(response body rỗng)"
                    data = {
                        "detail": f"Server trả response không phải JSON (HTTP {r.status_code}): {preview}"
                    }

                if r.status_code != 200:
                    status.config(text=str(data.get("detail", data)), fg="red")
                    return

                if not isinstance(data, dict) or not data.get("access_token"):
                    status.config(
                        text=f"Server đăng nhập thiếu access_token: {str(data)[:300]}",
                        fg="red"
                    )
                    return

                self.config["access_token"] = data["access_token"]
                user_data = data.get("user") or {}
                self.config["phone"] = user_data.get("phone", phone.get().strip())
                self.config["nickname"] = user_data.get("nickname", nickname.get().strip())
                save_config(self.config)
                self.authenticated = True
                status.config(text="Đăng nhập thành công.", fg="green")
                top.after(700, lambda: (top.destroy(), self.refresh_session_welcome()))

            except requests.RequestException as e:
                status.config(text=f"Lỗi kết nối server: {e}", fg="red")
            except Exception as e:
                status.config(text=f"Lỗi: {e}", fg="red")

        buttons = tk.Frame(top); buttons.pack(pady=8)
        ttk.Button(buttons, text="Đăng nhập", command=do_login).pack(side="left", padx=5)
        ttk.Button(buttons, text="Đăng ký", command=do_register).pack(side="left", padx=5)

        if self.config.get("access_token"):
            tk.Button(top, text="Đăng xuất", command=lambda: self.logout(top)).pack(pady=3)

    def refresh_session_welcome(self):
        token = self.config.get("access_token", "") or self.config.get("client_token", "")
        if not token:
            return

        def worker():
            try:
                r = requests.get(
                    WELCOME_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20
                )
                if r.status_code == 200:
                    data = r.json()
                    msg = data.get("message", "")
                    if msg:
                        self.chat_history = []
                        self.root.after(0, lambda msg=msg: self.show_response(msg))
                elif r.status_code in (401, 403):
                    print("Session welcome skipped:", r.text[:200])
            except Exception as exc:
                print("Session welcome error:", exc)

        threading.Thread(target=worker, daemon=True).start()

    def reset_learning_history(self, parent=None):
        if not messagebox.askyesno(
            "Xóa lịch sử học",
            "Bạn có chắc muốn xóa toàn bộ lịch sử học không?\n\n"
            "Tiến độ giáo trình sẽ được reset về từ đầu như một user mới. "
            "Tài khoản và gói học sẽ không bị xóa.",
            parent=parent or self.root
        ):
            return

        token = self.config.get("access_token", "") or self.config.get("client_token", "")
        if not token:
            messagebox.showinfo("Doraemon", "Bạn cần đăng nhập trước.", parent=parent or self.root)
            return

        def worker():
            try:
                r = requests.post(
                    RESET_LEARNING_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20
                )
                if r.status_code == 200:
                    self.chat_history = []
                    data = r.json()
                    self.root.after(0, lambda: (
                        messagebox.showinfo(
                            "Doraemon",
                            "Đã xóa lịch sử học. Giáo trình đã được reset về từ đầu. 🤖",
                            parent=parent or self.root
                        ),
                        self.refresh_session_welcome()
                    ))
                else:
                    try:
                        detail = r.json().get("detail", r.text)
                    except Exception:
                        detail = r.text
                    self.root.after(
                        0,
                        lambda detail=detail: messagebox.showerror(
                            "Không thể xóa lịch sử học",
                            str(detail),
                            parent=parent or self.root
                        )
                    )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda exc=exc: messagebox.showerror(
                        "Lỗi kết nối",
                        f"Không thể kết nối server: {exc}",
                        parent=parent or self.root
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def logout(self, top=None):
        self.config["access_token"] = ""
        save_config(self.config)
        self.authenticated = False
        self.close_admin_socket()
        if top:
            top.destroy()
        self.show_response("Đã đăng xuất.")

    def close_admin_socket(self):
        try:
            ws = getattr(self, "admin_ws", None)
            if ws:
                try:
                    ws.keep_running = False
                except Exception:
                    pass
                try:
                    ws.close()
                except Exception:
                    pass
        finally:
            self.admin_ws = None

    def open_admin_chat(self):
        token = self.config.get("access_token", "")
        if not token:
            messagebox.showinfo("Doraemon", "Bạn cần đăng nhập trước.", parent=self.root)
            self.open_auth_window()
            return

        win = tk.Toplevel(self.root)
        win.title("💬 Chat với Admin")
        win.geometry("420x500")
        win.wm_attributes("-topmost", True)
        self.admin_chat_open = True

        text = tk.Text(win, wrap="word", state="disabled", font=("Segoe UI", 10))
        text.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        seen_ids = set()
        poll_stop = threading.Event()

        bottom = tk.Frame(win); bottom.pack(fill="x", padx=10, pady=10)
        entry = tk.Entry(bottom, font=("Segoe UI", 10))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        status_label = tk.Label(win, text="Đang kết nối Chat Admin...", fg="#777")
        status_label.pack(anchor="w", padx=10)

        def append(msg):
            mid = msg.get("id")
            if mid is not None:
                try:
                    mid = int(mid)
                except Exception:
                    pass
                if mid in seen_ids:
                    return
                seen_ids.add(mid)
            def do_append():
                text.config(state="normal")
                sender = "Bạn" if msg.get("sender") == "user" else "Admin"
                text.insert(tk.END, f"{sender}: {msg.get('message','')}\n\n")
                text.see(tk.END)
                text.config(state="disabled")
            self.root.after(0, do_append)

        try:
            r = requests.get(ADMIN_HISTORY_URL,
                             headers={"Authorization": f"Bearer {token}"}, timeout=20)
            if r.status_code == 200:
                for msg in r.json().get("messages", []):
                    append(msg)
            elif r.status_code in (401, 403):
                self.logout()
                messagebox.showerror("Doraemon", "Tài khoản chưa được kích hoạt hoặc phiên đăng nhập đã hết hạn.", parent=win)
                win.destroy()
                return
        except Exception as e:
            append({"sender":"admin", "message":f"Không tải được lịch sử: {e}"})

        def poll_history():
            last_max = max(seen_ids) if seen_ids else 0
            while self.admin_chat_open and not poll_stop.is_set():
                try:
                    r = requests.get(
                        ADMIN_HISTORY_URL,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10
                    )
                    if r.status_code == 200:
                        for msg in r.json().get("messages", []):
                            mid = msg.get("id")
                            try:
                                mid_num = int(mid)
                            except Exception:
                                mid_num = 0
                            if mid_num > last_max:
                                last_max = max(last_max, mid_num)
                                append(msg)
                    elif r.status_code in (401, 403):
                        self.root.after(0, lambda: status_label.config(
                            text="● Phiên đăng nhập hết hạn", fg="red"))
                        break
                except Exception as e:
                    print("Admin chat polling error:", e)
                poll_stop.wait(1.5)

        self.admin_poll_thread = threading.Thread(target=poll_history, daemon=True)
        self.admin_poll_thread.start()

        def on_open(ws):
            self.root.after(0, lambda: status_label.config(text="● Chat Admin đã kết nối realtime", fg="green"))

        def on_message(ws, raw):
            try:
                data = json.loads(raw)
                if data.get("type") == "connected":
                    self.root.after(0, lambda: status_label.config(text="● Chat Admin đã kết nối realtime", fg="green"))
                elif data.get("type") == "message":
                    msg = data.get("data", {})
                    append(msg)
                    self.root.after(0, lambda: status_label.config(
                        text="● Realtime hoạt động", fg="green"))
                elif data.get("type") == "error":
                    self.root.after(0, lambda: status_label.config(text="Lỗi: " + str(data.get("message", "")), fg="red"))
            except Exception:
                pass

        def on_error(ws, error):
            print("Admin WS error:", error)
            self.root.after(0, lambda: status_label.config(
                text="● Realtime gián đoạn, chat vẫn hoạt động", fg="#b26a00"))

        def on_close(ws, code, msg):
            print("Admin WS closed:", code, msg)
            self.root.after(0, lambda: status_label.config(
                text="● Realtime đang kết nối lại, chat vẫn hoạt động", fg="#b26a00"))

        def run_ws():
            # WebSocket chỉ là kênh realtime phụ. HTTP polling phía trên vẫn là
            # kênh đảm bảo để nhận tin nhắn ngay cả khi Render/người dùng bị rớt WS.
            websocket.enableTrace(False)
            backoff = 1.0
            while self.admin_chat_open and self.config.get("access_token") == token:
                try:
                    self.admin_ws = websocket.WebSocketApp(
                        WS_USER_URL + "?token=" + urllib.parse.quote(token, safe=""),
                        on_open=on_open,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close
                    )
                    self.admin_ws.run_forever(
                        ping_interval=15,
                        ping_timeout=8,
                        http_proxy_host=None,
                        http_proxy_port=None
                    )
                    backoff = 1.0
                except Exception as e:
                    print("Admin WS exception:", e)
                    self.root.after(0, lambda e=e: status_label.config(
                        text="● Realtime gián đoạn, chat vẫn hoạt động", fg="#b26a00"))
                if self.admin_chat_open:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.8, 15.0)

        self.admin_ws_thread = threading.Thread(target=run_ws, daemon=True)
        self.admin_ws_thread.start()

        def send():
            msg = entry.get().strip()
            if not msg:
                return

            payload = {"message": msg}
            ws = self.admin_ws
            ws_ok = bool(ws and getattr(ws, "sock", None) and ws.sock and ws.sock.connected)

            # Prefer WebSocket for realtime delivery, but NEVER block the user
            # from chatting just because the persistent WS is temporarily down.
            if ws_ok:
                try:
                    ws.send(json.dumps(payload, ensure_ascii=False))
                    entry.delete(0, tk.END)
                    return
                except Exception as e:
                    print("Admin WS send failed, fallback HTTP:", e)

            try:
                r = requests.post(
                    ADMIN_SEND_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10
                )
                if r.status_code == 200:
                    entry.delete(0, tk.END)
                    data = r.json() if r.content else {}
                    row = data.get("message") or {}
                    if row:
                        append(row)
                    self.root.after(0, lambda: status_label.config(
                        text="● Đã gửi qua HTTP; realtime sẽ tự kết nối lại", fg="#b26a00"))
                    return
                if r.status_code in (401, 403):
                    self.root.after(0, lambda: status_label.config(
                        text="● Phiên đăng nhập hết hạn", fg="red"))
                    return
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                messagebox.showerror("Chat Admin", f"Không gửi được tin nhắn: {e}", parent=win)

        ttk.Button(bottom, text="Gửi", command=send).pack(side="right")
        entry.bind("<Return>", lambda e: send())

        def close():
            self.admin_chat_open = False
            poll_stop.set()
            self.close_admin_socket()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", close)

    def open_settings(self):
        top = tk.Toplevel(self.root)
        top.title("Cấu hình Doraemon")
        top.geometry("400x365")
        top.resizable(False, False)
        top.wm_attributes("-topmost", True)
        top.geometry(f"+{self.screen_width//2 - 200}+{self.screen_height//2 - 180}")

        tk.Label(top, text="Nhập Client Token được cấp:", font=('Segoe UI', 9, 'bold')).pack(anchor="w", padx=20, pady=(15, 5))

        entry = tk.Entry(top, width=44, font=('Segoe UI', 10))
        entry.pack(padx=20, pady=5)
        entry.insert(0, self.config.get("client_token", ""))
        entry.focus()

        auto_start_var = tk.BooleanVar(value=self.config.get("auto_start", False))
        chk_autostart = tk.Checkbutton(
            top, text="Khởi động cùng Windows",
            variable=auto_start_var, font=('Segoe UI', 9)
        )
        chk_autostart.pack(anchor="w", padx=20, pady=(5, 2))

        auto_chat_var = tk.BooleanVar(value=self.config.get("auto_chat", False))
        chk_auto_chat = tk.Checkbutton(
            top,
            text="Tự động bắt chuyện",
            variable=auto_chat_var,
            font=('Segoe UI', 9),
            anchor="w",
            justify="left"
        )
        chk_auto_chat.pack(anchor="w", padx=20, pady=(2, 0))

        tk.Label(
            top,
            text="Nếu bật: Doraemon sẽ chủ động hỏi han khoảng 2–3 lần/ngày,\n"
                 "mỗi lần cách nhau khoảng 4–6 giờ.",
            font=('Segoe UI', 8),
            fg="#555555",
            justify="left"
        ).pack(anchor="w", padx=40, pady=(0, 8))

        tk.Frame(top, height=1, bg="#dddddd").pack(fill="x", padx=20, pady=(2, 10))

        tk.Label(
            top,
            text="⚠️ Quản lý lịch sử học",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor="w", padx=20, pady=(0, 3))

        tk.Label(
            top,
            text="Xóa toàn bộ tiến độ sẽ đưa giáo trình về từ đầu\n"
                 "như một user mới. Tài khoản không bị xóa.",
            font=('Segoe UI', 8),
            fg="#555555",
            justify="left"
        ).pack(anchor="w", padx=40, pady=(0, 5))

        ttk.Button(
            top,
            text="🗑️ Xóa lịch sử học",
            command=lambda: self.reset_learning_history(top)
        ).pack(pady=(2, 10))

        def save_and_close():
            old_auto = bool(self.config.get("auto_chat", False))
            self.config["client_token"] = entry.get().strip()
            self.config["auto_start"] = auto_start_var.get()
            self.config["auto_chat"] = auto_chat_var.get()

            if not self.config["auto_chat"]:
                self.config["proactive_last_at"] = 0
                self.config["proactive_day"] = ""
                self.config["proactive_count"] = 0

            save_config(self.config)
            set_autostart(self.config["auto_start"])

            if self.config["auto_chat"] and not old_auto:
                self.schedule_proactive_chat()

            messagebox.showinfo("Thành công", "Đã lưu cấu hình thành công!", parent=top)
            top.destroy()

        ttk.Button(top, text="Lưu Cấu Hình", command=save_and_close).pack(pady=5)

    def close_app(self):
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = DoraemonPet(root)
    root.mainloop()