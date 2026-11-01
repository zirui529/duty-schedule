import streamlit as st
import fitz
import pandas as pd
import json
import os
import re
import random
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ========== 确保 set_page_config 第一个执行 ==========
st.set_page_config(page_title="值班排班系统", layout="wide")

# ========== 基础配置 ==========
DATA_FILE = "member_data.json"
ADMIN_FILE = "admin_info.json"

# 火山方舟配置：从 secrets 读取（若无则给出明确提示，但阻止后续 AI 调用）
VOLC_API_KEY = None
VOLC_ENDPOINT = None
MODEL_ID = None
try:
    VOLC_API_KEY = st.secrets["ARK_API_KEY"]
    VOLC_ENDPOINT = st.secrets["ARK_ENDPOINT"]
    MODEL_ID = st.secrets["ARK_MODEL_ENDPOINT_ID"]
except KeyError:
    # 不调用 st.stop()，只显示警告，保证界面仍能正常加载
    st.warning("⚠️ 缺少火山方舟密钥，请在 Streamlit Secrets 中配置 ARK_API_KEY、ARK_ENDPOINT、ARK_MODEL_ENDPOINT_ID。AI 识别功能将不可用。")

API_TIMEOUT = 30
MAX_WORKERS = 4
SHIFT_LIST = [
    {"shift_idx": 0, "shift_name": "9:00-13:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx": 1, "shift_name": "13:00-17:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx": 2, "shift_name": "17:00-20:00", "need_stu": 2, "need_wei": 1}
]
WEEK_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TEMPLATE_OPTIONS = ["默认样式", "武汉大学学生会水印模板"]
SHIFT_PERIOD = {0: "上午", 1: "下午", 2: "晚上"}

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT_NAME = 'STSong-Light'

def safe_str(val):
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()

# ========== 原子化 JSON 读写 ==========
def _write_json_atomically(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def init_admin():
    if not os.path.exists(ADMIN_FILE):
        _write_json_atomically(ADMIN_FILE, {"admins": [{"username": "admin", "password": "123456", "is_super": True}]})
    else:
        try:
            with open(ADMIN_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)
            if "admins" not in old:
                old_pwd = old.get("admin_pwd", "123456")
                _write_json_atomically(ADMIN_FILE, {"admins": [{"username": "admin", "password": old_pwd, "is_super": True}]})
        except (json.JSONDecodeError, OSError):
            _write_json_atomically(ADMIN_FILE, {"admins": [{"username": "admin", "password": "123456", "is_super": True}]})

def load_admin():
    init_admin()
    try:
        with open(ADMIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        st.error("管理员文件损坏，已重置为默认账号。")
        init_admin()
        with open(ADMIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

def save_admin(data):
    _write_json_atomically(ADMIN_FILE, data)

def verify_admin(username, password):
    data = load_admin()
    for a in data["admins"]:
        if a["username"] == username and a["password"] == password:
            return a
    return None

def is_super_admin(username):
    data = load_admin()
    for a in data["admins"]:
        if a["username"] == username and a.get("is_super", False):
            return True
    return False

def load_members():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for m in data:
                if "class_conflict" not in m:
                    m["class_conflict"] = m.pop("limit_time", "")
                if "personal_need" not in m:
                    m["personal_need"] = ""
                if "role" not in m or not m["role"]:
                    m["role"] = "学员"
            return data
        except (json.JSONDecodeError, OSError):
            st.warning("成员数据文件损坏，已清空。")
            return []
    return []

def save_members(data):
    _write_json_atomically(DATA_FILE, data)

# ========== PDF 文本提取 ==========
def get_pdf_full_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    txt = "".join(page.get_text() for page in doc)
    doc.close()
    return txt

def extract_name_from_text(text):
    for line in text.splitlines():
        m = re.fullmatch(r"([\u4e00-\u9fa5]{2,4})\s*课表", line.strip())
        if m:
            return m.group(1).strip()
    m = re.search(r"([\u4e00-\u9fa5]{2,4})课表", text)
    return m.group(1).strip() if m else ""

def parse_filename_phone_and_role(filename):
    name_part = filename.split("-课表")[0]
    parts = name_part.split("&")
    role = "学员"
    phone = ""
    if len(parts) >= 3:
        role_raw = parts[0]
        if "职委" in role_raw or "干部" in role_raw or "部长" in role_raw or "主席" in role_raw:
            role = "职委"
        else:
            role = "学员"
        phone_candidate = parts[2].strip()
        try:
            if re.fullmatch(r"1[3-9]\d{9}", phone_candidate):
                phone = phone_candidate
        except re.error:
            phone = ""
    return role, phone

# ========== AI 调用（带防幻觉） ==========
def _call_course_ai(prompt_text):
    if not VOLC_API_KEY:
        raise RuntimeError("未配置 API Key")
    headers = {"Authorization": f"Bearer {VOLC_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": """
你是课表提取工具，**只输出JSON数组，绝对不能输出任何说明、注释、markdown、```符号**。

【节次标准时间表，严格参照】
1:08:00-08:45｜2:08:50-09:35｜3:09:50-10:35｜4:10:40-11:25｜5:11:30-12:15
6:14:05-14:50｜7:14:55-15:40｜8:15:45-16:30｜9:16:40-17:25
10:17:30-18:15｜11:18:30-19:15｜12:19:20-20:05｜13:20:10-20:55

【极高优先级强制约束】
1. **只提取课表文字中明确出现的课程**。没有在课表文本中看到的星期+节次，一律不输出。
2. **表格中的空白格、没有课程名称的格子、没有地点/教师的格子，全部视为没课，坚决不输出。**
3. **禁止脑补、禁止推测、禁止根据同一行/同一列的其他课程补全空白格。**
4. 星期只能是：周一、周二、周三、周四、周五、周六、周日；节次只能是1-13的整数，其他数字一律丢弃。
5. 遍历全部文本，不要忽略页面后半段、周六、周日课程；如果某星期完全没有课，该星期不要出现在数组中。
6. 如果整个课表没有任何课程，输出[]。
7. 输出格式固定：[{"week":"周一","sections":[1,3,10]},{"week":"周六","sections":[11,12]}]，不要添加其他字段。
8. 不确定某一节是否有课时，宁可忽略，不要输出。

【示例1】
输入：
周一：
1-2节 高等数学
3-4节 
5-6节 大学英语
输出：
[{"week":"周一","sections":[1,2,5,6]}]

【示例2】
输入：
周六：
1-2节 
3-4节 
5-6节 
输出：
[]
"""},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0
    }
    resp = requests.post(VOLC_ENDPOINT, headers=headers, json=payload, timeout=API_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:120]}")
    res = resp.json()
    out = res["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\\[.*\\\]", out, re.DOTALL)
    if not match:
        raise ValueError("模型没有返回合法JSON数组")
    return json.loads(match.group(0))

def _call_personal_ai(prompt_text):
    if not VOLC_API_KEY:
        return []
    headers = {"Authorization": f"Bearer {VOLC_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": """
只输出JSON字符串数组，内容是【周X上午/下午/晚上】，只能追加，不能删除课表算出的冲突。
无额外限制返回空数组[]。示例：["周日上午","周二晚上"]
禁止markdown、禁止解释文字。
"""},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0
    }
    resp = requests.post(VOLC_ENDPOINT, headers=headers, json=payload, timeout=API_TIMEOUT)
    if resp.status_code != 200:
        return []
    res =
