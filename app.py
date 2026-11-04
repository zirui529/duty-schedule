import streamlit as st
import fitz
import pandas as pd
import json
import os
import re
import random
import base64
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import dashscope
from dashscope import Generation
from dashscope import MultiModalConversation
import numpy as np
import pdfplumber

# ========== 基础配置 ==========
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PWD = "123456"
DATA_FILE = "member_data.json"
ADMIN_FILE = "admin_info.json"
# -------- 分工 --------
# 课表识别主路线：qwen-vl-max 视觉模型直接看课表图片（最准，绕过文本坐标解析）
# 课表识别备选：规整表格文本 + qwen-plus 文本模型（有清晰表格线时使用，快且省）
# doubao2.1‑seed‑pro 火山chat：个性化需求解析
COURSE_TEXT_MODEL = "qwen-plus"
COURSE_VL_MODEL = "qwen-vl-max-latest"
try:
    ARK_API_KEY = st.secrets["ark"]["api_key"]
    ARK_ENDPOINT = st.secrets["ark"]["endpoint"]
    ARK_MODEL = st.secrets["ark"]["model_name"]
    DS_API_KEY = st.secrets["dashscope"]["api_key"]
    dashscope.api_key = DS_API_KEY
except KeyError:
    st.error("secrets必须配置[ark]、[dashscope]两组密钥！")
    st.stop()
API_TIMEOUT = 30
MAX_RETRY = 2
MAX_WORKERS = 4
SHIFT_LIST = [
    {"shift_idx":0,"shift_name": "9:00-13:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx":1,"shift_name": "13:00-17:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx":2,"shift_name": "17:00-20:00", "need_stu": 2, "need_wei": 1}
]
WEEK_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TEMPLATE_OPTIONS = ["默认样式", "武汉大学学生会水印模板"]
SHIFT_PERIOD = {0: "上午", 1: "下午", 2: "晚上"}
ALL_SLOTS = {"周一上午","周一下午","周一晚上","周二上午","周二下午","周二晚上","周三上午","周三下午","周三晚上","周四上午","周四下午","周四晚上","周五上午","周五下午","周五晚上","周六上午","周六下午","周六晚上","周日上午","周日下午","周日晚上"}
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT_NAME = 'STSong-Light'

def safe_str(val):
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()

# ========== 识别结果失败判定 & 规范化（管理员上传 / 成员自助上传 两条路径共用） ==========
FAIL_PREFIXES = ("课表识别失败", "识别为空", "课表文本提取失败")

def is_recognize_failed(text):
    t = safe_str(text)
    return (not t) or t.startswith(FAIL_PREFIXES)

def normalize_conflict_text(text):
    """把「周二下午,周五晚上」等各种分隔符写法统一成「周二下午、周五晚上」，只保留合法时段。"""
    tokens = re.split(r"[、,，;；/\s]+", safe_str(text))
    valid = [t.strip() for t in tokens if t.strip() in ALL_SLOTS]
    return "、".join(sorted(set(valid)))

# ========== 管理员文件读写 ==========
def init_admin():
    if not os.path.exists(ADMIN_FILE):
        data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
        json.dump(data, open(ADMIN_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    else:
        try:
            old = json.load(open(ADMIN_FILE, "r", encoding="utf-8"))
            if "admins" not in old:
                old_pwd = old.get("admin_pwd", "123456")
                new_data = {"admins": [{"username": "admin", "password": old_pwd, "is_super": True}]}
                json.dump(new_data, open(ADMIN_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except:
            data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
            json.dump(data, open(ADMIN_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def reset_default_admin():
    data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
    json.dump(data, open(ADMIN_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def load_admin():
    init_admin()
    return json.load(open(ADMIN_FILE, "r", encoding="utf-8"))

def save_admin(data):
    json.dump(data, open(ADMIN_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

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
        data = json.load(open(DATA_FILE, "r", encoding="utf-8"))
        for m in data:
            if "class_conflict" not in m:
                m["class_conflict"] = m.pop("limit_time", "")
            if "personal_need" not in m:
                m["personal_need"] = ""
            if "role" not in m or not m["role"]:
                m["role"] = "学员"
        return data
    return []

def save_members(data):
    json.dump(data, open(DATA_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ========== PDF读取【三级降级：①表格线识别 ②表头锚点网格 ③纯文本】 ==========
def _cell_text(v):
    t = re.sub(r"[|\n\r]", " ", safe_str(v))
    # 先去掉所有空白（源PDF里字段名常被换行拆开，如"教学班组 成"）
    t = re.sub(r"\s+", "", t)
    # 按"(X-Y节)"课程条目切分：一格里可能有多门课，逐条保留课名+节次+周次+场地+教师
    ms = list(re.finditer(r"[^/]{1,40}\(\d+(?:-\d+)?节\)", t))
    if not ms:
        return t[:300]
    entries = []
    for i, m in enumerate(ms):
        seg = t[m.start(): ms[i+1].start() if i + 1 < len(ms) else len(t)]
        seg = re.split(r"/教学班", seg)[0]  # 每条记录在教学班号等元数据处截断
        if seg:
            entries.append(seg)
    return "；".join(entries)[:400]

def _norm_header(c):
    return re.sub(r"星期", "周", _cell_text(c))

def _build_markdown_table(rows, header):
    n_col = len(header)
    md = ["| " + " | ".join(header) + " |",
          "| " + " | ".join(["----"] * n_col) + " |"]
    for r in rows:
        cells = [_cell_text(c) for c in r]
        cells += [""] * (n_col - len(cells))
        md.append("| " + " | ".join(cells[:n_col]) + " |")
    return "\n".join(md)

def _grid_extract_text(pdf_bytes, row_count=13, col_count=7):
    """改进版坐标网格：用表头'周一..周日'的横坐标锚定列边界，
    左侧节次标签栏落在边界外被丢弃，避免污染周一列。"""
    col_names = WEEK_NAMES
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        grid = [["" for _ in range(col_count)] for _ in range(row_count)]
        for page in pdf.pages:
            words = page.extract_words()
            if not words or len(words) < 5:
                continue
            # 1) 找表头，确定7列的中心锚点
            col_centers = {}
            for w in words:
                t = re.sub(r"星期", "周", w["text"].strip())
                if t in col_names:
                    col_centers[col_names.index(t)] = (w["x0"] + w["x1"]) / 2
            if len(col_centers) == col_count:
                centers = [col_centers[i] for i in range(col_count)]
                span = (centers[-1] - centers[0]) / (col_count - 1)
                bounds = []
                for i, cx in enumerate(centers):
                    left = cx - span / 2 if i == 0 else (centers[i-1] + cx) / 2
                    right = cx + span / 2 if i == col_count - 1 else (cx + centers[i+1]) / 2
                    bounds.append((left, right))
                header_bottom = max(w["bottom"] for w in words
                                    if re.sub(r"星期", "周", w["text"].strip()) in col_names)
                content_words = [w for w in words if w["top"] > header_bottom]
                # 左侧节次标签栏（在最左列边界之外的）直接丢弃
                content_words = [w for w in content_words
                                 if any(b[0] <= (w["x0"] + w["x1"]) / 2 <= b[1] for b in bounds)]
            else:
                # 找不到表头，退回旧的均匀网格（x0改用中心点）
                content_words = words
                ys = np.array([w["top"] for w in words])
                y_low, y_high = np.percentile(ys, 10), np.percentile(ys, 90)
                content_words = [w for w in content_words if y_low <= w["top"] <= y_high]
                if len(content_words) < 3:
                    content_words = words
                xs = [(w["x0"] + w["x1"]) / 2 for w in content_words]
                x_min, x_max = min(xs), max(xs)
                bounds = [(x_min + i * (x_max - x_min) / col_count,
                           x_min + (i + 1) * (x_max - x_min) / col_count) for i in range(col_count)]
            if not content_words:
                continue
            y_min = min(w["top"] for w in content_words)
            y_max = max(w["bottom"] for w in content_words)
            cell_h = max((y_max - y_min) / row_count, 0.1)
            for w in content_words:
                wx = (w["x0"] + w["x1"]) / 2
                wy = (w["top"] + w["bottom"]) / 2 - y_min
                col_idx = next((i for i, b in enumerate(bounds) if b[0] <= wx <= b[1]), -1)
                row_idx = int(wy / cell_h)
                if 0 <= col_idx < col_count and 0 <= row_idx < row_count:
                    grid[row_idx][col_idx] += w["text"] + " "
    # 整表为空（如图片型PDF无文字层）→ 返回空串，让上层降级
    if not any(cell.strip() for row in grid for cell in row):
        return ""
    md = ["| 节次 | " + " | ".join(WEEK_NAMES) + " |",
          "| ---- | " + " | ".join(["----"] * 7) + " |"]
    for r in range(row_count):
        md.append(f"| {r+1} | " + " | ".join(grid[r]) + " |")
    table_text = "\n".join(md)
    table_text = re.sub(r"打印时间[:：].*", "", table_text)
    table_text = re.sub(r"学号[:：].*", "", table_text)
    return table_text

def get_pdf_table_text(pdf_bytes):
    """三级提取，返回 (文本, 方式)。方式: table=表格线识别 grid=坐标网格 raw=纯文本 none=失败"""
    # ① pdfplumber 表格线识别（有框线课表最准）
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            all_rows = []
            for page in pdf.pages:
                for table in page.extract_tables():
                    for row in (table or []):
                        all_rows.append(row)
            if all_rows:
                # 过滤图例/杂项行（如": 理论 : 上机 : 实验..."）
                all_rows = [r for r in all_rows
                            if not re.match(r"^[:：]", "".join(_cell_text(c) for c in r).strip())]
                # 自动定位表头行（含≥3个星期标签的行），丢弃其上方的标题/杂项行
                hdr_idx = -1
                for i, r in enumerate(all_rows):
                    if sum(1 for c in r if _norm_header(c) in WEEK_NAMES) >= 3:
                        hdr_idx = i
                        break
                has_day_header = hdr_idx >= 0 or bool(re.search(r"星期|周[一二三四五六日]", "".join(_cell_text(c) for c in all_rows[0]))) if all_rows else False
                # 表头不在表格里时不用表格线结果（列无法映射星期），降级走锚点网格
                if has_day_header:
                    if hdr_idx >= 0:
                        header = [_norm_header(c) for c in all_rows[hdr_idx]]
                        body = all_rows[hdr_idx + 1:]
                    else:
                        header = [_norm_header(c) for c in all_rows[0]]
                        body = all_rows[1:]
                    if body:
                        # 确保有节次列，便于AI对齐行号
                        if not any("节次" in h for h in header):
                            header = ["节次"] + list(header)
                            body = [[str(i + 1)] + list(r) for i, r in enumerate(body)]
                        # 合并跨页/合并单元格产生的无节次延续行，避免AI误判行位置
                        sec_col = next((i for i, h in enumerate(header) if "节次" in h), -1)
                        if sec_col >= 0:
                            merged = []
                            for r in body:
                                cells = ([_cell_text(c) for c in list(r)] + [""] * len(header))[:len(header)]
                                if not cells[sec_col] and merged:
                                    prev = (list(merged[-1]) + [""] * len(header))[:len(header)]
                                    merged[-1] = [(a + " " + b).strip() if b else a for a, b in zip(prev, cells)]
                                else:
                                    merged.append(cells)
                            body = merged
                        text = _build_markdown_table(body, header)
                        text = re.sub(r"打印时间[:：].*", "", text)
                        text = re.sub(r"学号[:：].*", "", text)
                        if text.strip():
                            return text, "table"
    except Exception:
        pass
    # ② 表头锚点坐标网格
    try:
        text = _grid_extract_text(pdf_bytes)
        if text and len(text.strip()) > 30:
            return text, "grid"
    except Exception:
        pass
    # ③ 纯文本兜底
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        txt = "".join(page.get_text() for page in doc)
        doc.close()
        if txt.strip():
            return txt, "raw"
    except Exception:
        pass
    return "", "none"

def pdf_to_images_b64(pdf_bytes, max_pages=2, zoom=2.5):
    """把课表PDF整页渲染成高清PNG，供视觉大模型直接读图。"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    imgs = []
    for page in doc:
        if len(imgs) >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        imgs.append(base64.b64encode(pix.tobytes("png")).decode())
    doc.close()
    return imgs

def extract_name_from_text(text):
    m = re.search(r"([\u4e00-\u9fa5]{2,4})课表", text)
    return m.group(1).strip() if m else ""

def parse_filename_phone_and_role(filename):
    name_part = filename.split("-课表")[0]
    parts = name_part.split("&")
    role = "学员"
    phone = ""
    if len(parts) >= 3:
        role_raw = parts[0]
        if "职委" in role_raw:
            role = "职委"
        else:
            role = "学员"
        phone_candidate = parts[2].strip()
        if re.fullmatch(r"1[3-9]\d{9}", phone_candidate):
            phone = phone_candidate
    return role, phone

# ====================== AI提示词【抗幻觉】 ======================
EXTRACT_SYS_PROMPT = """
你是严谨课表提取工具，负责从大学课表中提取有课时段。
⚠️最高优先级铁律：**没有在提供的课表材料中明确出现的星期、节次，绝对禁止写入结果，禁止脑补、禁止推测、禁止联想！原文没有就不要输出，宁可返回空数组。**
1.星期只能：周一、周二、周三、周四、周五、周六、周日，不许简写。
2.节次只允许1‑13的整数，其余全部丢弃。
3.输出字段仅允许week、sections、courses，不能增加任何别的key。
4.courses是该星期的课程名数组（如["高等数学","大学英语"]），仅作核对用，取不到就给[]。
5.某星期完全无课，该星期不要写入数组。
6.⚠️合并单元格：一个课程格子如果跨多节（如第1‑2节连排、第3‑5节大课），sections必须包含它占据的所有节次，一个都不能漏。
7.只输出纯净JSON数组，禁止```、注释、markdown、任何解释文字。
节次参考：
1:08:00-08:45｜2:08:50-09:35｜3:09:50-10:35｜4:10:40-11:25｜5:11:30-12:15
6:14:05-14:50｜7:14:55-15:40｜8:15:45-16:30｜9:16:40-17:25
10:17:30-18:15｜11:18:30-19:15｜12:19:20-20:05｜13:20:10-20:55
✅正确示例：[{"week":"周一","sections":[1,2],"courses":["高等数学"]},{"week":"周六","sections":[11],"courses":[]}]
"""
EXTRACT_USER_PROMPT_VL = """这是一张大学课表的图片（可能有多张，图1为主课表）。
表格布局：最上面一行是表头（周一到周日），左边一列是节次标签（第1节到第13节及时间），中间每个格子是课程信息（课程名、教室、教师，一门课可能连排多节）。
请逐格仔细看图，提取所有有课的（星期, 节次）组合，严格按系统要求的JSON数组格式输出。格子为空就是没课，绝对不要脑补。"""
PERSONAL_SYS_PROMPT = """
任务：解析用户个性化额外禁止值班需求，输出JSON字符串数组。
铁律：
1. 不许重复课表已经算出的冲突时段！**只提取用户原话明确说出的限制，禁止脑补、禁止联想额外限制。**
2.输出只能是："周X上午"、"周X下午"、"周X晚上"
3.合法集合：["周一上午","周一下午","周一晚上","周二上午","周二下午","周二晚上","周三上午","周三下午","周三晚上","周四上午","周四下午","周四晚上","周五上午","周五下午","周五晚上","周六上午","周六下午","周六晚上","周日上午","周日下午","周日晚上"]
4.没有额外限制返回[]。
5.只输出纯JSON，禁止markdown、注释、解释。
✅示例：["周日上午","周二晚上"]
✅无限制示例：[]
"""

def _make_http_session():
    sess = requests.Session()
    adapt = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1), pool_connections=1, pool_maxsize=1)
    sess.mount("https://", adapt)
    return sess

def _parse_json_array(out: str):
    out = (out or "").strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n?```\s*$", "", out)
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        raise ValueError("AI没有返回合法JSON数组")
    arr = json.loads(match.group(0))
    if not isinstance(arr, list):
        raise ValueError("AI返回的不是JSON数组")
    return arr

def _clean_course_data(raw):
    """字段白名单校验：星期、节次、课程名全部过滤非法值。"""
    valid_weeks = {"周一","周二","周三","周四","周五","周六","周日"}
    cleaned = []
    if not isinstance(raw, list):
        return cleaned
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        w = entry.get("week", "")
        sec_list = entry.get("sections", [])
        if w not in valid_weeks:
            continue
        clean_sec = sorted({x for x in sec_list if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 13})
        if clean_sec:
            courses = [safe_str(c) for c in entry.get("courses", []) if safe_str(c)]
            cleaned.append({"week": w, "sections": clean_sec, "courses": courses})
    return cleaned

# ========= 文本路线：qwen-plus 解析规整表格文本（带重试） =========
def _call_course_ai(prompt_text:str):
    last_err = None
    for _ in range(MAX_RETRY + 1):
        try:
            resp = Generation.call(
                model=COURSE_TEXT_MODEL,
                messages=[
                    {"role":"system","content":EXTRACT_SYS_PROMPT},
                    {"role":"user","content":prompt_text}
                ],
                temperature=0.0,
                max_tokens=2000,
                result_format="message"
            )
            if resp.status_code != 200:
                raise ValueError(f"千问API异常 code:{resp.status_code} {resp.message}")
            out = resp.output.choices[0].message.content
            return _parse_json_array(out)
        except Exception as e:
            last_err = e
    raise last_err

# ========= 二次校验：把初稿+原文交回AI逐条核对，删幻觉、补遗漏 =========
VERIFY_PROMPT = """
任务：核对并修正课表提取初稿。
规则：
1. 初稿中每一条(week,sections)都必须能在课表原文中找到对应星期列、对应节次行的课程格子；找不到的整条删除，禁止保留。
2. 原文中存在但初稿漏掉的课程格子一律补上，注意单元格内"(X-Y节)"表示连排，sections要含全部节次。
3. 输出字段仅允许week、sections、courses；只输出纯净JSON数组，禁止解释、禁止markdown。
"""

def _verify_course_ai(table_text:str, draft):
    check_prompt = f"""课表原文：
{table_text[:12000]}

初步提取结果（待核对）：
{json.dumps(draft, ensure_ascii=False)}
"""
    last_err = None
    for _ in range(MAX_RETRY + 1):
        try:
            resp = Generation.call(
                model=COURSE_TEXT_MODEL,
                messages=[
                    {"role":"system","content":EXTRACT_SYS_PROMPT + "\n" + VERIFY_PROMPT},
                    {"role":"user","content":check_prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
                result_format="message"
            )
            if resp.status_code != 200:
                raise ValueError(f"千问校验API异常 code:{resp.status_code} {resp.message}")
            out = resp.output.choices[0].message.content
            return _parse_json_array(out)
        except Exception as e:
            last_err = e
    raise last_err

# ========= 视觉路线：qwen-vl-max 直接看课表图片（带重试） =========
def _call_course_ai_vl(images_b64):
    if not images_b64:
        raise ValueError("课表PDF渲染图片失败")
    content = [{"image": f"data:image/png;base64,{img}"} for img in images_b64]
    content.append({"text": EXTRACT_USER_PROMPT_VL + "\n系统要求：星期只能周一~周日；节次1‑13整数；输出字段week/sections/courses；只输出纯净JSON数组。"})
    last_err = None
    for _ in range(MAX_RETRY + 1):
        try:
            resp = MultiModalConversation.call(
                model=COURSE_VL_MODEL,
                messages=[{"role":"user","content":content}],
                temperature=0.0,
                max_tokens=2000
            )
            if resp.status_code != 200:
                raise ValueError(f"千问视觉API异常 code:{resp.status_code} {resp.message}")
            parts = resp.output.choices[0].message.content
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            return _parse_json_array(text)
        except Exception as e:
            last_err = e
    raise last_err

# ========= 火山方舟doubao，关闭深度思考，防止脑补 =========
def _call_personal_ai(prompt_text:str):
    sess = _make_http_session()
    headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
    url = ARK_ENDPOINT.rstrip("/") + "/chat/completions"
    payload = {
        "model": ARK_MODEL,
        "messages": [
            {"role":"system","content":PERSONAL_SYS_PROMPT},
            {"role":"user","content":prompt_text}
        ],
        "temperature":0,
        "max_tokens":1000,
        "extra_body":{"thinking":{"type":"disabled"}}
    }
    resp = sess.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        return []
    arr = json.loads(match.group(0))
    raw_list = [s.strip() for s in arr if isinstance(s,str)]
    filtered = [x for x in raw_list if x in ALL_SLOTS]
    return filtered

# 节次时间(分钟，从0点起算)
SECTION_TIME = {
    1: (8*60+0, 8*60+45),
    2: (8*60+50, 9*60+35),
    3: (9*60+50,10*60+35),
    4: (10*60+40,11*60+25),
    5: (11*60+30,12*60+15),
    6: (14*60+5, 14*60+50),
    7: (14*60+55,15*60+40),
    8: (15*60+45,16*60+30),
    9: (16*60+40,17*60+25),
    10:(17*60+30,18*60+15),
    11:(18*60+30,19*60+15),
    12:(19*60+20,20*60+5),
    13:(20*60+10,20*60+55)
}

def calc_course_conflict(ai_course_data):
    conflict = set()
    for item in ai_course_data:
        w = item["week"]
        secs = item["sections"]
        for s in secs:
            if s not in SECTION_TIME:
                continue
            c_start, _ = SECTION_TIME[s]
            if c_start < 13 * 60:
                shift_name = "上午"
            elif c_start < 17 * 60:
                shift_name = "下午"
            else:
                shift_name = "晚上"
            conflict.add(f"{w}{shift_name}")
    return sorted(list(conflict))

def fast_ai_extract(pdf_bytes:bytes, personal_need:str=""):
    table_text, method = get_pdf_table_text(pdf_bytes)
    ai_course_data = []
    route = ""
    prompt_course = f"""
下面是课表原文（Markdown表格）。表头行标注了星期几（顺序可能从周日开始，以表头实际标注为准）；有一列是节次编号；每个课程单元格里的"(X-Y节)"表示该课占据第X节到第Y节。请提取所有有课的星期和节次，以单元格的星期列位置和其内的(X-Y节)信息为准：
{table_text[:12000]}
"""
    # ---- 路线1：仅当提取出"框线+真实星期表头"的高置信表格 → 文本模型（快、省） ----
    if method == "table" and len(table_text.strip()) >= 10:
        try:
            ai_course_data = _clean_course_data(_call_course_ai(prompt_course))
            if ai_course_data:
                # 二次校验：核对初稿，删除原文不支持的、补充遗漏的
                try:
                    verified = _clean_course_data(_verify_course_ai(table_text, ai_course_data))
                    if verified:
                        ai_course_data = verified
                        route = f"文本路线：框线表格提取 + {COURSE_TEXT_MODEL}（含二次校验）"
                    else:
                        route = f"文本路线：框线表格提取 + {COURSE_TEXT_MODEL}"
                except Exception:
                    route = f"文本路线：框线表格提取 + {COURSE_TEXT_MODEL}"
        except Exception:
            ai_course_data = []
    # ---- 路线2：其余情况（网格近似/图片型PDF/文本路线失败）→ 视觉大模型直接读图（最准） ----
    if not ai_course_data:
        try:
            imgs = pdf_to_images_b64(pdf_bytes)
            ai_course_data = _clean_course_data(_call_course_ai_vl(imgs))
            route = f"视觉路线：{COURSE_VL_MODEL} 直接读图"
        except Exception as e_vl:
            # ---- 路线3：视觉路线也失败 → 文本路线最后兜底 ----
            if method in ("grid", "raw") and len(table_text.strip()) >= 10:
                try:
                    ai_course_data = _clean_course_data(_call_course_ai(prompt_course))
                    route = f"文本兜底路线：{method}表格 + {COURSE_TEXT_MODEL}"
                except Exception:
                    ai_course_data = []
            if not ai_course_data:
                return f"课表识别失败:{str(e_vl)[:60]}"
    try:
        if st.runtime.exists():
            with st.expander(f"🔍DEBUG AI读取课表结果｜识别路线：{route}｜个性化模型：doubao2.1‑seed‑pro"):
                st.code(json.dumps(ai_course_data, ensure_ascii=False, indent=2), language="json")
                if table_text and method in ("table", "grid"):
                    with st.expander("📄 查看喂给AI的课表表格（核对是否错位）"):
                        st.text(table_text[:3000])
        course_conflict = calc_course_conflict(ai_course_data)
    except Exception as e:
        return f"课表识别失败:{str(e)[:60]}"
    personal_conflict = []
    if personal_need and personal_need.strip() not in ("","无"):
        prompt_personal = f"""
课表已经算出不能值班时段：{json.dumps(course_conflict, ensure_ascii=False)}
用户额外要求：{personal_need}
只输出用户额外新增禁止的时段，不要包含课表已经有的。
"""
        try:
            personal_conflict = _call_personal_ai(prompt_personal)
        except Exception:
            personal_conflict = []
    total = list(set(course_conflict + personal_conflict))
    if not total:
        return "识别为空，请手动填写"
    return "、".join(sorted(total))

# ========== 判断某人在某时段是否有空 ==========
def is_available(member, week_idx, shift_idx):
    class_conflict = safe_str(member.get("class_conflict"))
    week_name = WEEK_NAMES[week_idx]
    period = SHIFT_PERIOD[shift_idx]
    conflict_str = f"{week_name}{period}"
    if class_conflict and class_conflict != "无" and conflict_str in class_conflict:
        return False
    return True

# ========== 智能排班 ==========
def smart_schedule(member_list):
    stu_pool = [m for m in member_list if m["role"]=="学员"]
    wei_pool = [m for m in member_list if m["role"]=="职委"]
    st.info(f"📊 人员统计：学员 {len(stu_pool)} 人，职委 {len(wei_pool)} 人")
    if len(wei_pool) == 0:
        st.error("⚠️ 警告：人员库里没有职委！请去「人员库管理」把需要当职委的人身份改成「职委」")
    random.shuffle(stu_pool)
    random.shuffle(wei_pool)
    all_slots = []
    for w_idx, w in enumerate(WEEK_NAMES):
        for s_idx, s in enumerate(SHIFT_LIST):
            all_slots.append({"week_idx":w_idx, "shift_idx":s_idx, "week":w, "shift":s["shift_name"]})
    stu_count = {m["name"]:0 for m in stu_pool}
    wei_count = {m["name"]:0 for m in wei_pool}
    schedule = []
    slots_remaining = all_slots.copy()
    for slot in slots_remaining:
        w_idx = slot["week_idx"]
        s_idx = slot["shift_idx"]
        need_stu = SHIFT_LIST[s_idx]["need_stu"]
        slot["学员"] = []
        slot["职委"] = []
        available_stu = [m for m in stu_pool if is_available(m, w_idx, s_idx) and stu_count[m["name"]] < 1]
        if len(available_stu) < need_stu:
            for m in stu_pool:
                if m not in available_stu and stu_count[m["name"]] < 1:
                    available_stu.append(m)
        for i in range(min(need_stu, len(available_stu))):
            p = available_stu[i]
            slot["学员"].append(p)
            stu_count[p["name"]] += 1
    if wei_pool:
        wei_cycle = wei_pool.copy()
        wei_ptr = 0
        for slot in slots_remaining:
            w_idx = slot["week_idx"]
            s_idx = slot["shift_idx"]
            need_wei = SHIFT_LIST[s_idx]["need_wei"]
            picked = False
            for _ in range(len(wei_cycle)):
                candidate = wei_cycle[wei_ptr % len(wei_cycle)]
                wei_ptr += 1
                if wei_count[candidate["name"]] == 0 and is_available(candidate, w_idx, s_idx):
                    slot["职委"].append(candidate)
                    wei_count[candidate["name"]] += 1
                    picked = True
                    break
            if not picked:
                available_wei = [m for m in wei_pool if is_available(m, w_idx, s_idx)]
                if available_wei:
                    available_wei.sort(key=lambda x: wei_count[x["name"]])
                    slot["职委"].append(available_wei[0])
                    wei_count[available_wei[0]["name"]] += 1
                else:
                    slot["职委"].append(wei_pool[0])
                    wei_count[wei_pool[0]["name"]] += 1
            schedule.append(slot)
    rested = [name for name, cnt in stu_count.items() if cnt == 0]
    if rested:
        st.info(f"ℹ️ 本周共 {len(rested)} 名学员轮空：{'、'.join(rested)}")
    if wei_pool:
        st.success(f"✅ 职委排班完成：共 {len(wei_pool)} 名职委，平均每人排 {sum(wei_count.values())/len(wei_pool):.1f} 班")
    return schedule

# ========== 模式A：系统自动生成PDF ==========
def export_pdf_auto(schedule_data, template_name):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 14)
    c.drawCentredString(w/2, h-40, "武汉大学学生会值班表")
    y = h - 70
    if template_name == "武汉大学学生会水印模板":
        c.setFillColorRGB(0.85,0.85,0.85)
        c.setFont(FONT_NAME, 35)
        c.drawCentredString(w/2, h/2, "武汉大学学生会")
        c.setFillColorRGB(0,0,0)
    c.setFont(FONT_NAME, 9)
    for row in schedule_data:
        stu_text = ",".join([f'{x["name"]}({x["phone"]})' for x in row['学员']])
        wei_text = ",".join([f'{x["name"]}({x["phone"]})' for x in row['职委']])
        line = f"{row['week']} {row['shift']} | 学员：{stu_text} | 职委：{wei_text}"
        c.drawString(30, y, line)
        y -= 16
        if y < 40:
            c.showPage()
            y = h-40
            c.setFont(FONT_NAME,9)
    c.save()
    buf.seek(0)
    return buf

# ========== 模式B：填充用户模板 ==========
def fill_user_template_pdf(template_bytes, schedule_data, offset_x:int=0, offset_y:int=-30):
    doc = fitz.open(stream=template_bytes, filetype="pdf")
    page = doc[0]
    start_x = 158 + offset_x
    start_y = 175 + offset_y
    font_title = 6
    font_name = 6
    font_phone = 4.5
    gap_title = 5
    gap_name = 4.5
    gap_phone = 4
    cell_w = 82
    cell_h = 115
    cell_dict = {}
    for si in range(3):
        for wi in range(7):
            cell_dict[(si,wi)] = {"stu":[], "wei":[]}
    for entry in schedule_data:
        k = (entry["shift_idx"], entry["week_idx"])
        if k in cell_dict:
            for p in entry["学员"]:
                cell_dict[k]["stu"].append( (p["name"], p.get("phone","")) )
            for p in entry["职委"]:
                cell_dict[k]["wei"].append( (p["name"], p.get("phone","")) )
    total_block_height = gap_title + gap_name + gap_phone + gap_name + gap_phone + gap_title + gap_name + gap_phone
    top_pad = (cell_h - total_block_height) // 2
    for shift_idx in range(3):
        for week_idx in range(7):
            cell_data = cell_dict[(shift_idx, week_idx)]
            stu_list = cell_data["stu"][0:2]
            wei_list = cell_data["wei"][0:1]
            pos_x = start_x + week_idx * cell_w + 5
            pos_y = start_y + shift_idx * cell_h + top_pad
            cy = pos_y
            page.insert_text((pos_x, cy), "值班学员", fontsize=font_title, fontname="china-s")
            cy += gap_title
            for name, phone in stu_list:
                page.insert_text((pos_x, cy), name, fontsize=font_name, fontname="china-s")
                cy += gap_name
                if phone:
                    page.insert_text((pos_x, cy), str(phone), fontsize=font_phone, fontname="china-s")
                    cy += gap_phone
                else:
                    cy += gap_name
            page.insert_text((pos_x, cy), "轮值职委", fontsize=font_title, fontname="china-s")
            cy += gap_title
            for name, phone in wei_list:
                page.insert_text((pos_x, cy), name, fontsize=font_name, fontname="china-s")
                cy += gap_name
                if phone:
                    page.insert_text((pos_x, cy), str(phone), fontsize=font_phone, fontname="china-s")
                    cy += gap_phone
                else:
                    cy += gap_name
    out_buf = BytesIO()
    doc.save(out_buf)
    doc.close()
    out_buf.seek(0)
    return out_buf

# ========== 页面入口 ==========
st.set_page_config(page_title="值班排班系统", layout="wide")
if "login" not in st.session_state:
    st.session_state.login = False
if "current_admin" not in st.session_state:
    st.session_state.current_admin = ""
if "schedule_result" not in st.session_state:
    st.session_state.schedule_result = None
if "user_template_bytes" not in st.session_state:
    st.session_state.user_template_bytes = None
if "temp_new_member" not in st.session_state:
    st.session_state.temp_new_member = []
if "temp_pdf_bytes" not in st.session_state:
    st.session_state.temp_pdf_bytes = []
if "member_pending" not in st.session_state:
    st.session_state.member_pending = None
if "ai_failed_names" not in st.session_state:
    st.session_state.ai_failed_names = []
_query = st.query_params
_is_self_page = _query.get("page", "") == "self"
if _is_self_page:
    menu = "成员自助上传"
else:
    menu = st.sidebar.radio("导航", ["管理员后台", "成员自助上传"])
if menu == "管理员后台":
    if not st.session_state.login:
        st.title("管理员登录")
        admin_user = st.text_input("管理员账号")
        pwd_in = st.text_input("输入密码", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认登录"):
                admin_info = verify_admin(admin_user, pwd_in)
                if admin_info:
                    st.session_state.login = True
                    st.session_state.current_admin = admin_user
                    st.rerun()
                else:
                    st.error("账号或密码错误（默认账号：admin，密码：123456）")
    else:
        st.title(f"管理员后台｜排班管理（当前登录：{st.session_state.current_admin}）")
        st.caption("链路：PDF→表格线/锚点网格提取（失败自动切换）→qwen‑vl‑max视觉模型直接读图识别课表；doubao2.1‑seed‑pro解析个性化")
        if is_super_admin(st.session_state.current_admin):
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["批量上传课表","人员库管理","自动排班","导出PDF","分享链接","管理员管理"])
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["批量上传课表","人员库管理","自动排班","导出PDF","分享链接"])
            tab6 = None
        with tab1:
            st.subheader("批量上传PDF课表（文件名格式：身份&姓名&手机号‑课表.pdf）")
            if st.session_state.ai_failed_names:
                st.warning("⚠️ 以下人员课表识别异常，请手动核对/补填不能值班时段：" + "、".join(st.session_state.ai_failed_names))
            up_files = st.file_uploader("选择课表PDF", type="pdf", accept_multiple_files=True)
            if up_files and st.button("📂 第一步：缓存PDF二进制，解析姓名身份电话"):
                with st.spinner("读取PDF文件..."):
                    pdf_bytes_list = []
                    new_members = []
                    for f in up_files:
                        b = f.read()
                        pdf_bytes_list.append(b)
                        doc_tmp = fitz.open(stream=b, filetype="pdf")
                        txt_tmp = doc_tmp[0].get_text()
                        doc_tmp.close()
                        name = extract_name_from_text(txt_tmp)
                        role_from_file, phone_from_file = parse_filename_phone_and_role(f.name)
                        if not name:
                            continue
                        new_members.append({
                            "name": name,
                            "phone": phone_from_file,
                            "role": role_from_file,
                            "class_conflict": "",
                            "personal_need": ""
                        })
                    st.session_state.temp_pdf_bytes = pdf_bytes_list
                    seen = set()
                    unique = []
                    for item in new_members:
                        if item["name"] not in seen:
                            seen.add(item["name"])
                            unique.append(item)
                    st.session_state.temp_new_member = unique
                st.success(f"已缓存 {len(unique)} 份PDF，填写个性化需求后执行AI识别")
                st.rerun()
            if len(st.session_state.temp_new_member) > 0:
                st.divider()
                st.subheader("📋 预览表格（先填个性化需求，再点AI识别，AI会综合课表+需求一起算）")
                df_temp = pd.DataFrame(st.session_state.temp_new_member)
                temp_config = {
                    "role": st.column_config.SelectboxColumn("身份", options=["学员","职委"], required=True),
                    "class_conflict": st.column_config.TextColumn("AI识别不能值班时段（课表+需求综合）"),
                    "personal_need": st.column_config.TextColumn("个性化需求（手动填，AI会自动识别各种说法）")
                }
                edited_temp = st.data_editor(df_temp, use_container_width=True, column_config=temp_config, num_rows="dynamic", key="temp_editor")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("🤖 第二步：AI综合识别（课表+个性化需求）"):
                        current_rows = edited_temp.to_dict("records")
                        need_ai_idx = []
                        need_pdf_bin = []
                        need_ai_personal = []
                        for i, row in enumerate(current_rows):
                            if not safe_str(row.get("class_conflict")):
                                need_ai_idx.append(i)
                                need_pdf_bin.append(st.session_state.temp_pdf_bytes[i] if i < len(st.session_state.temp_pdf_bytes) else b"")
                                need_ai_personal.append(safe_str(row.get("personal_need")))
                        if need_pdf_bin:
                            with st.spinner(f"AI处理 {len(need_pdf_bin)} 份课表..."):
                                bar = st.progress(0)
                                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                    futures = {executor.submit(fast_ai_extract, pdf_bin, p): idx for idx, pdf_bin, p in zip(need_ai_idx, need_pdf_bin, need_ai_personal)}
                                    done = 0
                                    for fut in as_completed(futures):
                                        idx = futures[fut]
                                        try:
                                            current_rows[idx]["class_conflict"] = fut.result()
                                        except Exception as e:
                                            current_rows[idx]["class_conflict"] = f"识别失败: {str(e)[:60]}"
                                        done += 1
                                        bar.progress(done/len(need_pdf_bin))
                            for row in current_rows:
                                row["personal_need"] = safe_str(row.get("personal_need"))
                                row["class_conflict"] = safe_str(row.get("class_conflict"))
                                row["name"] = safe_str(row.get("name"))
                                row["phone"] = safe_str(row.get("phone"))
                                row["role"] = safe_str(row.get("role")) if safe_str(row.get("role")) else "学员"
                            # 记录识别异常人员，rerun后在页面上方提示人工核对
                            st.session_state.ai_failed_names = [safe_str(r.get("name")) for r in current_rows if is_recognize_failed(r.get("class_conflict"))]
                            st.session_state.temp_new_member = current_rows
                            st.rerun()
                        else:
                            st.info("所有人都已识别，不需要AI！")
                with col_b:
                    if st.button("✅ 确认保存入库"):
                        all_mem = load_members()
                        exist_names = {x["name"] for x in all_mem}
                        add_list = edited_temp.to_dict("records")
                        new_add = []
                        skipped = []
                        for row in add_list:
                            nm = safe_str(row.get("name"))
                            if not nm or nm in exist_names:
                                continue
                            row["personal_need"] = safe_str(row.get("personal_need"))
                            row["class_conflict"] = safe_str(row.get("class_conflict"))
                            row["role"] = safe_str(row.get("role")) if safe_str(row.get("role")) else "学员"
                            # 识别失败/为空的记录直接入库会导致排班静默出错，必须人工补填后再保存
                            if is_recognize_failed(row["class_conflict"]):
                                skipped.append(nm)
                                continue
                            new_add.append(row)
                        all_mem.extend(new_add)
                        save_members(all_mem)
                        if skipped:
                            st.session_state.ai_failed_names = skipped
                            st.error(f"⚠️ 已有 {len(new_add)} 人入库；以下 {len(skipped)} 人识别异常或时段为空，未入库，请手动补填后再次保存：{'、'.join(skipped)}")
                        else:
                            st.session_state.ai_failed_names = []
                            st.session_state.temp_new_member = []
                            st.session_state.temp_pdf_bytes = []
                            st.success(f"成功录入 {len(new_add)} 人！")
                            st.rerun()
                with col_c:
                    if st.button("❌ 放弃本批"):
                        st.session_state.temp_new_member = []
                        st.session_state.temp_pdf_bytes = []
                        st.session_state.ai_failed_names = []
                        st.rerun()
        with tab2:
            st.subheader(f"人员库管理（共 {len(load_members())} 人）")
            mem = load_members()
            df = pd.DataFrame(mem)
            df.insert(0, "序号", range(1, len(df)+1))
            edit_config = {
                "序号": st.column_config.NumberColumn("序号", disabled=True),
                "role": st.column_config.SelectboxColumn("身份", options=["学员", "职委"], required=True),
                "class_conflict": st.column_config.TextColumn("不能值班时段（课表+需求综合）"),
                "personal_need": st.column_config.TextColumn("个性化需求原文")
            }
            edit_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", column_config=edit_config)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾保存修改"):
                    with st.spinner("正在保存人员数据..."):
                        save_df = edit_df.drop(columns=["序号"], errors="ignore")
                        records = save_df.to_dict("records")
                        for row in records:
                            row["personal_need"] = safe_str(row.get("personal_need"))
                            row["class_conflict"] = safe_str(row.get("class_conflict"))
                            row["role"] = safe_str(row.get("role")) if safe_str(row.get("role")) else "学员"
                        save_members(records)
                    st.success("已保存人员库！")
                    st.rerun()
            with col2:
                if st.button("🗑️一键清空全部人员"):
                    save_members([])
                    st.warning("人员库已清空！")
                    st.rerun()
        with tab3:
            st.subheader("一键排班（AI综合考虑课表冲突+个性化需求+职委均匀轮班）")
            mlist = load_members()
            if st.button("生成值班表"):
                with st.spinner("正在智能排班，综合考虑课表冲突和个性化需求..."):
                    result = smart_schedule(mlist)
                    st.session_state.schedule_result = result
                st.success("排班完成！")
            if st.session_state.schedule_result:
                st.dataframe(pd.DataFrame(st.session_state.schedule_result), use_container_width=True)
        with tab4:
            st.subheader("导出PDF（两种模式任选）")
            mode_sel = st.radio("选择导出模式", ["模式A：系统自动生成值班表", "模式B：上传我的空白模板PDF，自动填充"])
            if mode_sel == "模式A：系统自动生成值班表":
                pdf_template_sel = st.selectbox("PDF样式", TEMPLATE_OPTIONS)
                if not st.session_state.schedule_result:
                    st.warning("请先生成排班结果")
                else:
                    if st.button("导出自动生成的值班表PDF"):
                        with st.spinner("正在生成PDF..."):
                            pdf_buf = export_pdf_auto(st.session_state.schedule_result, pdf_template_sel)
                        st.download_button("下载值班表", pdf_buf, file_name="值班表.pdf", mime="application/pdf")
            else:
                st.info("上传你自己的空白值班表PDF模板，文字块整体缩小居中")
                user_up = st.file_uploader("上传你的空白值班表PDF模板", type="pdf")
                if user_up:
                    st.session_state.user_template_bytes = user_up.read()
                    st.success("模板已加载！")
                off_x = st.number_input("横向微调偏移", value=0)
                off_y = st.number_input("纵向微调偏移", value=-30)
                if not st.session_state.schedule_result:
                    st.warning("请先生成排班结果")
                elif not st.session_state.user_template_bytes:
                    st.warning("请先上传你的空白模板PDF")
                else:
                    if st.button("自动填充我的模板并导出"):
                        with st.spinner("正在填充你的模板PDF..."):
                            out_buf = fill_user_template_pdf(st.session_state.user_template_bytes, st.session_state.schedule_result, int(off_x), int(off_y))
                        st.download_button("下载填充好的值班表", out_buf, file_name="值班表_已填充.pdf", mime="application/pdf")
        with tab5:
            st.subheader("分享自助提交链接")
            st.info("把下面这个链接发给成员，他们可以免登录自助上传课表、填写个性化需求")
            try:
                base_app_url = str(st.context.url)
            except Exception:
                base_app_url = "http://localhost:8501"
            base_app_url = base_app_url.rstrip("/")
            self_link = f"{base_app_url}?page=self"
            st.text_input("自动识别得到网址（请勿修改）", value=base_app_url, disabled=True)
            st.code(self_link, language="text")
            if st.button("复制链接提示"):
                st.success("链接已生成，直接复制发给成员即可！")
        if tab6 is not None:
            with tab6:
                st.subheader("管理员管理（仅超级管理员可见）")
                st.markdown("#### 修改超级管理员账号")
                old_pwd = st.text_input("原密码", type="password", key="old_pwd_super")
                new_username = st.text_input("新管理员账号（留空则不修改）", value="", key="new_username_super")
                new_pwd = st.text_input("新密码（留空则不修改）", type="password", key="new_pwd_super")
                confirm_pwd = st.text_input("确认新密码", type="password", key="confirm_pwd_super")
                if st.button("保存超级管理员修改"):
                    admin_data = load_admin()
                    super_admin = None
                    for a in admin_data["admins"]:
                        if a.get("is_super", False):
                            super_admin = a
                            break
                    if not super_admin:
                        st.error("未找到超级管理员账号")
                    elif super_admin["password"] != old_pwd:
                        st.error("原密码错误")
                    elif new_pwd and new_pwd != confirm_pwd:
                        st.error("两次新密码不一致")
                    else:
                        if new_username:
                            super_admin["username"] = new_username.strip()
                        if new_pwd:
                            super_admin["password"] = new_pwd
                        save_admin(admin_data)
                        if new_username:
                            st.session_state.current_admin = new_username.strip()
                        st.success("超级管理员账号信息已更新！")
                        st.rerun()
                st.divider()
                st.markdown("#### 修改当前账号密码")
                old_pwd2 = st.text_input("原密码", type="password", key="old_pwd")
                new_pwd2 = st.text_input("新密码", type="password", key="new_pwd")
                confirm_pwd2 = st.text_input("确认新密码", type="password", key="confirm_pwd")
                if st.button("修改密码"):
                    if new_pwd2 != confirm_pwd2:
                        st.error("两次新密码不一致")
                    elif len(new_pwd2) < 6:
                        st.error("新密码至少6位")
                    else:
                        admin_data = load_admin()
                        for a in admin_data["admins"]:
                            if a["username"] == st.session_state.current_admin and a["password"] == old_pwd2:
                                a["password"] = new_pwd2
                                save_admin(admin_data)
                                st.success("密码修改成功！")
                                break
                        else:
                            st.error("原密码错误")
                st.divider()
                st.markdown("#### 添加新管理员")
                new_admin_user = st.text_input("新管理员账号", key="new_admin_user")
                new_admin_pwd = st.text_input("新管理员密码", type="password", key="new_admin_pwd")
                if st.button("添加管理员"):
                    admin_data = load_admin()
                    exist = any(a["username"] == new_admin_user for a in admin_data["admins"])
                    if exist:
                        st.error("该账号已存在")
                    elif not new_admin_user or not new_admin_pwd:
                        st.warning("账号和密码不能为空")
                    else:
                        admin_data["admins"].append({"username": new_admin_user, "password": new_admin_pwd, "is_super": False})
                        save_admin(admin_data)
                        st.success(f"管理员 {new_admin_user} 添加成功！")
                st.divider()
                st.markdown("#### 管理员列表")
                admin_data = load_admin()
                df_admin = pd.DataFrame(admin_data["admins"])
                df_admin["角色"] = df_admin["is_super"].apply(lambda x: "超级管理员" if x else "普通管理员")
                st.dataframe(df_admin[["username","角色"]], use_container_width=True)
                st.divider()
                st.markdown("#### 删除管理员")
                normal_admins = [a["username"] for a in admin_data["admins"] if not a.get("is_super", False)]
                if normal_admins:
                    delete_user = st.selectbox("选择要删除的管理员", normal_admins)
                    if st.button("删除该管理员"):
                        admin_data = load_admin()
                        admin_data["admins"] = [a for a in admin_data["admins"] if a["username"] != delete_user]
                        save_admin(admin_data)
                        st.success(f"管理员 {delete_user} 已删除！")
                else:
                    st.info("暂无普通管理员可删除")
        if st.button("退出登录"):
            st.session_state.login = False
            st.session_state.current_admin = ""
            st.rerun()
elif menu == "成员自助上传":
    st.title("成员自助提交信息 & 上传课表")
    st.caption("识别逻辑与管理员后台上传完全一致：PDF→表格线/锚点网格提取（失败自动切换）→qwen‑vl‑max视觉模型直接读图识别课表；doubao2.1‑seed‑pro解析个性化需求。提交前请核对识别结果。")
    with st.form("member_form"):
        name = st.text_input("姓名")
        phone = st.text_input("联系电话")
        role_sel = st.radio("身份", ["学员","职委"])
        limit_input = st.text_input("个性化需求（选填，各种说法都行：周末不值班 / 只在周六下午有空 / 周一上午有课）")
        pdf_file = st.file_uploader("上传你的课表PDF", type="pdf")
        submit = st.form_submit_button("① 开始识别课表")
    if submit:
        if not (name and phone and pdf_file):
            st.warning("姓名、电话、课表都需要填！")
        else:
            with st.spinner("AI分析课表，请稍候..."):
                pdf_bin = pdf_file.read()
                class_conflict = fast_ai_extract(pdf_bin, limit_input)
            if is_recognize_failed(class_conflict):
                st.session_state.member_pending = None
                st.error(f"❌ 课表识别未成功：{class_conflict}。请重新上传清晰、完整的教务系统课表PDF；若多次失败请联系管理员手动录入。")
            else:
                st.session_state.member_pending = {
                    "name": safe_str(name),
                    "phone": safe_str(phone),
                    "role": role_sel,
                    "class_conflict": class_conflict,
                    "personal_need": safe_str(limit_input),
                }
                st.rerun()
    pending = st.session_state.get("member_pending")
    if pending:
        st.divider()
        st.subheader("② 核对识别结果，确认后提交入库")
        st.info(f"识别到 {len([x for x in pending['class_conflict'].split('、') if x])} 个不能值班时段。请对照你的课表检查：多出或缺少的时段可以直接修改下面的文字。")
        edited_conflict = st.text_area(
            "我不能值班的时段（用「、」分隔，例如：周二下午、周五晚上）",
            value=pending["class_conflict"], height=100,
            key=f"member_confirm_{pending['name']}_{abs(hash(pending['class_conflict'])) % 100000}")
        st.caption(f"提交人：{pending['name']} ｜ 电话：{pending['phone']} ｜ 身份：{pending['role']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 确认无误，提交入库"):
                final_conflict = normalize_conflict_text(edited_conflict)
                if not final_conflict:
                    st.error("时段格式不正确：请按「周二下午」这样的格式填写，多个时段用「、」分隔。")
                else:
                    all_mem = load_members()
                    exist = False
                    for m in all_mem:
                        if m["name"] == pending["name"]:
                            m["phone"] = pending["phone"]
                            m["role"] = pending["role"]
                            m["class_conflict"] = final_conflict
                            m["personal_need"] = pending["personal_need"]
                            exist = True
                    if not exist:
                        all_mem.append({"name": pending["name"], "phone": pending["phone"], "role": pending["role"],
                                        "class_conflict": final_conflict, "personal_need": pending["personal_need"]})
                    save_members(all_mem)
                    st.session_state.member_pending = None
                    st.success("提交成功！管理员已能看到你的不能值班时段。")
                    st.rerun()
        with c2:
            if st.button("🔁 重新上传课表"):
                st.session_state.member_pending = None
                st.rerun()
