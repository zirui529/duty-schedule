import streamlit as st
import fitz
import pandas as pd
import json
import os
import re
import random
import base64
import time
from datetime import datetime
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
MAX_WORKERS = 8
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
# 注意：批量识别线程异常时写的是"识别失败: ..."，必须一并拦截，否则会当作有效数据入库
FAIL_PREFIXES = ("课表识别失败", "识别失败", "识别为空", "课表文本提取失败")

def is_recognize_failed(text):
    t = safe_str(text)
    return (not t) or t.startswith(FAIL_PREFIXES)

def normalize_conflict_text(text):
    """把「周二下午,周五晚上」等各种分隔符写法统一成「周二下午、周五晚上」，只保留合法时段。"""
    tokens = re.split(r"[、,，;；/\s]+", safe_str(text))
    valid = {t.strip() for t in tokens if t.strip() in ALL_SLOTS}
    return "、".join(sorted(valid, key=slot_sort_key))

# ========== 管理员文件读写 ==========
def _atomic_write_json(path, data):
    """原子写入：先写临时文件再替换，避免进程中断把JSON写坏（写坏的JSON会让应用启动即崩）。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _safe_load_json(path, default):
    """读JSON：文件损坏时自动备份为 .broken 并返回默认值，绝不让整个应用崩掉。"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            os.replace(path, f"{path}.broken")
        except Exception:
            pass
        return default

# ========== 识别质量日志（可观测：路线 / 规则覆盖率 / AI一致性 / 耗时） ==========
# 目的：累积后能看出"哪类课表模板容易识别错"，也便于把出错样本沉淀成回归测试集。
LOG_FILE = "recognition_log.json"
LOG_MAX = 500          # 只保留最近 N 条，防止文件无限增长吃满云端磁盘

ROUTE_TYPES = ("双通道", "规则", "视觉", "文本兜底")

def route_type_of(route):
    r = safe_str(route)
    if not r:
        return "未识别"
    if "失败" in r:
        return "失败"
    for t in ROUTE_TYPES:
        if t in r:
            return t
    return "其他"

def load_recognition_log():
    data = _safe_load_json(LOG_FILE, [])
    return data if isinstance(data, list) else []

def append_recognition_log(records):
    """批量追加日志。调用方统一在主线程写入，避免多线程同时写文件。"""
    records = [r for r in (records or []) if isinstance(r, dict)]
    if not records:
        return
    log = load_recognition_log()
    log.extend(records)
    if len(log) > LOG_MAX:
        log = log[-LOG_MAX:]
    _atomic_write_json(LOG_FILE, log)

def build_log_record(name, dbg, personal_need=""):
    """把 fast_ai_extract 的 debug_sink 整理成一行可统计的日志。"""
    dbg = dbg if isinstance(dbg, dict) else {}
    total = dbg.get("rule_text_cells") or 0
    covered = dbg.get("rule_covered") or 0
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": safe_str(name),
        "method": safe_str(dbg.get("method")) or "unknown",
        "route_type": route_type_of(dbg.get("route")),
        "route": safe_str(dbg.get("route")),
        "rule_covered": covered,
        "rule_text_cells": total,
        "cover_rate": round(covered / total, 3) if total else None,
        "ai_agree": dbg.get("ai_agree"),      # True=AI与规则完全一致 False=有差异 None=未比对
        "ai_used": dbg.get("ai_used", True),  # False=批量时关闭了AI（纯规则）
        "failed": bool(dbg.get("failed")),
        "elapsed_ms": dbg.get("elapsed_ms"),
        "has_personal_need": bool(safe_str(personal_need)),
    }

def summarize_recognition_log(log):
    """统计：总数、失败数、平均覆盖率、AI与规则不一致次数、各路线/提取方式分布、平均耗时。"""
    log = [r for r in (log or []) if isinstance(r, dict)]
    total = len(log)
    failed = sum(1 for r in log if r.get("failed"))
    rates = [r["cover_rate"] for r in log if isinstance(r.get("cover_rate"), (int, float))]
    ms = [r["elapsed_ms"] for r in log if isinstance(r.get("elapsed_ms"), (int, float))]
    route_cnt, method_cnt = {}, {}
    for r in log:
        rt = safe_str(r.get("route_type")) or "未识别"
        mt = safe_str(r.get("method")) or "unknown"
        route_cnt[rt] = route_cnt.get(rt, 0) + 1
        method_cnt[mt] = method_cnt.get(mt, 0) + 1
    return {
        "total": total,
        "failed": failed,
        "success": total - failed,
        "avg_cover": (sum(rates) / len(rates)) if rates else None,
        "ai_diff": sum(1 for r in log if r.get("ai_agree") is False),
        "ai_off": sum(1 for r in log if r.get("ai_used") is False),
        "avg_ms": (sum(ms) / len(ms)) if ms else None,
        "route_cnt": route_cnt,
        "method_cnt": method_cnt,
    }

def init_admin():
    if not os.path.exists(ADMIN_FILE):
        _atomic_write_json(ADMIN_FILE, {"admins": [{"username": DEFAULT_ADMIN_USER, "password": DEFAULT_ADMIN_PWD, "is_super": True}]})
    else:
        old = _safe_load_json(ADMIN_FILE, None)
        if not isinstance(old, dict):
            _atomic_write_json(ADMIN_FILE, {"admins": [{"username": DEFAULT_ADMIN_USER, "password": DEFAULT_ADMIN_PWD, "is_super": True}]})
        elif "admins" not in old:
            old_pwd = old.get("admin_pwd", DEFAULT_ADMIN_PWD)
            _atomic_write_json(ADMIN_FILE, {"admins": [{"username": DEFAULT_ADMIN_USER, "password": old_pwd, "is_super": True}]})

def reset_default_admin():
    _atomic_write_json(ADMIN_FILE, {"admins": [{"username": DEFAULT_ADMIN_USER, "password": DEFAULT_ADMIN_PWD, "is_super": True}]})

def load_admin():
    init_admin()
    data = _safe_load_json(ADMIN_FILE, None)
    if not isinstance(data, dict) or "admins" not in data or not isinstance(data["admins"], list):
        data = {"admins": [{"username": DEFAULT_ADMIN_USER, "password": DEFAULT_ADMIN_PWD, "is_super": True}]}
    # 防止"没有任何超级管理员"导致管理员管理页永远进不去、也无法重置（把自己锁在门外）
    if not any(a.get("is_super") for a in data["admins"] if isinstance(a, dict)):
        if not any(a.get("username") == DEFAULT_ADMIN_USER for a in data["admins"] if isinstance(a, dict)):
            data["admins"].append({"username": DEFAULT_ADMIN_USER, "password": DEFAULT_ADMIN_PWD, "is_super": True})
            _atomic_write_json(ADMIN_FILE, data)
    return data

def save_admin(data):
    _atomic_write_json(ADMIN_FILE, data)

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
    data = _safe_load_json(DATA_FILE, [])
    if not isinstance(data, list):
        return []
    for m in data:
        if not isinstance(m, dict):
            continue
        if "class_conflict" not in m:
            m["class_conflict"] = m.pop("limit_time", "")
        if "personal_need" not in m:
            m["personal_need"] = ""
        if "role" not in m or not m["role"]:
            m["role"] = "学员"
        if "phone" not in m:
            m["phone"] = ""
        m["name"] = safe_str(m.get("name"))
    return [m for m in data if isinstance(m, dict) and m.get("name")]

def save_members(data):
    _atomic_write_json(DATA_FILE, data)

# ========== PDF读取【三级降级：①表格线识别 ②表头锚点网格 ③纯文本】 ==========
def _cell_text(v):
    t = re.sub(r"[|\n\r]", " ", safe_str(v))
    # 先去掉所有空白（源PDF里字段名常被换行拆开，如"教学班组 成"）
    t = re.sub(r"\s+", "", t)
    # 按"(X-Y节)"课程条目切分：一格里可能有多门课，逐条保留课名+节次+周次+场地+教师
    ms = list(re.finditer(r"[^/]{1,40}\(\d+(?:-\d+)?节\)", t))
    if not ms:
        return t[:1000]
    entries = []
    for i, m in enumerate(ms):
        seg = t[m.start(): ms[i+1].start() if i + 1 < len(ms) else len(t)]
        seg = re.split(r"/教学班", seg)[0]  # 每条记录在教学班号等元数据处截断
        if seg:
            entries.append(seg)
    return "；".join(entries)[:1200]

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
    """四级提取，返回 (文本, 方式)。方式: list=清单式 table=表格线识别 grid=坐标网格 raw=纯文本 none=失败"""
    # ⓪ 清单式课表解析（武大部分导出版式：逐块列出 星期X/节次段/课程名/周数详情，无网格）
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception:
        raw = ""
    if raw:
        try:
            list_md = _parse_list_style_text(raw)
            if list_md:
                return list_md, "list"
        except Exception:
            pass
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

def _parse_list_style_text(raw_text):
    """清单式课表解析：结构为 星期X → [节次段(如3-4), 课程名, 周数:...] 逐块排列。
    每个块必须看到"周数"行才认定有效（防止网格版课表误触发）。
    返回规整Markdown表格（| 星期 | 节次 | 课程 | 详情 |），不匹配返回空串。"""
    if not raw_text or "周数" not in raw_text:
        return ""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    sec_line_re = re.compile(r"^(\d{1,2})\s*[-–—]\s*(\d{1,2})$")
    rows = []
    week = None
    state = "idle"  # idle=等星期/节次段  course=收课程名  detail=收周数详情
    cur_secs, cur_course, cur_detail, cur_has_detail = "", [], [], False

    def flush():
        nonlocal cur_secs, cur_course, cur_detail, cur_has_detail
        if week and cur_secs and cur_course and cur_has_detail:
            rows.append((week, cur_secs, "".join(cur_course), "".join(cur_detail)))
        cur_secs, cur_course, cur_detail, cur_has_detail = "", [], [], False

    for l in lines:
        wm = re.fullmatch(r"星期([一二三四五六日])", l)
        if wm:
            flush()
            week = "周" + wm.group(1)
            state = "idle"
            continue
        sm = sec_line_re.match(l)
        if sm and week:
            flush()  # 上一块异常未闭合也先落盘
            cur_secs = f"{int(sm.group(1))}-{int(sm.group(2))}"
            state = "course"
            continue
        if state == "course":
            if l.startswith("周数"):
                cur_detail = [l]
                cur_has_detail = True
                state = "detail"
            elif len(l) <= 60 and not l.startswith(("(", ")")):
                cur_course.append(l)
        elif state == "detail":
            if l.startswith("周数"):
                flush()
                cur_detail = [l]
                cur_has_detail = True
                state = "detail"
            else:
                cur_detail.append(l)  # 周数行被换行拆开的尾巴
    flush()
    if len(rows) < 2:
        return ""
    md = ["| 星期 | 节次 | 课程 | 详情 |", "| ---- | ---- | ---- | ---- |"]
    for wk, secs, name, detail in rows:
        detail = re.split(r"/教学班", detail)[0][:120]
        md.append(f"| {wk} | {secs} | {name[:40]} | {detail} |")
    return "\n".join(md)

def _week_norm(s):
    return re.sub(r"星期", "周", safe_str(s).strip())

def parse_table_courses(table_text, method):
    """确定性规则解析：直接从规整表格提取（星期,节次），零幻觉、零成本。
    返回 (course_data, covered_cells, total_text_cells, partial_weeks)：
    covered==text_cells 表示有内容的星期单元格全部解析成功；
    partial_weeks 是"存在无法解析单元格"的星期集合，允许AI在这些星期上补漏。"""
    result, covered, text_cells, partial = [], 0, 0, set()
    lines = [l for l in (table_text or "").splitlines() if l.startswith("|")]
    if len(lines) < 3:
        return result, covered, text_cells, partial

    def cells_of(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells_of(lines[0])
    sec_pat = re.compile(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*节")
    single_pat = re.compile(r"(?<![0-9])(\d{1,2})\s*节(?![0-9])")
    data = {}  # week -> set(sections)

    if method == "list":
        # | 星期 | 节次 | 课程 | 详情 |，节次列为"3-4"/"11-13"
        wi = next((i for i, h in enumerate(header) if h == "星期"), -1)
        si = next((i for i, h in enumerate(header) if h == "节次"), -1)
        if wi < 0 or si < 0:
            return result, covered, text_cells, partial
        for line in lines[2:]:
            cells = cells_of(line)
            cells += [""] * (len(header) - len(cells))
            wk = _week_norm(cells[wi])
            if wk not in WEEK_NAMES:
                continue
            text_cells += 1
            m = re.fullmatch(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})", cells[si])
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if 1 <= a <= b <= 13:
                    data.setdefault(wk, set()).update(range(a, b + 1))
                    covered += 1
                    continue
            partial.add(wk)  # 节次列解析失败 → 交给AI补
    else:
        # 框线表格：表头为星期名，单元格内"(X-Y节)"标注节次
        week_cols = {}
        for i, h in enumerate(header):
            w = _week_norm(h)
            if w in WEEK_NAMES:
                week_cols[i] = w
        if not week_cols:
            return result, covered, text_cells, partial
        sec_col = next((i for i, h in enumerate(header) if "节次" in h), -1)
        week_stat = {w: [0, 0] for w in week_cols.values()}  # [text_cells, covered]
        for line in lines[2:]:
            cells = cells_of(line)
            cells += [""] * (len(header) - len(cells))
            row_sec = None
            if sec_col >= 0:
                m = re.fullmatch(r"(\d{1,2})", cells[sec_col])
                if m:
                    row_sec = int(m.group(1))
            for ci, wk in week_cols.items():
                cell = cells[ci] if ci < len(cells) else ""
                if not cell:
                    continue
                text_cells += 1
                week_stat[wk][0] += 1
                secs = set()
                for m2 in sec_pat.finditer(cell):
                    a, b = int(m2.group(1)), int(m2.group(2))
                    if 1 <= a <= b <= 13:
                        secs.update(range(a, b + 1))
                if not secs:
                    m3 = single_pat.search(cell)
                    if m3 and 1 <= int(m3.group(1)) <= 13:
                        secs.add(int(m3.group(1)))
                if secs:
                    covered += 1
                    week_stat[wk][1] += 1
                elif row_sec and re.search(r"[\u4e00-\u9fa5A-Za-z]", cell):
                    secs = {row_sec}
                    covered += 1
                    week_stat[wk][1] += 1
                if secs:
                    data.setdefault(wk, set()).update(secs)
        partial = {w for w, (t, c) in week_stat.items() if t > c}
    result = [{"week": w, "sections": sorted(s), "courses": []} for w, s in data.items()]
    return result, covered, text_cells, partial

def _merge_course_data(rule, ai):
    """规则解析与AI结果取并集（按星期合并节次），偏召回：宁可多核对，不可漏课。"""
    merged = {}
    for src in (rule or [], ai or []):
        for e in src:
            w = e.get("week", "")
            if w not in WEEK_NAMES:
                continue
            m = merged.setdefault(w, {"week": w, "sections": set(), "courses": []})
            m["sections"].update(x for x in e.get("sections", []) if isinstance(x, int))
            for c in e.get("courses", []):
                c = safe_str(c)
                if c and c not in m["courses"]:
                    m["courses"].append(c)
    return [{"week": w, "sections": sorted(s), "courses": c} for w, m in merged.items()
            for s, c in [(m["sections"], m["courses"])] if s]

def _filter_ai_by_table(ai_data, table_text):
    """廉价抗幻觉：AI给出的星期必须在表格中真实存在（对应列/行有非空内容），否则丢弃该条。"""
    if not ai_data:
        return ai_data
    lines = [l for l in (table_text or "").splitlines() if l.startswith("|")]
    if len(lines) < 3:
        return ai_data
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    col_idx, list_week_col = {}, -1
    for i, h in enumerate(header):
        w = _week_norm(h)
        if w in WEEK_NAMES:
            col_idx[w] = i
        elif h == "星期":
            list_week_col = i
    if list_week_col >= 0:
        # 清单式：星期列里出现过的星期才算真实存在
        present = set()
        for line in lines[2:]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if list_week_col < len(cells):
                w = _week_norm(cells[list_week_col])
                if w in WEEK_NAMES:
                    present.add(w)
        if present:
            return [e for e in ai_data if e.get("week") in present]
        return ai_data
    if not col_idx:
        return ai_data
    col_has_content = {w: False for w in col_idx}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for w, i in col_idx.items():
            if i < len(cells) and cells[i]:
                col_has_content[w] = True
    return [e for e in ai_data if col_has_content.get(e.get("week", ""), True)]

def _gate_ai_by_rule(ai_data, rule_data, partial_weeks):
    """AI补漏闸门：规则已完整解析的星期，AI不得新增或删减节次（防幻觉）；
    只有规则解析存疑的星期（partial_weeks），才允许AI补充。规则整体为空时不限制。"""
    if not ai_data:
        return ai_data
    if not rule_data:
        return ai_data
    base = {e["week"]: set(e["sections"]) for e in rule_data}
    out = []
    for e in ai_data:
        w = e.get("week", "")
        secs = set(e.get("sections", []))
        keep = secs if w in partial_weeks else (secs & base.get(w, set()))
        if keep:
            out.append({"week": w, "sections": sorted(keep), "courses": list(e.get("courses", []))})
    return out

def pdf_to_images_b64(pdf_bytes, max_pages=3, zoom=2.5):
    """把课表PDF整页渲染成高清PNG，供视觉大模型直接读图（默认3页，避免课表跨页时漏掉后页课程）。"""
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
7.表格里的节次有两种写法：单元格内"(X-Y节)"，或单独一节次列写成"3-4"/"11-13"，两种都要按区间展开成全部节次。
8.只输出纯净JSON数组，禁止```、注释、markdown、任何解释文字。
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
任务：从用户原话中提取"额外不能值班的时段"，输出JSON数组，每条必须带原话证据。
输出格式：[{"slot":"周日下午","evidence":"用户原话中支持该时段的逐字片段"}]
铁律：
1. **evidence必须是用户原话里连续逐字出现的片段**（可忽略空格与标点），禁止改写、拼接、编造、同义替换。原话里找不到依据的时段，一条都不许输出。
2. slot只能是：周X上午 / 周X下午 / 周X晚上（X=一二三四五六日），共21种，禁止其他写法（不许写"周末"、"工作日"、"周天"）。
3. 只提取用户明确说出的限制，禁止联想、禁止推断、禁止帮忙补全。用户没提到的时段绝不输出。宁可少输出，绝不脑补。
4. 语义方向判定（关键，不要搞反）：
   - "有课/上课/不能/不行/没空/没时间/不值班/不方便/有事/要走" → 该时段**要输出**
   - "没课/有空/有时间/可以/能值班/能来/方便" → 该时段是**可用时段，禁止输出**
   - "周末不值班" → 周六上午/周六下午/周六晚上/周日上午/周日下午/周日晚上（6项）
   - "只在周六下午有空"（或"只有周六下午能来"）→ 输出**除周六下午外的其余20项**
   - "每天上午有课" → 周一~周日上午（7项）；"工作日"=周一~周五；"周中"=周一~周五
5. 不要输出课表已经算出的冲突时段（用户消息里已给出）。
6. 只输出纯JSON数组，禁止markdown、注释、解释。没有额外限制就输出 []。
✅示例：用户说"我周二下午有课，另外周末不值班"
→ [{"slot":"周二下午","evidence":"周二下午有课"},{"slot":"周六上午","evidence":"周末不值班"},{"slot":"周六下午","evidence":"周末不值班"},{"slot":"周六晚上","evidence":"周末不值班"},{"slot":"周日上午","evidence":"周末不值班"},{"slot":"周日下午","evidence":"周末不值班"},{"slot":"周日晚上","evidence":"周末不值班"}]
✅示例：用户说"周二下午没课"
→ []
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
    return arr if isinstance(arr, list) else []

# ========= 个性化需求【幻觉压制层】：原话证据回溯 + 语义方向判定 + 确定性规则底稿 =========
_BUSY_MARKERS = ("有课", "上课", "不能", "不行", "不可以", "没空", "没有空", "没时间",
                 "不值班", "不能值班", "不方便", "走不开", "冲突", "忙", "有事")
_AVAIL_MARKERS = ("没课", "没有课", "不上课", "有空", "有时间", "可以", "能值班", "能来", "方便", "没事")

def _norm_for_match(s):
    """用于"原话证据逐字回溯"的归一化：只留文字与数字，去掉空白标点。"""
    return re.sub(r"[\s，,。.、；;：:！!？?“”\"'‘’（）()【】\[\]…—－\-~·]+", "", safe_str(s))

def _days_in_clause(c):
    days = []
    for d in WEEK_NAMES:
        if d in c:
            days.append(d)
    if "周末" in c or "双休" in c:
        days += ["周六", "周日"]
    if "每天" in c or "天天" in c:
        days += list(WEEK_NAMES)
    if "工作日" in c or "周中" in c or "平时" in c:
        days += WEEK_NAMES[:5]
    seen, out = set(), []
    for d in days:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out

def _shifts_in_clause(c):
    shifts = []
    if "上午" in c or "早上" in c or "早晨" in c:
        shifts.append("上午")
    if "下午" in c or "中午" in c:
        shifts.append("下午")
    if "晚上" in c or "傍晚" in c or "夜间" in c:
        shifts.append("晚上")
    seen, out = set(), []
    for s in shifts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _only_available_set(c):
    """"只在周六下午有空" → 返回可用时段集合（其余全部禁排）；不是这种句式返回 None。"""
    if not ("只" in c or "仅" in c):
        return None
    if not any(m in c for m in ("有空", "可以", "能", "方便", "有时间", "没课", "不上课")):
        return None
    days = _days_in_clause(c)
    if not days:
        return None
    shifts = _shifts_in_clause(c) or list(("上午", "下午", "晚上"))
    return {f"{d}{s}" for d in days for s in shifts}

def parse_personal_rulebased(text):
    """确定性规则底稿：常见口语说法直接翻译为禁排时段，不依赖AI，永不幻觉。"""
    slots = set()
    t = safe_str(text)
    if not t:
        return slots
    # 先按标点+转折词切分，避免"周一上午没课但周二下午不能来"被当成一个整体
    parts = re.split(r"[，,。.、；;！!？?\n]+|但是|但|不过|可是|然而", t)
    for c0 in parts:
        c = _norm_for_match(c0)
        if not c:
            continue
        days = _days_in_clause(c)
        if not days:
            continue
        shifts = _shifts_in_clause(c) or ["上午", "下午", "晚上"]
        allowed = _only_available_set(c)
        if allowed is not None:  # "只在X有空" → 补集
            slots |= {s for s in ALL_SLOTS if s not in allowed}
            continue
        if any(m in c for m in _BUSY_MARKERS):
            slots |= {f"{d}{s}" for d in days for s in shifts}
        # 只出现"有空/没课"等可用词、没有忙词 → 不加入
    return slots

def _validate_personal_slots(raw, text):
    """逐条校验AI输出：证据必须能在原话中逐字回溯，且星期/时段对得上、语义方向正确。"""
    valid = set()
    tnorm = _norm_for_match(text)
    if not tnorm:
        return valid
    for item in raw or []:
        if isinstance(item, dict):
            slot = _norm_for_match(item.get("slot", ""))
            ev = _norm_for_match(item.get("evidence", ""))
        else:
            slot, ev = _norm_for_match(item), ""
        slot = re.sub(r"星[期其]", "周", slot)
        slot = slot.replace("周天", "周日")
        if slot not in ALL_SLOTS:
            continue                      # ① 非法时段写法直接丢
        if not ev or ev not in tnorm:
            continue                      # ② 证据无法在原话中逐字回溯 → 判定为幻觉，丢
        allowed = _only_available_set(ev)
        if allowed is not None:           # ③ "只在X有空"的补集语义
            if slot not in allowed:
                valid.add(slot)
            continue
        if slot[:2] not in _days_in_clause(ev):
            continue                      # ④ 证据里没提这个星期 → 丢
        ev_shifts = _shifts_in_clause(ev)
        if ev_shifts and slot[2:] not in ev_shifts:
            continue                      # ⑤ 证据写了具体时段但对不上 → 丢（没写时段视为早中晚全含）
        if any(m in ev for m in _BUSY_MARKERS):
            valid.add(slot)               # ⑥ 语义方向为"忙/不能" → 采纳
        elif any(m in ev for m in _AVAIL_MARKERS):
            continue                      # ⑦ 语义方向是"有空/没课" → 拒绝（AI搞反了）
    return valid

def extract_personal_slots(personal_need, course_conflict):
    """个性化需求统一入口：规则底稿 ∪ 经证据校验的AI结果，返回 (slots, 调试信息)。"""
    rule_slots = parse_personal_rulebased(personal_need)
    ai_raw, ai_slots, ai_err = [], set(), ""
    prompt_personal = f"""
课表已经算出的不能值班时段（不要再重复输出）：{json.dumps(course_conflict, ensure_ascii=False)}
用户个性化需求原话：{personal_need}
请按系统要求输出JSON数组，每条含 slot 与 evidence（evidence必须逐字来自上面的原话）。
"""
    try:
        ai_raw = _call_personal_ai(prompt_personal)
        ai_slots = _validate_personal_slots(ai_raw, personal_need)
    except Exception as e:
        ai_err = str(e)[:80]
    final = sorted((rule_slots | ai_slots) - set(course_conflict), key=slot_sort_key)
    info = {
        "rule_slots": sorted(rule_slots),
        "ai_raw": ai_raw,
        "ai_valid_slots": sorted(ai_slots),
        "ai_err": ai_err,
        "dropped_by_validation": len(ai_raw) - len([1 for x in ai_raw if isinstance(x, dict) and _norm_for_match(x.get("slot", "")) in ai_slots]),
    }
    return final, info

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

# 班次时间区间(分钟)，与 SHIFT_LIST 的三个时段一一对应
SHIFT_TIME = {0: (9 * 60, 13 * 60), 1: (13 * 60, 17 * 60), 2: (17 * 60, 20 * 60)}

_SHIFT_ORDER = {"上午": 0, "下午": 1, "晚上": 2}

def slot_sort_key(s):
    """时段按「周几→早中晚」的常识顺序排序（直接sorted会按Unicode排成周一、周三、周二…）。"""
    s = safe_str(s)
    w = WEEK_NAMES.index(s[:2]) if s[:2] in WEEK_NAMES else 9
    return (w, _SHIFT_ORDER.get(s[2:], 9))

def calc_course_conflict(ai_course_data):
    """把节次映射为班次冲突：只要课程时间与班次时间有重叠，该班次就算不可用
    （例：第9节16:40-17:25 与晚班17:00-20:00重叠 → 下午、晚上都冲突）。"""
    conflict = set()
    for item in ai_course_data:
        w = item["week"]
        for s in item["sections"]:
            if s not in SECTION_TIME:
                continue
            c_start, c_end = SECTION_TIME[s]
            for si, (sh_start, sh_end) in SHIFT_TIME.items():
                if c_start < sh_end and c_end > sh_start:  # 区间相交
                    conflict.add(f"{w}{SHIFT_PERIOD[si]}")
    return sorted(conflict, key=slot_sort_key)

def _in_script_thread():
    """是否运行在 Streamlit 脚本主线程（批量识别在工作线程里，st 调用无效）。"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def _can_render_debug():
    """仅管理员会话 + 脚本主线程才内联渲染DEBUG（避免把内部JSON暴露给成员，也避免工作日线程无效调用）。"""
    try:
        return _in_script_thread() and bool(st.session_state.get("login"))
    except Exception:
        return False

def fast_ai_extract(pdf_bytes:bytes, personal_need:str="", debug_sink:dict=None, allow_ai:bool=True):
    """智能体主链路：AI 全程参与识别；规则解析作为确定性底稿与防幻觉闸门。
    路线：① 表格/清单式 → 规则解析 + qwen 文本模型（AI补漏、规则兜底）
          ② 网格/图片型 → qwen-vl-max 视觉直接读图
          ③ 全部失败 → 文本模型兜底
    debug_sink: 传入 dict 时把识别详情写进去（供工作线程调用方在主线程渲染）。
    allow_ai: 批量提速开关。False 时，若该文件规则解析已 100% 覆盖（结果与AI完全等价），
              跳过课表文本模型调用；规则未全覆盖时仍会自动启用 AI（否则不准）。
              ⚠️ 个性化需求（doubao）不受此开关影响，有需求一定调用。
    """
    t0 = time.time()
    table_text, method = get_pdf_table_text(pdf_bytes)
    ai_course_data = []
    route = ""
    ai_used = True
    dbg = debug_sink if isinstance(debug_sink, dict) else {}
    rule_data, rule_covered, rule_text_cells, rule_partial = [], 0, 0, set()
    ai_agree = None
    raw_ai_count = 0
    prompt_course = f"""
下面是课表原文（Markdown表格）。表头行标注了星期几（顺序可能从周日开始，以表头实际标注为准）；有一列是节次编号；节次可能写在单元格内"(X-Y节)"，也可能写成"3-4"这种区间，两种都表示该课占据第X节到第Y节。请提取所有有课的星期和节次，以单元格的星期列位置和节次信息为准：
{table_text[:12000]}
"""
    # ---- 路线①：表格/清单式 → 规则解析 + AI 文本模型 双通道（AI始终参与） ----
    if method in ("table", "list") and len(table_text.strip()) >= 10:
        try:
            rule_data, rule_covered, rule_text_cells, rule_partial = parse_table_courses(table_text, method)
        except Exception:
            rule_data, rule_covered, rule_text_cells, rule_partial = [], 0, 0, set()
        ai_part, ai_err, ai_skipped = [], "", False
        # 提速：批量且规则已100%覆盖 → 跳过课表AI调用（此时AI加不了任何信息，结果等价）
        rule_complete = bool(rule_data) and rule_text_cells >= 2 and rule_covered >= rule_text_cells
        if (not allow_ai) and rule_complete:
            ai_skipped = True
            ai_used = False
        else:
            try:
                ai_part = _clean_course_data(_call_course_ai(prompt_course))
            except Exception as e:
                ai_err = str(e)[:60]
        raw_ai_count = len(ai_part)  # 供DEBUG参考：AI原始条数（过滤前）
        ai_part = _filter_ai_by_table(ai_part, table_text)          # 抗幻觉①：AI说的星期必须在表格真实存在
        ai_part = _gate_ai_by_rule(ai_part, rule_data, rule_partial)  # 抗幻觉②：规则已完整覆盖的星期不许AI新增
        ai_course_data = _merge_course_data(rule_data, ai_part)
        if ai_course_data:
            if ai_skipped:
                route = f"规则路线：{method}表格直接解析（批量AI已关闭，规则100%覆盖，零调用）"
            elif rule_data:
                # AI 与规则是否完全一致（用于DEBUG核对）
                rule_map = {e["week"]: set(e["sections"]) for e in rule_data}
                ai_map = {e["week"]: set(e["sections"]) for e in ai_part}
                ai_agree = (rule_map == ai_map)
                route = (f"双通道路线：规则解析(确定性底稿) ∪ {COURSE_TEXT_MODEL}(AI补漏防幻觉)"
                         f"｜AI{'一致' if ai_agree else '有补充/差异'}")
            else:
                route = f"文本路线：{method}表格 + {COURSE_TEXT_MODEL}（AI主导，规则无可解析项）"
            if (not allow_ai) and not ai_skipped:
                route += "｜该文件规则未完全覆盖，已自动启用AI保准确"
        elif ai_err:
            route = f"规则路线：{method}表格直接解析（AI异常已忽略：{ai_err}）"
    # ---- 路线②：网格近似/图片型PDF/上一路线失败 → 视觉大模型直接读图 ----
    if not ai_course_data:
        try:
            imgs = pdf_to_images_b64(pdf_bytes)
            ai_course_data = _clean_course_data(_call_course_ai_vl(imgs))
            route = f"视觉路线：{COURSE_VL_MODEL} 直接读图"
        except Exception as e_vl:
            # ---- 路线③：视觉路线也失败 → 文本路线最后兜底 ----
            if method in ("grid", "raw", "list") and len(table_text.strip()) >= 10:
                try:
                    ai_course_data = _clean_course_data(_call_course_ai(prompt_course))
                    route = f"文本兜底路线：{method}表格 + {COURSE_TEXT_MODEL}"
                except Exception:
                    ai_course_data = []
            if not ai_course_data:
                dbg.update({"route": route or "全部路线失败", "method": method, "failed": True,
                            "ai_used": ai_used, "rule_covered": rule_covered,
                            "rule_text_cells": rule_text_cells, "elapsed_ms": int((time.time() - t0) * 1000)})
                return f"课表识别失败:{str(e_vl)[:60]}"
    try:
        course_conflict = calc_course_conflict(ai_course_data)
    except Exception as e:
        dbg.update({"route": route, "method": method, "failed": True, "ai_used": ai_used,
                    "elapsed_ms": int((time.time() - t0) * 1000)})
        return f"课表识别失败:{str(e)[:60]}"
    # ---- 识别详情：收集到 debug_sink（批量线程）或在线程内直接渲染 ----
    dbg.update({
        "route": route,
        "method": method,
        "failed": False,
        "ai_used": ai_used,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "rule_weeks": len(rule_data),
        "rule_covered": rule_covered,
        "rule_text_cells": rule_text_cells,
        "ai_agree": ai_agree,
        "ai_raw_count": raw_ai_count,
        "course_data": ai_course_data,
        "table_text": table_text[:3000] if table_text and method in ("table", "grid", "list") else "",
    })
    if debug_sink is None and _can_render_debug():
        try:
            with st.expander(f"🔍DEBUG 识别详情｜路线：{route}｜个性化模型：doubao2.1‑seed‑pro"):
                st.caption(f"规则解析（确定性）：{len(rule_data)} 个星期，节次覆盖 {rule_covered}/{rule_text_cells}"
                           + (f"，AI与规则一致：{'是' if ai_agree else '否'}" if ai_agree is not None else ""))
                st.code(json.dumps(ai_course_data, ensure_ascii=False, indent=2), language="json")
                if dbg["table_text"]:
                    with st.expander("📄 查看喂给AI的课表表格（核对是否错位）"):
                        st.text(dbg["table_text"])
        except Exception:
            pass
    # ---- 个性化需求：doubao 识别 + 原话证据回溯校验（强幻觉压制） ----
    personal_slots, personal_info = [], {}
    if personal_need and personal_need.strip() not in ("", "无"):
        personal_slots, personal_info = extract_personal_slots(personal_need, course_conflict)
        dbg["personal"] = personal_info
        dbg["personal_slots"] = personal_slots
        if debug_sink is None and _can_render_debug():
            try:
                with st.expander("🔍DEBUG 个性化需求解析（doubao + 原话证据校验）"):
                    st.caption(f"规则底稿识别：{'、'.join(personal_info.get('rule_slots') or []) or '无'}")
                    st.caption(f"AI原始输出（{len(personal_info.get('ai_raw') or [])} 条）")
                    st.code(json.dumps(personal_info.get("ai_raw") or [], ensure_ascii=False, indent=2), language="json")
                    st.caption(f"AI通过证据校验：{'、'.join(personal_info.get('ai_valid_slots') or []) or '无'}"
                               + (f"｜AI异常：{personal_info.get('ai_err')}" if personal_info.get("ai_err") else ""))
                    st.caption("说明：AI输出的每条时段都必须能在你的原话里找到逐字依据，找不到依据的会被自动丢弃。")
            except Exception:
                pass
    total = sorted(set(course_conflict + personal_slots), key=slot_sort_key)
    if not total:
        dbg["failed"] = True
        return "识别为空，请手动填写"
    return "、".join(total)

# ========== 判断某人在某时段是否有空 ==========
def is_available(member, week_idx, shift_idx):
    class_conflict = safe_str(member.get("class_conflict"))
    week_name = WEEK_NAMES[week_idx]
    period = SHIFT_PERIOD[shift_idx]
    conflict_str = f"{week_name}{period}"
    if class_conflict and class_conflict != "无" and conflict_str in class_conflict:
        return False
    return True

# ========== 智能排班（稀缺度优先 + 均衡 + 绝不违背课表） ==========
def _available_slot_keys(member):
    """该成员本周可以值班的 (week_idx, shift_idx) 集合。"""
    return {(w, s) for w in range(len(WEEK_NAMES)) for s in range(len(SHIFT_LIST))
            if is_available(member, w, s)}

def _slot_key(slot):
    return (slot["week_idx"], slot["shift_idx"])

def smart_schedule(member_list):
    """排班策略：
    ① 稀缺度优先：可用人数越少的班次越先排，避免"好排的时段先把稀缺人员抢光、难排的时段最后没人"；
    ② 越受限的人越先排：候选按自身可用时段数升序（可选时段少的人优先锁定，把自由的人留到后面）；
    ③ 学员分多轮放开上限（1→2→3 班）：先保证人人有一次班，人不够时再用"确实有空"的人补位；
    ④ 职委按当前班次数升序轮转，天然均衡；
    ⑤ 绝不违背课表：找不到有空的人时，该班次留空并标记「待人工指派」，不再强制安排有课的人。
    """
    stu_pool = [m for m in member_list if safe_str(m.get("role")) == "学员"]
    wei_pool = [m for m in member_list if safe_str(m.get("role")) == "职委"]
    st.info(f"📊 人员统计：学员 {len(stu_pool)} 人，职委 {len(wei_pool)} 人")
    if len(wei_pool) == 0:
        st.error("⚠️ 警告：人员库里没有职委！请去「人员库管理」把需要当职委的人身份改成「职委」")
    if not stu_pool:
        st.error("⚠️ 人员库里没有学员，无法排班。请先上传课表录入人员。")
        return []

    all_slots = []
    for w_idx, w in enumerate(WEEK_NAMES):
        for s_idx, s in enumerate(SHIFT_LIST):
            all_slots.append({"week_idx": w_idx, "shift_idx": s_idx, "week": w, "shift": s["shift_name"]})

    tie = {id(m): random.random() for m in stu_pool + wei_pool}   # 同分时随机打破平局（每次排班不完全相同）

    # ---------- 学员 ----------
    stu_avail = {id(m): _available_slot_keys(m) for m in stu_pool}
    stu_cnt = {id(m): 0 for m in stu_pool}
    # 每个班次有多少学员可选 → 稀缺度（人数越少越难排，必须最先满足）
    cand_cnt = {_slot_key(s): sum(1 for m in stu_pool if _slot_key(s) in stu_avail[id(m)]) for s in all_slots}
    order = sorted(all_slots, key=lambda s: (cand_cnt[_slot_key(s)], s["week_idx"], s["shift_idx"]))
    assign = {_slot_key(s): [] for s in all_slots}
    for cap in (1, 2, 3):        # 第一轮每人最多1班；仍有空缺再用有空的人补第2、3班（不违背课表）
        for slot in order:
            k = _slot_key(slot)
            need = SHIFT_LIST[slot["shift_idx"]]["need_stu"]
            if len(assign[k]) >= need:
                continue
            cands = [m for m in stu_pool
                     if k in stu_avail[id(m)] and stu_cnt[id(m)] < cap
                     and not any(id(x) == id(m) for x in assign[k])]
            cands.sort(key=lambda m: (len(stu_avail[id(m)]), stu_cnt[id(m)], tie[id(m)]))
            for m in cands:
                if len(assign[k]) >= need:
                    break
                assign[k].append(m)
                stu_cnt[id(m)] += 1

    # ---------- 职委 ----------
    wei_avail = {id(m): _available_slot_keys(m) for m in wei_pool}
    wei_cnt = {id(m): 0 for m in wei_pool}
    wei_cand_cnt = {_slot_key(s): sum(1 for m in wei_pool if _slot_key(s) in wei_avail[id(m)]) for s in all_slots}
    wei_order = sorted(all_slots, key=lambda s: (wei_cand_cnt[_slot_key(s)], s["week_idx"], s["shift_idx"]))
    wei_assign = {_slot_key(s): [] for s in all_slots}
    for slot in wei_order:
        k = _slot_key(slot)
        need = SHIFT_LIST[slot["shift_idx"]]["need_wei"]
        cands = [m for m in wei_pool if k in wei_avail[id(m)]]
        cands.sort(key=lambda m: (wei_cnt[id(m)], len(wei_avail[id(m)]), tie[id(m)]))   # 班次少的先上 → 均衡
        for m in cands[:need]:
            wei_assign[k].append(m)
            wei_cnt[id(m)] += 1

    # ---------- 组装 + 缺口诊断 ----------
    schedule, pending = [], []
    for slot in all_slots:
        k = _slot_key(slot)
        need_stu = SHIFT_LIST[slot["shift_idx"]]["need_stu"]
        need_wei = SHIFT_LIST[slot["shift_idx"]]["need_wei"]
        stu_list, wei_list = assign[k], wei_assign[k]
        gap_stu = max(0, need_stu - len(stu_list))
        gap_wei = max(0, need_wei - len(wei_list))
        reasons = []
        if gap_stu:
            avail_names = [m for m in stu_pool if k in stu_avail[id(m)]]
            if not avail_names:
                reasons.append("该时段全部学员都有课")
            else:
                reasons.append(f"有空学员的排班已达上限（缺{gap_stu}人）")
        if gap_wei:
            avail_wei = [m for m in wei_pool if k in wei_avail[id(m)]]
            reasons.append("该时段无职委有空" if not avail_wei else f"职委人数不足（缺{gap_wei}人）")
        note = ("待人工指派：" + "；".join(reasons)) if reasons else ""
        if reasons:
            pending.append({"班次": f"{slot['week']}{SHIFT_PERIOD[slot['shift_idx']]}",
                            "缺学员": gap_stu, "缺职委": gap_wei, "原因": "；".join(reasons)})
        schedule.append({
            "week_idx": slot["week_idx"], "shift_idx": slot["shift_idx"],
            "week": slot["week"], "shift": slot["shift"],
            "学员": stu_list, "职委": wei_list,
            "缺口": gap_stu + gap_wei, "备注": note,
        })

    # ---------- 诊断输出 ----------
    rested = [safe_str(m.get("name")) for m in stu_pool if stu_cnt[id(m)] == 0]
    if rested:
        st.info(f"ℹ️ 本周共 {len(rested)} 名学员轮空：{'、'.join(rested)}")
    if pending:
        st.error(f"⚠️ {len(pending)} 个班次无法在「不违背课表」的前提下排满，已留空标记「待人工指派」"
                 f"（未自动安排有课人员）。请手动指派后再导出。")
        st.dataframe(pd.DataFrame(pending), use_container_width=True)
    if stu_pool:
        c = sorted(stu_cnt.values())
        st.caption(f"学员班次分布：共 {sum(c)} 个学员岗，最多 {max(c)} 班 / 最少 {min(c)} 班")
    if wei_pool:
        wc = sorted(wei_cnt.values())
        st.success(f"✅ 职委排班完成：共 {len(wei_pool)} 名职委，平均每人排 {sum(wc)/len(wei_pool):.1f} 班"
                   f"（最多 {max(wc)} / 最少 {min(wc)} 班）")
    tight = sorted([s for s in all_slots if cand_cnt[_slot_key(s)] <= 2],
                   key=lambda s: (cand_cnt[_slot_key(s)], s["week_idx"], s["shift_idx"]))[:5]
    if tight:
        st.caption("🔍 最紧张的时段（可用学员≤2人）：" + "、".join(
            f"{s['week']}{SHIFT_PERIOD[s['shift_idx']]}({cand_cnt[_slot_key(s)]}人可选)" for s in tight))
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
def fill_user_template_pdf(template_bytes, schedule_data, offset_x:int=0, offset_y:int=-30, cell_w:int=82, cell_h:int=115):
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

# ========== 页面壁纸（自定义操作页背景图） ==========
WALLPAPER_CFG_FILE = "wallpaper_config.json"
WALLPAPER_IMG_BASE = "wallpaper"

def load_wallpaper_config():
    cfg = _safe_load_json(WALLPAPER_CFG_FILE, {})
    return cfg if isinstance(cfg, dict) else {}

def save_wallpaper_config(cfg):
    _atomic_write_json(WALLPAPER_CFG_FILE, cfg)

def _compress_wallpaper(data):
    """有 Pillow 就缩放+转JPEG（避免几MB图片把每页CSS撑大）；没有就原样使用。"""
    try:
        from PIL import Image
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.width > 1920:
            img = img.resize((1920, max(1, int(img.height * 1920 / img.width))), Image.LANCZOS)
        out = BytesIO()
        img.save(out, format="JPEG", quality=82)
        return out.getvalue(), "jpg"
    except Exception:
        is_jpg = data[:3] == b"\xff\xd8\xff"
        return data, ("jpg" if is_jpg else "png")

def save_wallpaper_image(data):
    data, ext = _compress_wallpaper(data)
    path = f"{WALLPAPER_IMG_BASE}.{ext}"
    for old in (f"{WALLPAPER_IMG_BASE}.png", f"{WALLPAPER_IMG_BASE}.jpg",
                f"{WALLPAPER_IMG_BASE}.jpeg", f"{WALLPAPER_IMG_BASE}.webp"):
        if old != path and os.path.exists(old):
            try:
                os.remove(old)
            except Exception:
                pass
    with open(path, "wb") as f:
        f.write(data)
    return path

def clear_wallpaper():
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = f"{WALLPAPER_IMG_BASE}.{ext}"
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    save_wallpaper_config({"enabled": False, "alpha": 0.82, "blur": 0, "path": ""})

def apply_wallpaper():
    """全局注入背景图（登录页 / 管理员后台 / 成员自助页都生效）。
    滑块值优先取 session_state，做到"边调边看"，保存只是持久化。"""
    cfg = load_wallpaper_config()
    enabled = bool(st.session_state.get("wallpaper_enabled", cfg.get("enabled", False)))
    if not enabled:
        return
    path = safe_str(cfg.get("path"))
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except Exception:
        return
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    alpha = min(1.0, max(0.0, float(st.session_state.get("wallpaper_alpha", cfg.get("alpha", 0.82)))))
    blur = int(st.session_state.get("wallpaper_blur", cfg.get("blur", 0)))
    st.markdown(f"""
<style>
[data-testid="stAppViewContainer"], .stApp {{
    background-image: url("data:{mime};base64,{b64}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
.main .block-container, [data-testid="stAppViewBlockContainer"] {{
    background: rgba(255,255,255,{alpha:.2f});
    border-radius: 18px;
    backdrop-filter: blur({blur}px);
    -webkit-backdrop-filter: blur({blur}px);
    padding: 2rem 2.4rem;
}}
</style>
""", unsafe_allow_html=True)

# ========== 页面入口 ==========
st.set_page_config(page_title="值班排班系统", layout="wide")
apply_wallpaper()
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
if "temp_pdf_by_name" not in st.session_state:
    st.session_state.temp_pdf_by_name = {}
if "member_pending" not in st.session_state:
    st.session_state.member_pending = None
if "ai_failed_names" not in st.session_state:
    st.session_state.ai_failed_names = []
if "temp_debug" not in st.session_state:
    st.session_state.temp_debug = {}
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
                    st.error("账号或密码错误")
        st.caption("首次使用请用管理员提供的初始账号登录，忘记密码请联系超级管理员或使用「管理员管理」中的重置功能。")
    else:
        st.title(f"管理员后台｜排班管理（当前登录：{st.session_state.current_admin}）")
        try:
            _cur = next((a for a in load_admin()["admins"] if a.get("username") == st.session_state.current_admin), None)
            if _cur and _cur.get("password") == DEFAULT_ADMIN_PWD:
                st.warning("⚠️ 当前账号仍在使用默认密码，建议立即到「管理员管理」修改，避免他人直接登录后台。")
        except Exception:
            pass
        st.caption("智能体链路：PDF课表→四级结构化提取（清单式/框线表格/坐标网格/纯文本）→ qwen‑plus 与确定性规则解析双通道互补（AI补漏+闸门防幻觉）；图片型自动切 qwen‑vl‑max 读图；个性化需求由 doubao2.1‑seed‑pro 解析并逐条回溯原话证据")
        if is_super_admin(st.session_state.current_admin):
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["批量上传课表","人员库管理","自动排班","导出PDF","分享链接","管理员管理"])
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["批量上传课表","人员库管理","自动排班","导出PDF","分享链接"])
            tab6 = None
        with tab1:
            st.subheader("批量上传PDF课表（文件名格式：身份&姓名&手机号‑课表.pdf）")
            if st.session_state.get("last_batch_cost") is not None:
                _s = st.session_state.get("last_batch_stat") or {}
                st.success(f"⏱️ 上一批 {_s.get('total', 0)} 份课表用时 {st.session_state.last_batch_cost} 秒"
                           f"（平均 {(st.session_state.last_batch_cost / _s['total']):.2f} 秒/份，"
                           f"失败 {_s.get('failed', 0)} 份"
                           + (f"，规则平均覆盖 {_s['avg_cover']*100:.0f}%" if _s.get("avg_cover") is not None else "")
                           + "）")
            if st.session_state.ai_failed_names:
                st.warning("⚠️ 以下人员课表识别异常，请手动核对/补填不能值班时段：" + "、".join(st.session_state.ai_failed_names))
            up_files = st.file_uploader("选择课表PDF", type="pdf", accept_multiple_files=True)
            st.caption("💡 若上传时出现「AxiosError: Request failed with status code 500」：这是 Streamlit 云端上传通道的偶发故障（与课表内容无关），刷新页面后重新选择文件即可；建议一次少选几个、分批上传更稳。")
            if up_files and st.button("📂 第一步：缓存PDF二进制，解析姓名身份电话"):
                with st.spinner("读取PDF文件..."):
                    pairs = []          # [(成员dict, pdf字节)] —— 成对保存，杜绝"人与文件串位"
                    bad_files = []      # 解析不出姓名的文件，明确提示管理员
                    for f in up_files:
                        b = f.read()
                        try:
                            doc_tmp = fitz.open(stream=b, filetype="pdf")
                            txt_tmp = doc_tmp[0].get_text() if len(doc_tmp) > 0 else ""
                            doc_tmp.close()
                        except Exception:
                            txt_tmp = ""
                        name = extract_name_from_text(txt_tmp)
                        role_from_file, phone_from_file = parse_filename_phone_and_role(f.name)
                        if not name:
                            bad_files.append(f.name)
                            continue
                        pairs.append(({
                            "name": name,
                            "phone": phone_from_file,
                            "role": role_from_file,
                            "class_conflict": "",
                            "personal_need": ""
                        }, b))
                    seen = set()
                    unique, unique_pdf = [], []
                    for item, b in pairs:
                        if item["name"] in seen:
                            continue
                        seen.add(item["name"])
                        unique.append(item)
                        unique_pdf.append(b)
                    st.session_state.temp_new_member = unique
                    st.session_state.temp_pdf_bytes = unique_pdf                # 与 unique 严格同序、同长
                    st.session_state.temp_pdf_by_name = {item["name"]: b for item, b in zip(unique, unique_pdf)}  # 按姓名绑定，删行/调序也不会串位
                msg = f"已缓存 {len(unique)} 份PDF，填写个性化需求后执行AI识别"
                if bad_files:
                    msg += f"；⚠️ 有 {len(bad_files)} 个文件未能从PDF解析出姓名，已跳过，请核对后单独上传：{'、'.join(bad_files[:5])}"
                st.success(msg)
                st.rerun()
            if len(st.session_state.temp_new_member) > 0:
                st.divider()
                st.subheader("📋 预览表格（先填个性化需求，再点AI识别，AI会综合课表+需求一起算）")
                allow_ai = st.toggle(
                    "启用AI参与课表识别（推荐开启）",
                    value=True, key="allow_ai_toggle",
                    help="开启：每份课表都调用 qwen 文本模型，AI 与规则解析双通道互补，最稳。\n"
                         "关闭：规则已100%覆盖的课表直接走规则解析、跳过AI调用，批量60份可从数十秒降到几秒；"
                         "规则覆盖不全的文件仍会自动启用AI保准确。个性化需求（doubao）不受此开关影响。")
                st.caption("当前模式：" + ("🤖 AI全程参与（双通道）" if allow_ai else "⚡ 极速模式（规则100%覆盖时跳过课表AI调用）"))
                df_temp = pd.DataFrame(st.session_state.temp_new_member)
                temp_config = {
                    "role": st.column_config.SelectboxColumn("身份", options=["学员","职委"], required=True),
                    "class_conflict": st.column_config.TextColumn("AI识别不能值班时段（课表+需求综合）"),
                    "personal_need": st.column_config.TextColumn("个性化需求（手动填，AI会自动识别各种说法）")
                }
                edited_temp = st.data_editor(df_temp, use_container_width=True, column_config=temp_config, num_rows="dynamic",
                                             key=f"temp_editor_{len(st.session_state.temp_new_member)}_{abs(hash(tuple(m.get('name','') for m in st.session_state.temp_new_member))) % 100000}")
                if st.session_state.get("temp_debug"):
                    with st.expander("🔍 识别详情（逐人核对：走了哪条路线、AI与规则是否一致）"):
                        for _nm, _d in st.session_state.temp_debug.items():
                            if not _d:
                                continue
                            st.markdown(f"**{_nm}**｜{_d.get('route', '')}")
                            st.caption(f"规则解析 {_d.get('rule_weeks')} 个星期，节次覆盖 {_d.get('rule_covered')}/{_d.get('rule_text_cells')}"
                                       + (f"｜AI与规则一致：{'是' if _d.get('ai_agree') else '否'}" if _d.get('ai_agree') is not None else ""))
                            st.code(json.dumps(_d.get("course_data") or [], ensure_ascii=False, indent=2), language="json")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("🤖 第二步：AI综合识别（课表+个性化需求）"):
                        current_rows = edited_temp.to_dict("records")
                        need_ai_idx = []
                        need_pdf_bin = []
                        need_ai_personal = []
                        pdf_by_name = st.session_state.get("temp_pdf_by_name", {})
                        no_pdf_names = []
                        for i, row in enumerate(current_rows):
                            if not safe_str(row.get("class_conflict")):
                                nm = safe_str(row.get("name"))
                                b = pdf_by_name.get(nm, b"")
                                if not b:
                                    no_pdf_names.append(nm or f"第{i+1}行")
                                need_ai_idx.append(i)
                                need_pdf_bin.append(b)
                                need_ai_personal.append(safe_str(row.get("personal_need")))
                        if no_pdf_names:
                            st.warning("以下人员没有对应的课表PDF（可能是手动新增的行），这些行无法识别课表，请单独上传：" + "、".join(no_pdf_names))
                        if need_pdf_bin:
                            _t_start = time.time()
                            with st.spinner(f"处理 {len(need_pdf_bin)} 份课表（{'AI双通道' if allow_ai else '极速模式'}）..."):
                                bar = st.progress(0)
                                dbg_map = {}
                                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                    futures = {}
                                    for idx, pdf_bin, p in zip(need_ai_idx, need_pdf_bin, need_ai_personal):
                                        sink = {}
                                        dbg_map[idx] = sink
                                        futures[executor.submit(fast_ai_extract, pdf_bin, p, sink, allow_ai)] = idx
                                    done = 0
                                    for fut in as_completed(futures):
                                        idx = futures[fut]
                                        try:
                                            current_rows[idx]["class_conflict"] = fut.result()
                                        except Exception as e:
                                            current_rows[idx]["class_conflict"] = f"识别失败: {str(e)[:60]}"
                                            dbg_map.setdefault(idx, {}).update({"failed": True, "route": f"线程异常:{str(e)[:40]}"})
                                        done += 1
                                        bar.progress(done/len(need_pdf_bin))
                            _cost = time.time() - _t_start
                            for row in current_rows:
                                row["personal_need"] = safe_str(row.get("personal_need"))
                                row["class_conflict"] = safe_str(row.get("class_conflict"))
                                row["name"] = safe_str(row.get("name"))
                                row["phone"] = safe_str(row.get("phone"))
                                row["role"] = safe_str(row.get("role")) if safe_str(row.get("role")) else "学员"
                            # 记录识别异常人员，rerun后在页面上方提示人工核对
                            st.session_state.ai_failed_names = [safe_str(r.get("name")) for r in current_rows if is_recognize_failed(r.get("class_conflict"))]
                            st.session_state.temp_debug = {safe_str(current_rows[i].get("name")): dbg_map.get(i, {}) for i in need_ai_idx}
                            st.session_state.temp_new_member = current_rows
                            # ---- 写入识别质量日志（主线程统一写，避免多线程争抢文件） ----
                            try:
                                append_recognition_log([
                                    build_log_record(current_rows[i].get("name"), dbg_map.get(i, {}),
                                                     current_rows[i].get("personal_need", ""))
                                    for i in need_ai_idx
                                ])
                                _sum = summarize_recognition_log(load_recognition_log()[-len(need_ai_idx):])
                                st.session_state.last_batch_cost = round(_cost, 1)
                                st.session_state.last_batch_stat = _sum
                            except Exception:
                                pass
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
                            st.session_state.temp_pdf_by_name = {}
                            st.session_state.temp_debug = {}
                            st.success(f"成功录入 {len(new_add)} 人！")
                            st.rerun()
                with col_c:
                    if st.button("❌ 放弃本批"):
                        st.session_state.temp_new_member = []
                        st.session_state.temp_pdf_bytes = []
                        st.session_state.temp_pdf_by_name = {}
                        st.session_state.ai_failed_names = []
                        st.session_state.temp_debug = {}
                        st.rerun()
            # ---------- 识别质量看板（可观测：哪类模板容易出错） ----------
            st.divider()
            _log = load_recognition_log()
            with st.expander(f"📈 识别质量日志（已累计 {len(_log)} 条，保留最近 {LOG_MAX} 条）"):
                if not _log:
                    st.caption("暂无线索。上传并识别课表后，这里会自动记录每份课表的识别路线、规则覆盖率、AI一致性、耗时。")
                else:
                    _sm = summarize_recognition_log(_log)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("累计识别", f"{_sm['total']} 份", f"失败 {_sm['failed']}")
                    m2.metric("规则平均覆盖", f"{_sm['avg_cover']*100:.0f}%" if _sm["avg_cover"] is not None else "—")
                    m3.metric("AI与规则不一致", f"{_sm['ai_diff']} 次")
                    m4.metric("平均耗时", f"{_sm['avg_ms']/1000:.1f} 秒" if _sm["avg_ms"] is not None else "—")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.caption("识别路线分布")
                        st.dataframe(pd.DataFrame([{"路线": k, "份数": v} for k, v in
                                                   sorted(_sm["route_cnt"].items(), key=lambda x: -x[1])]),
                                     use_container_width=True, hide_index=True)
                    with cc2:
                        st.caption("课表版式分布（哪种模板最容易出问题可据此判断）")
                        _mt_name = {"table": "框线表格", "list": "清单式", "grid": "坐标网格", "raw": "纯文本",
                                    "none": "无文本层(图片型)", "unknown": "未知"}
                        st.dataframe(pd.DataFrame([{"版式": _mt_name.get(k, k), "份数": v} for k, v in
                                                   sorted(_sm["method_cnt"].items(), key=lambda x: -x[1])]),
                                     use_container_width=True, hide_index=True)
                    _df_log = pd.DataFrame(_log[-100:])[::-1]
                    _show_cols = [c for c in ["time", "name", "method", "route_type", "cover_rate",
                                              "ai_agree", "ai_used", "failed", "elapsed_ms"] if c in _df_log.columns]
                    st.caption("最近 100 条明细（倒序）")
                    st.dataframe(_df_log[_show_cols], use_container_width=True, hide_index=True)
                    _csv = pd.DataFrame(_log).to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇️ 导出识别日志CSV（用于沉淀回归样本）", _csv,
                                       file_name="识别质量日志.csv", mime="text/csv")
                    st.caption("用法建议：把「失败」或「AI与规则不一致」的课表单独存一份样本，下次改动识别逻辑时先拿它们回归。")
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
            st.subheader("一键排班（稀缺度优先 + 职委均衡 + 绝不违背课表）")
            st.caption("策略：可用人数越少的班次越先排（难排的优先）；可选时段越少的人越先锁定；"
                       "学员先保证每人一班，人不够时才用「确实有空」的人补第二、三班；"
                       "实在没人可用的班次留空标记「待人工指派」，不会把有课的人硬排上去。")
            mlist = load_members()
            if st.button("生成值班表"):
                with st.spinner("正在智能排班，综合考虑课表冲突和个性化需求..."):
                    result = smart_schedule(mlist)
                    st.session_state.schedule_result = result
                st.success("排班完成！")
            if st.session_state.schedule_result:
                _disp = []
                for _row in st.session_state.schedule_result:
                    _disp.append({
                        "星期": _row["week"], "班次": _row["shift"],
                        "学员": "、".join(safe_str(x.get("name")) for x in _row["学员"]) or "—",
                        "职委": "、".join(safe_str(x.get("name")) for x in _row["职委"]) or "—",
                        "缺口": _row.get("缺口", 0),
                        "备注": _row.get("备注", ""),
                    })
                _gap = sum(1 for r in st.session_state.schedule_result if r.get("缺口", 0) > 0)
                if _gap:
                    st.warning(f"⚠️ 共 {_gap} 个班次存在缺口（见下方「备注」列），导出前请手动指派。")
                st.dataframe(pd.DataFrame(_disp), use_container_width=True, hide_index=True)
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
                cw = st.number_input("单元格宽度（不同模板需要微调）", value=82, min_value=20, max_value=300)
                ch = st.number_input("单元格高度（不同模板需要微调）", value=115, min_value=20, max_value=300)
                if not st.session_state.schedule_result:
                    st.warning("请先生成排班结果")
                elif not st.session_state.user_template_bytes:
                    st.warning("请先上传你的空白模板PDF")
                else:
                    if st.button("自动填充我的模板并导出"):
                        with st.spinner("正在填充你的模板PDF..."):
                            out_buf = fill_user_template_pdf(st.session_state.user_template_bytes, st.session_state.schedule_result,
                                                             int(off_x), int(off_y), int(cw), int(ch))
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
                admin_rows = [{"账号": safe_str(a.get("username")),
                               "角色": "超级管理员" if a.get("is_super") else "普通管理员"}
                              for a in admin_data["admins"] if isinstance(a, dict)]
                st.dataframe(pd.DataFrame(admin_rows), use_container_width=True)
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
                st.divider()
                st.markdown("#### 恢复默认账号密码（忘记密码时使用）")
                st.caption("⚠️ 将清空全部管理员账号，恢复为唯一的默认账号 admin / 123456（添加的其他管理员会被移除）。")
                confirm_reset = st.checkbox("我确认要执行重置", key="confirm_reset_admin")
                if st.button("🔧 一键恢复默认账号密码", disabled=not confirm_reset):
                    reset_default_admin()
                    st.success("✅ 已重置！现在账号 admin，密码 123456，请重新登录后立即修改密码。")
                # ---------- 页面壁纸 ----------
                st.divider()
                st.markdown("#### 🖼️ 页面壁纸（自定义操作页背景图）")
                _wcfg = load_wallpaper_config()
                st.file_uploader("上传背景图片（建议横图 1920×1080 左右，会自动压缩；支持 png/jpg/webp）",
                                 type=["png", "jpg", "jpeg", "webp"], key="wallpaper_upload")
                wc1, wc2 = st.columns(2)
                with wc1:
                    _w_alpha = st.slider("内容区不透明度（越大文字越清晰、背景越淡）", 0.30, 1.00,
                                         float(_wcfg.get("alpha", 0.82)), 0.02, key="wallpaper_alpha")
                with wc2:
                    _w_blur = st.slider("背景模糊（px，可让文字更易读）", 0, 16,
                                        int(_wcfg.get("blur", 0)), 1, key="wallpaper_blur")
                _w_enabled = st.toggle("启用壁纸", value=bool(_wcfg.get("enabled", False)), key="wallpaper_enabled")
                _cur_path = safe_str(_wcfg.get("path"))
                if _cur_path and os.path.exists(_cur_path):
                    st.caption("当前壁纸预览：")
                    try:
                        st.image(_cur_path, width=420)
                    except Exception:
                        pass
                wb1, wb2 = st.columns(2)
                with wb1:
                    if st.button("💾 保存壁纸设置"):
                        _up = st.session_state.get("wallpaper_upload")
                        _p = save_wallpaper_image(_up.read()) if _up is not None else _cur_path
                        save_wallpaper_config({"enabled": bool(_w_enabled), "alpha": float(_w_alpha),
                                               "blur": int(_w_blur), "path": _p})
                        st.success("✅ 壁纸已保存并对所有页面生效")
                        st.rerun()
                with wb2:
                    if st.button("🗑️ 移除壁纸"):
                        clear_wallpaper()
                        # 不能直接赋值：toggle 已在本轮实例化，赋值会抛
                        # StreamlitWidgetAlreadyInstantiatedError。删除 key 让控件
                        # 回落到默认值，下一轮 apply_wallpaper 会读到配置里的 enabled=False。
                        st.session_state.pop("wallpaper_enabled", None)
                        st.success("已移除壁纸，恢复默认白色背景")
                        st.rerun()
                st.caption("说明：拖动滑块会立即预览效果，点「保存壁纸设置」才会持久化（云端重启后仍生效）。"
                           "壁纸对所有页面生效，包括登录页和成员自助上传页。")
        if st.button("退出登录"):
            st.session_state.login = False
            st.session_state.current_admin = ""
            st.rerun()
elif menu == "成员自助上传":
    st.title("成员自助提交信息 & 上传课表")
    st.caption("识别逻辑与管理员后台上传完全一致：qwen‑plus 与确定性规则解析双通道互补（AI补漏+闸门防幻觉），图片型课表自动切 qwen‑vl‑max 读图；doubao2.1‑seed‑pro 解析个性化需求并逐条回溯原话证据。提交前请核对识别结果。")
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
                _m_sink = {}
                class_conflict = fast_ai_extract(pdf_bin, limit_input, _m_sink)
            try:
                append_recognition_log([build_log_record(name, _m_sink, limit_input)])
            except Exception:
                pass
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
                    conflict_owner = None
                    for m in all_mem:
                        if m["name"] == pending["name"]:
                            # 同名但手机号不同：可能是重名或误填，避免覆盖别人的时段
                            if safe_str(m.get("phone")) and safe_str(m.get("phone")) != pending["phone"]:
                                conflict_owner = m
                                break
                            m["phone"] = pending["phone"]
                            m["role"] = pending["role"]
                            m["class_conflict"] = final_conflict
                            m["personal_need"] = pending["personal_need"]
                            exist = True
                    if conflict_owner is not None:
                        st.error(f"❌ 人员库中已存在姓名「{pending['name']}」但手机号不一致，为避免覆盖他人信息已中止。请联系管理员处理，或确认姓名是否重名。")
                    else:
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
