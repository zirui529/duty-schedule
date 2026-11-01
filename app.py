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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ========== 基础配置 ==========
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PWD = "123456"
DATA_FILE = "member_data.json"
ADMIN_FILE = "admin_info.json"

# 火山方舟配置，只读取secrets，不硬编码密钥
try:
    VOLC_API_KEY = st.secrets["ARK_API_KEY"]
    VOLC_ENDPOINT = st.secrets["ARK_ENDPOINT"]
    MODEL_ID = st.secrets["ARK_MODEL_ENDPOINT_ID"]
except KeyError:
    st.error("缺少火山方舟密钥，请在Streamlit Settings -> Secrets配置ARK_API_KEY、ARK_ENDPOINT、ARK_MODEL_ENDPOINT_ID")
    st.stop()

API_TIMEOUT = 30
MAX_RETRY = 0
MAX_WORKERS = 4
SHIFT_LIST = [
    # 将全角连字符 '‑' (U+2011) 替换为半角连字符 '-' (U+002D) 以提高兼容性
    {"shift_idx": 0, "shift_name": "9:00-13:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx": 1, "shift_name": "13:00-17:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx": 2, "shift_name": "17:00-20:00", "need_stu": 2, "need_wei": 1}
]
WEEK_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TEMPLATE_OPTIONS = ["默认样式", "武汉大学学生会水印模板"]
SHIFT_PERIOD = {0: "上午", 1: "下午", 2: "晚上"}
ALL_SLOTS = [
    "周一上午", "周一下午", "周一晚上",
    "周二上午", "周二下午", "周二晚上",
    "周三上午", "周三下午", "周三晚上",
    "周四上午", "周四下午", "周四晚上",
    "周五上午", "周五下午", "周五晚上",
    "周六上午", "周六下午", "周六晚上",
    "周日上午", "周日下午", "周日晚上"
]

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT_NAME = 'STSong-Light'


def safe_str(val):
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


# ========== 管理员文件读写 ==========
# 优化：所有文件读写都使用 with 语句确保文件正确关闭
# 优化：保存操作使用临时文件 + os.replace 实现原子性写入，防止数据损坏
def _write_json_atomically(path, data):
    """原子性地将数据写入 JSON 文件，防止文件损坏。"""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # 确保数据写入磁盘
    os.replace(tmp_path, path) # 原子性替换原文件

def init_admin():
    if not os.path.exists(ADMIN_FILE):
        data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
        _write_json_atomically(ADMIN_FILE, data)
    else:
        try:
            with open(ADMIN_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)
            if "admins" not in old: # 迁移旧版数据结构
                old_pwd = old.get("admin_pwd", "123456")
                new_data = {"admins": [{"username": "admin", "password": old_pwd, "is_super": True}]}
                _write_json_atomically(ADMIN_FILE, new_data)
        except (json.JSONDecodeError, OSError): # 文件损坏或无法读取时，重新初始化
            data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
            _write_json_atomically(ADMIN_FILE, data)

def load_admin():
    init_admin() # 确保文件存在且格式正确
    try:
        with open(ADMIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # 即使 init_admin 尝试修复，极端情况下也可能失败，这里提供一个回退
        st.error("管理员信息文件损坏，已重置为默认管理员账号：admin，密码：123456。")
        init_admin() # 再次尝试修复或创建
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
            # 兼容旧数据结构，确保字段存在
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
            return [] # 文件损坏时返回空列表
    return []

def save_members(data):
    _write_json_atomically(DATA_FILE, data)

# ========== PDF读取 ==========
def get_pdf_full_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    txt = "".join(page.get_text() for page in doc)
    doc.close()
    return txt

def get_session():
    # 此函数在代码中未被实际调用，保留以供未来扩展
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    return session

def extract_name_from_text(text):
    # 优化：优先匹配行首，避免多余文字干扰
    for line in text.splitlines():
        m = re.fullmatch(r"([\u4e00-\u9fa5]{2,4})\s*课表", line.strip())
        if m:
            return m.group(1).strip()
    # 如果行首匹配不到，再进行全文搜索
    m = re.search(r"([\u4e00-\u9fa5]{2,4})课表", text)
    return m.group(1).strip() if m else ""

def parse_filename_phone_and_role(filename):
    name_part = filename.split("-课表")[0]
    parts = name_part.split("&")
    role = "学员" # 默认身份为学员
    phone = ""
    if len(parts) >= 3:
        role_raw = parts[0]
        # 优化：增加更多职委关键词，提高识别率
        if "职委" in role_raw or "干部" in role_raw or "部长" in role_raw or "主席" in role_raw: 
            role = "职委"
        else:
            role = "学员"
        phone_candidate = parts[2].strip()
        try:
            if re.fullmatch(r"1[3-9]\d{9}", phone_candidate):
                phone = phone_candidate
        except re.error as e:
            # 增加防御性错误处理，如果正则表达式本身出错，则记录并忽略手机号
            st.error(f"解析文件名中的手机号正则匹配失败: {e}。请检查文件名或代码中的手机号正则。")
            phone = "" # 发生正则错误时清空手机号
    return role, phone

# ========== AI底层调用 两套独立接口 ==========
def _call_course_ai(prompt_text):
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
    if resp.status_code != 200: # 优化：检查HTTP状态码
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:120]}")
    res = resp.json()
    out = res["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out) # 移除Markdown代码块开始标记
    out = re.sub(r"\n```$", "", out)      # 移除Markdown代码块结束标记
    match = re.search(r"\\[.*\\\]", out, re.DOTALL) # 提取最外层的JSON数组
    if not match:
        raise ValueError("模型没有返回合法JSON数组")
    arr = json.loads(match.group(0))
    return arr

def _call_personal_ai(prompt_text):
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
    if resp.status_code != 200: # 优化：检查HTTP状态码
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:120]}")
    res = resp.json()
    out = res["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\\[.*\\\]", out, re.DOTALL)
    if not match:
        return []
    arr = json.loads(match.group(0))
    return [s.strip() for s in arr if isinstance(s, str)]

# 【节次精确时间（分钟，从0点起算）】
SECTION_TIME = {
    1: (8 * 60 + 0, 8 * 60 + 45),
    2: (8 * 60 + 50, 9 * 60 + 35),
    3: (9 * 60 + 50, 10 * 60 + 35),
    4: (10 * 60 + 40, 11 * 60 + 25),
    5: (11 * 60 + 30, 12 * 60 + 15),
    6: (14 * 60 + 5, 14 * 60 + 50),
    7: (14 * 60 + 55, 15 * 60 + 40),
    8: (15 * 60 + 45, 16 * 60 + 30),
    9: (16 * 60 + 40, 17 * 60 + 25),
    10: (17 * 60 + 30, 18 * 60 + 15),
    11: (18 * 60 + 30, 19 * 60 + 15),
    12: (19 * 60 + 20, 20 * 60 + 5),
    13: (20 * 60 + 10, 20 * 60 + 55)
}

# ==========新版冲突计算：课程【开始时间】唯一归属班次，消除跨时段双重冲突==========
def calc_course_conflict(ai_course_data):
    conflict = set()
    for item in ai_course_data:
        # 优化：确保 week 字段是字符串且符合规范
        w = str(item.get("week", "")).strip().replace("星期", "周")
        if w not in WEEK_NAMES: # 过滤非法星期
            continue
        
        secs = item.get("sections", [])
        if not isinstance(secs, list): # 过滤非法节次列表
            continue

        for s in secs:
            if not isinstance(s, int) or s not in SECTION_TIME: # 过滤非法节次数字
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

def fast_ai_extract(text: str, personal_need: str = ""):
    if not text or len(text.strip()) < 10:
        return "课表文本提取失败，请手动填写"
    
    prompt_course = f"""
下面是课表原文，请提取里面所有真实存在的星期和节次。

【提醒】
- 表格中的空白格、没有课程名的格子，全部跳过。
- 不要根据周围的课程推断空白格。
- 如果整个课表无课，输出空数组[]。

课表原文：
{text[:8000]}
"""
    try:
        ai_course_data = _call_course_ai(prompt_course)
        
        # 本地后置校验：过滤AI幻觉输出的非法星期、非法节次，去重排序
        valid_weeks = set(WEEK_NAMES)
        valid_sections = set(range(1, 14))
        cleaned_dict = {} # 使用字典来按周分组并确保节次唯一
        
        for entry in ai_course_data:
            if not isinstance(entry, dict):
                continue
            
            w = str(entry.get("week", "")).strip().replace("星期", "周") # 再次清洗星期
            sec_list = entry.get("sections", [])
            
            if w not in valid_weeks:
                continue
            
            # 节次必须为int且在1-13之间
            clean_sec = [x for x in sec_list if isinstance(x, int) and x in valid_sections]
            if not clean_sec:
                continue
            
            # 将清洗后的节次加入字典，自动处理重复的节次
            if w in cleaned_dict:
                cleaned_dict[w].extend(clean_sec)
            else:
                cleaned_dict[w] = clean_sec
        
        # 将字典转换为最终的列表格式，并对每个星期的节次进行去重和排序
        ai_course_data = [{"week": w, "sections": sorted(list(set(secs)))} for w, secs in cleaned_dict.items()]
        
        # DEBUG折叠面板，仅streamlit页面环境输出
        if st.runtime.exists():
            with st.expander("🔍DEBUG AI读取原始课表(仅管理员批量上传可见)"):
                st.code(json.dumps(ai_course_data, ensure_ascii=False, indent=2), language="json")
        
        course_conflict = calc_course_conflict(ai_course_data)
        
    except Exception as e:
        return f"课表识别失败:{str(e)[:60]}"

    personal_conflict = []
    if personal_need and personal_need.strip() not in ("", "无"):
        prompt_personal = f"""
课表已经算出不能值班时段：{json.dumps(course_conflict, ensure_ascii=False)}
用户额外要求：{personal_need}
只输出用户额外新增禁止的时段，不要包含课表已经有的。
"""
        try:
            personal_conflict = _call_personal_ai(prompt_personal)
        except Exception: # 即使个人需求AI识别失败，也不影响课表冲突
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
    stu_pool = [m for m in member_list if m["role"] == "学员"]
    wei_pool = [m for m in member_list if m["role"] == "职委"]
    st.info(f"📊 人员统计：学员 {len(stu_pool)} 人，职委 {len(wei_pool)} 人")
    if len(wei_pool) == 0:
        st.error("⚠️ 警告：人员库里没有职委！请去「人员库管理」把需要当职委的人身份改成「职委」")
        # 优化：如果没有职委，
