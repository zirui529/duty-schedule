```python
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
    {"shift_idx": 0, "shift_name": "9:00‑13:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx": 1, "shift_name": "13:00‑17:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx": 2, "shift_name": "17:00‑20:00", "need_stu": 2, "need_wei": 1}
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
def init_admin():
    if not os.path.exists(ADMIN_FILE):
        data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
        with open(ADMIN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        try:
            with open(ADMIN_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)
            if "admins" not in old:
                old_pwd = old.get("admin_pwd", "123456")
                new_data = {"admins": [{"username": "admin", "password": old_pwd, "is_super": True}]}
                with open(ADMIN_FILE, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, OSError):
            data = {"admins": [{"username": "admin", "password": "123456", "is_super": True}]}
            with open(ADMIN_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def load_admin():
    init_admin()
    with open(ADMIN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_admin(data):
    tmp = f"{ADMIN_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ADMIN_FILE)


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
            return []
    return []


def save_members(data):
    tmp = f"{DATA_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, DATA_FILE)


# ========== PDF读取 ==========
def get_pdf_full_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    txt = "".join(page.get_text() for page in doc)
    doc.close()
    return txt


def get_session():
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    return session


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


# ========== AI底层调用 两套独立接口 ==========
def _call_course_ai(prompt_text):
    headers = {"Authorization": f"Bearer {VOLC_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": """
你是课表提取工具，**只输出JSON数组，绝对不能输出任何说明、注释、markdown、```符号**。

【节次标准时间表，严格参照】
1:08:00‑08:45｜2:08:50‑09:35｜3:09:50‑10:35｜4:10:40‑11:25｜5:11:30‑12:15
6:14:05‑14:50｜7:14:55‑15:40｜8:15:45‑16:30｜9:16:40‑17:25
10:17:30‑18:15｜11:18:30‑19:15｜12:19:20‑20:05｜13:20:10‑20:55

【极高优先级强制约束】
1. **只提取课表文字中明确出现的课程**。没有在课表文本中看到的星期+节次，一律不输出。
2. **表格中的空白格、没有课程名称的格子、没有地点/教师的格子，全部视为没课，坚决不输出。**
3. **禁止脑补、禁止推测、禁止根据同一行/同一列的其他课程补全空白格。**
4. 星期只能是：周一、周二、周三、周四、周五、周六、周日；节次只能是1‑13的整数，其他数字一律丢弃。
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
    match = re.search(r"\[.*\]", out, re.DOTALL)
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
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:120]}")
    res = resp.json()
    out = res["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\[.*\]", out, re.DOTALL)
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


# ========== 新版冲突计算：课程【开始时间】唯一归属班次，消除跨时段双重冲突 ==========
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
        valid_weeks = {"周一", "周二", "周三", "周四", "周五", "周六", "周日"}
        valid_sections = set(range(1, 14))
        cleaned_dict = {}
        for entry in ai_course_data:
            if not isinstance(entry, dict):
                continue
            w = entry.get("week", "")
            sec_list = entry.get("sections", [])
            if w not in valid_weeks:
                continue
            # 节次必须为int且在1-13之间
            clean_sec = [x for x in sec_list if isinstance(x, int) and x in valid_sections]
            if not clean_sec:
                continue
            # 去重并保持顺序稳定
            clean_sec = sorted(set(clean_sec))
            if w in cleaned_dict:
                cleaned_dict[w].extend(clean_sec)
            else:
                cleaned_dict[w] = clean_sec
        # 再次去重
        ai_course_data = [{"week": w, "sections": sorted(set(secs))} for w, secs in cleaned_dict.items()]
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
    stu_pool = [m for m in member_list if m["role"] == "学员"]
    wei_pool = [m for m in member_list if m["role"] == "职委"]
    st.info(f"📊 人员统计：学员 {len(stu_pool)} 人，职委 {len(wei_pool)} 人")
    if len(wei_pool) == 0:
        st.error("⚠️ 警告：人员库里没有职委！请去「人员库管理」把需要当职委的人身份改成「职委」")
    random.shuffle(stu_pool)
    random.shuffle(wei_pool)
    all_slots = []
    for w_idx, w in enumerate(WEEK_NAMES):
        for s_idx, s in enumerate(SHIFT_LIST):
            all_slots.append({"week_idx": w_idx, "shift_idx": s_idx, "week": w, "shift": s["shift_name"]})
    stu_count = {m["name"]: 0 for m in stu_pool}
    wei_count = {m["name"]: 0 for m in wei_pool}
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
        st.success(f"✅ 职委排班完成：共 {len(wei_pool)} 名职委，平均每人排 {sum(wei_count.values()) / len(wei_pool):.1f} 班")
    return schedule


# ========== 模式A：系统自动生成PDF ==========
def export_pdf_auto(schedule_data, template_name):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 14)
    c.drawCentredString(w / 2, h - 40, "武汉大学学生会值班表")
    y = h - 70
    if template_name == "武汉大学学生会水印模板":
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.setFont(FONT_NAME, 35)
        c.drawCentredString(w / 2, h / 2, "武汉大学学生会")
        c.setFillColorRGB(0, 0, 0)
    c.setFont(FONT_NAME, 9)
    for row in schedule_data:
        stu_text = ",".join([f'{x["name"]}({x["phone"]})' for x in row['学员']])
        wei_text = ",".join([f'{x["name"]}({x["phone"]})' for x in row['职委']])
        line = f"{row['week']} {row['shift']} | 学员：{stu_text} | 职委：{wei_text}"
        c.drawString(30, y, line)
        y -= 16
        if y < 40:
            c.showPage()
            y = h - 40
            c.setFont(FONT_NAME, 9)
    c.save()
    buf.seek(0)
    return buf


# ========== 模式B：填充用户模板 ==========
def fill_user_template_pdf(template_bytes, schedule_data, offset_x: int = 0, offset_y: int = -30):
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
            cell_dict[(si, wi)] = {"stu": [], "wei": []}
    for entry in schedule_data:
        k = (entry["shift_idx"], entry["week_idx"])
        if k in cell_dict:
            for p in entry["学员"]:
                cell_dict[k]["stu"].append((p["name"], p.get("phone", "")))
            for p in entry["职委"]:
                cell_dict[k]["wei"].append((p["name"], p.get("phone", "")))
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
if "temp_pdf_texts" not in st.session_state:
    st.session_state.temp_pdf_texts = []

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
        if is_super_admin(st.session_state.current_admin):
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                ["批量上传课表", "人员库管理", "自动排班", "导出PDF", "分享链接", "管理员管理"]
            )
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["批量上传课表", "人员库管理", "自动排班", "导出PDF", "分享链接"]
            )
            tab6 = None
        with tab1:
            st.subheader("批量上传PDF课表（文件名格式：身份&姓名&手机号-课表.pdf）")
            up_files = st.file_uploader("选择课表PDF", type="pdf", accept_multiple_files=True)
            if up_files and st.button("📂 第一步：快速识别姓名、身份、电话，生成预览表格"):
                with st.spinner("正在本地读取PDF和文件名..."):
                    texts = []
                    new_members = []
                    for f in up_files:
                        b = f.read()
                        t = get_pdf_full_text(b)
                        texts.append(t)
                        name = extract_name_from_text(t)
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
                    st.session_state.temp_pdf_texts = texts
                    seen = set()
                    unique = []
                    for item in new_members:
                        if item["name"] not in seen:
                            seen.add(item["name"])
                            unique.append(item)
                    st.session_state.temp_new_member = unique
                st.success(f"本地识别完成，共 {len(unique)} 人！先在表格里填好个性化需求，再点第二步让AI综合识别。")
                st.rerun()
            if len(st.session_state.temp_new_member) > 0:
                st.divider()
                st.subheader("📋 预览表格（先填个性化需求，再点AI识别，AI会综合课表+需求一起算）")
                df_temp = pd.DataFrame(st.session_state.temp_new_member)
                temp_config = {
                    "role": st.column_config.SelectboxColumn("身份", options=["学员", "职委"], required=True),
                    "class_conflict": st.column_config.TextColumn("AI识别不能值班时段（课表+需求综合）"),
                    "personal_need": st.column_config.TextColumn("个性化需求（手动填，AI会自动识别各种说法）")
                }
                edited_temp = st.data_editor(
                    df_temp, use_container_width=True, column_config=temp_config,
                    num_rows="dynamic", key="temp_editor"
                )
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("🤖 第二步：AI综合识别（课表+个性化需求）"):
                        current_rows = edited_temp.to_dict("records")
                        need_ai_idx = []
                        need_ai_texts = []
                        need_ai_personal = []
                        for i, row in enumerate(current_rows):
                            if not safe_str(row.get("class_conflict")):
                                need_ai_idx.append(i)
                                need_ai_texts.append(
                                    st.session_state.temp_pdf_texts[i]
                                    if i < len(st.session_state.temp_pdf_texts) else ""
                                )
                                need_ai_personal.append(safe_str(row.get("personal_need")))
                        if need_ai_texts:
                            with st.spinner(f"AI正在综合分析 {len(need_ai_texts)} 人的课表和个性化需求..."):
                                bar = st.progress(0)
                                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                    futures = {
                                        executor.submit(fast_ai_extract, t, p): idx
                                        for idx, t, p in zip(need_ai_idx, need_ai_texts, need_ai_personal)
                                    }
                                    done = 0
                                    for fut in as_completed(futures):
                                        idx = futures[fut]
                                        try:
                                            current_rows[idx]["class_conflict"] = fut.result()
                                        except Exception as e:
                                            current_rows[idx]["class_conflict"] = f"识别失败: {str(e)[:60]}"
                                        done += 1
                                        bar.progress(done / len(need_ai_texts))
                            for row in current_rows:
                                row["personal_need"] = safe_str(row.get("personal_need"))
                                row["class_conflict"] = safe_str(row.get("class_conflict"))
                                row["name"] = safe_str(row.get("name"))
                                row["phone"] = safe_str(row.get("phone"))
                                row["role"] = safe_str(row.get("role")) if safe_str(row.get("role")) else "学员"
                            st.session_state.temp_new_member = current_rows
                            st.success("AI综合识别完成！课表冲突和个性化需求已合并计算。")
                            st.rerun()
                        else:
                            st.info("所有人都已识别，不需要AI！")
                with col_b:
                    if st.button("✅ 确认保存入库"):
                        all_mem = load_members()
                        exist_names = {x["name"] for x in all_mem}
                        add_list = edited_temp.to_dict("records")
                        new_add = []
                        for row in add_list:
                            nm = safe_str(row.get("name"))
                            if nm and nm not in exist_names:
                                row["personal_need"] = safe_str(row.get("personal_need"))
                                row["class_conflict"] = safe_str(row.get("class_conflict"))
                                row["role"] = safe_str(row.get("role")) if safe_str(row.get("role")) else "学员"
                                new_add.append(row)
                        all_mem.extend(new_add)
                        save_members(all_mem)
                        st.success(f"成功录入 {len(new_add)} 人！")
                        st.session_state.temp_new_member = []
                        st.session_state.temp_pdf_texts = []
                        st.rerun()
                with col_c:
                    if st.button("❌ 放弃本批"):
                        st.session_state.temp_new_member = []
                        st.session_state.temp_pdf_texts = []
                        st.rerun()
        with tab2:
            st.subheader(f"人员库管理（共 {len(load_members())} 人）")
            mem = load_members()
            df = pd.DataFrame(mem)
            df.insert(0, "序号", range(1, len(df) + 1))
            edit_config = {
                "序号": st.column_config.NumberColumn("序号", disabled=True),
                "role": st.column_config.SelectboxColumn("身份", options=["学员", "职委"], required=True),
                "class_conflict": st.column_config.TextColumn("不能值班时段（课表+需求综合）"),
                "personal_need": st.column_config.TextColumn("个性化需求原文")
            }
            edit_df = st.data_editor(
                df, use_container_width=True, num_rows="dynamic", column_config=edit_config
            )
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
