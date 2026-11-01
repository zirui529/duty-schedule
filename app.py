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
    {"shift_idx":0,"shift_name": "9:00‑13:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx":1,"shift_name": "13:00‑17:00", "need_stu": 2, "need_wei": 1},
    {"shift_idx":2,"shift_name": "17:00‑20:00", "need_stu": 2, "need_wei": 1}
]
WEEK_NAMES_ORDER = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]
WEEK_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TEMPLATE_OPTIONS = ["默认样式", "武汉大学学生会水印模板"]
SHIFT_PERIOD = {0: "上午", 1: "下午", 2: "晚上"}
ALL_SLOTS = ["周一上午","周一下午","周一晚上","周二上午","周二下午","周二晚上","周三上午","周三下午","周三晚上","周四上午","周四下午","周四晚上","周五上午","周五下午","周五晚上","周六上午","周六下午","周六晚上","周日上午","周日下午","周日晚上"]

# 班次时间（分钟 0点起算）
SHIFT_TIME_MIN = {
    "上午": (9*60, 13*60),
    "下午": (13*60, 17*60),
    "晚上": (17*60, 20*60)
}
# 节次时间
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

# ==========【纯Python本地动态坐标解析课表，完全不调用大模型】 ==========
def parse_timetable_local(pdf_bytes):
    """
    动态探测表头星期、节次坐标，本地解析课表
    返回: timetable_list, warn_list
    timetable_list: [{"week":"周三","sections":[1,2,3,4,11,12]}]
    warn_list: 告警文本列表
    """
    warn_list = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    words = page.get_text("words")
    doc.close()

    week_header_pos = dict()
    section_header_pos = dict()
    rest_words = []

    # 第一步：扫描表头文字块
    for w in words:
        x0,y0,x1,y1,txt = w[0],w[1],w[2],w[3],safe_str(w[4])
        if txt in WEEK_NAMES_ORDER:
            week_header_pos[txt] = x0
        m = re.fullmatch(r"(\d{1,2})", txt)
        if m:
            snum = int(m.group(1))
            if 1<=snum<=13:
                section_header_pos[snum] = y0
        else:
            rest_words.append((x0,y0,x1,y1,txt))

    # 校验表头完整性：区分周日缺失（正常）和其他星期缺失（严重）
    missing_week = [w for w in WEEK_NAMES_ORDER if w not in week_header_pos]
    for mw in missing_week:
        if mw == "周日":
            warn_list.append("提示：未探测到【周日】表头，该PDF导出不包含周日列，属于教务导出常见变体，可忽略。")
        else:
            warn_list.append(f"⚠️严重警告：未探测到星期表头：{mw}，解析结果极可能出错，请人工复核！")

    missing_sec = [s for s in range(1,14) if s not in section_header_pos]
    if missing_sec:
        warn_list.append(f"⚠️警告：未探测到节次表头：{','.join(map(str,missing_sec))}，解析结果可能出错，请人工复核！")

    # 构造星期X边界，按x坐标从小到大排序
    sorted_week = sorted(week_header_pos.items(), key=lambda x:x[1])
    week_bounds = []
    for idx,(wname,x) in enumerate(sorted_week):
        if idx < len(sorted_week)-1:
            next_x = sorted_week[idx+1][1]
            week_bounds.append((wname, x, next_x))
        else:
            week_bounds.append((wname, x, 9999))

    # 构造节次Y边界，注意PDF y轴原点在页面左下角，y数值向上变小
    sorted_sec = sorted(section_header_pos.items(), key=lambda x:x[1])
    sec_bounds = []
    for idx,(snum,y) in enumerate(sorted_sec):
        if idx < len(sorted_sec)-1:
            next_y = sorted_sec[idx+1][1]
            sec_bounds.append((snum, next_y, y))
        else:
            sec_bounds.append((snum, -9999, y))

    week_sec_map = {wn:set() for wn in WEEK_NAMES_ORDER}
    # 兼容半角 "-" / 全角 "‑" 两种连堂符号
    range_pat = re.compile(r"(\d{1,2})\s*[\-‑]\s*(\d{1,2})")

    for (x0,y0,x1,y1,txt) in rest_words:
        match_range = range_pat.search(txt)
        nums_extract = []
        if match_range:
            a = int(match_range.group(1))
            b = int(match_range.group(2))
            if 1<=a<=13 and 1<=b<=13 and a<=b:
                nums_extract = list(range(a,b+1))
        else:
            for num_str in re.findall(r"\b([1-9]|1[0-3])\b", txt):
                ns = int(num_str)
                if 1<=ns<=13:
                    nums_extract.append(ns)
        if not nums_extract:
            continue
        # 判断属于哪个星期列
        hit_week = None
        for wname, bx_left, bx_right in week_bounds:
            if bx_left <= x0 < bx_right:
                hit_week = wname
                break
        if hit_week is None:
            warn_list.append(f"文字[{txt}]无法匹配到星期列，已丢弃")
            continue
        # 判断属于哪些节次行，取文字块垂直中心
        y_mid = (y0+y1)/2
        for snum, by_low, by_high in sec_bounds:
            if by_low <= y_mid <= by_high:
                week_sec_map[hit_week].update(nums_extract)

    # 组装输出结构
    timetable_list = []
    for wn in WEEK_NAMES_ORDER:
        sec_set = sorted(list(week_sec_map[wn]))
        if len(sec_set)>0:
            timetable_list.append({"week":wn,"sections":sec_set})
    return timetable_list, warn_list

def timetable_to_conflict(timetable_list):
    """把本地解析出来的课表结构，转换成冲突时段字符串列表"""
    conflict_set = set()
    for item in timetable_list:
        w = item["week"]
        secs = item["sections"]
        for s in secs:
            if s not in SECTION_TIME:
                continue
            c_start,c_end = SECTION_TIME[s]
            for shift_name,(sh_start,sh_end) in SHIFT_TIME_MIN.items():
                if not (c_end <= sh_start or c_start >= sh_end):
                    conflict_set.add(f"{w}{shift_name}")
    return sorted(list(conflict_set))

# ========== 个性化AI调用，仅非空才执行网络请求 ==========
def _call_personal_ai(prompt_text):
    headers = {"Authorization": f"Bearer {VOLC_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": """
只输出JSON字符串数组，内容是【周X上午/下午/晚上】。
只解析用户写的个人限制文字，**不要参考课表，不要推断课表信息**，只返回用户额外要求禁止的时段。
无额外限制返回空数组[]。示例：["周日上午","周二晚上"]
禁止markdown、禁止解释文字。
"""},
            {"role": "user", "content": prompt_text}
        ],
        "temperature":0
    }
    resp = requests.post(VOLC_ENDPOINT, headers=headers, json=payload, timeout=API_TIMEOUT)
    res = resp.json()
    out = res["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        return []
    arr = json.loads(match.group(0))
    return [s.strip() for s in arr if isinstance(s, str)]

def merge_personal_conflict(base_conflict_str:str, personal_need:str):
    personal_conflict = []
    need_ai = bool(personal_need and personal_need.strip() not in ("","无"))
    if need_ai:
        try:
            personal_conflict = _call_personal_ai(personal_need)
        except Exception:
            personal_conflict = []
    base_list = re.split(r"、", safe_str(base_conflict_str))
    base_list = [x.strip() for x in base_list if x.strip()]
    total_set = set(base_list)
    total_set.update(personal_conflict)
    final_str = "、".join(sorted(total_set)) if total_set else ""
    return final_str, personal_conflict

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
if "temp_raw_text" not in st.session_state:
    st.session_state.temp_raw_text = []
if "temp_timetable" not in st.session_state:
    st.session_state.temp_timetable = []
if "temp_warns" not in st.session_state:
    st.session_state.temp_warns = []

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
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["批量上传课表","人员库管理","自动排班","导出PDF","分享链接","管理员管理"])
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["批量上传课表","人员库管理","自动排班","导出PDF","分享链接"])
            tab6 = None
        with tab1:
            st.subheader("批量上传PDF课表【本地Python预解析课表；**仅个性化不为空才调用AI**；预填结果务必人工复核】")
            up_files = st.file_uploader("选择课表PDF，多个可批量", type="pdf", accept_multiple_files=True)
            if up_files and st.button("📂第一步：本地解析PDF，生成预填表格（无大模型调用）"):
                temp_members = []
                temp_raw = []
                temp_tt = []
                temp_wn = []
                for f in up_files:
                    pdf_bytes = f.read()
                    tt_list, warn_list = parse_timetable_local(pdf_bytes)
                    conflict_pre_list = timetable_to_conflict(tt_list)
                    conflict_pre_str = "、".join(conflict_pre_list)
                    doc_tmp = fitz.open(stream=pdf_bytes,filetype="pdf")
                    raw_text = doc_tmp.get_page_text(0)
                    doc_tmp.close()
                    name = extract_name_from_text(raw_text)
                    role, phone = parse_filename_phone_and_role(f.name)
                    temp_members.append({
                        "name":name,
                        "phone":phone,
                        "role":role,
                        "class_conflict": conflict_pre_str,
                        "personal_need": "",
                        "local_timetable": tt_list,
                        "local_warn": warn_list
                    })
                    temp_raw.append(raw_text)
                    temp_tt.append(tt_list)
                    temp_wn.append(warn_list)
                st.session_state.temp_new_member = temp_members
                st.session_state.temp_raw_text = temp_raw
                st.session_state.temp_timetable = temp_tt
                st.session_state.temp_warns = temp_wn
                st.success("✅本地解析完成！⚠️预填仅做参考，务必人工核对修改class_conflict再保存！")
                st.rerun()
            if len(st.session_state.temp_new_member) > 0:
                st.divider()
                st.warning("⚠️【重要】课表冲突是Python本地预填结果，**必须人工复核修改，确认无误再入库！课表解析不调用AI；只有个性化不为空才会调用AI接口**")
                df_temp = pd.DataFrame(st.session_state.temp_new_member)
                temp_config = {
                    "role": st.column_config.SelectboxColumn("身份", options=["学员","职委"], required=True),
                    "class_conflict": st.column_config.TextColumn("不能值班时段【人工复核修改】"),
                    "personal_need": st.column_config.TextColumn("个性化限制（非空才调用AI合并）")
                }
                edited_temp = st.data_editor(df_temp, use_container_width=True, column_config=temp_config, num_rows="dynamic", key="temp_edit")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("🤖第二步：合并个性化限制（仅personal_need非空会调用AI）"):
                        rows = edited_temp.to_dict("records")
                        for idx,row in enumerate(rows):
                            base = safe_str(row["class_conflict"])
                            pn = safe_str(row["personal_need"])
                            final_str, p_conf = merge_personal_conflict(base, pn)
                            rows[idx]["class_conflict"] = final_str
                        st.session_state.temp_new_member = rows
                        st.success("个性化合并完成，请再次复核结果")
                        st.rerun()
                with col_b:
                    if st.button("✅确认保存入库"):
                        all_mem = load_members()
                        exist_names = {x["name"] for x in all_mem}
                        add_rows = edited_temp.to_dict("records")
                        added = 0
                        for r in add_rows:
                            nm = safe_str(r["name"])
                            if nm and nm not in exist_names:
                                new_item = {
                                    "name":nm,
                                    "phone":safe_str(r["phone"]),
                                    "role":safe_str(r["role"]) if safe_str(r["role"]) else "学员",
                                    "class_conflict":safe_str(r["class_conflict"]),
                                    "personal_need":safe_str(r["personal_need"])
                                }
                                all_mem.append(new_item)
                                added +=1
                        save_members(all_mem)
                        st.success(f"成功入库{added}条记录")
                        st.session_state.temp_new_member = []
                        st.session_state.temp_raw_text = []
                        st.session_state.temp_timetable = []
                        st.session_state.temp_warns = []
                        st.rerun()
                with col_c:
                    if st.button("❌放弃本批"):
                        st.session_state.temp_new_member = []
                        st.session_state.temp_raw_text = []
                        st.session_state.temp_timetable = []
                        st.session_state.temp_warns = []
                        st.rerun()

                # 逐条展示解析告警
                st.divider()
                st.subheader("🔍各文件解析告警信息")
                for idx,row in enumerate(st.session_state.temp_new_member):
                    name = row["name"]
                    warns = row.get("local_warn",[])
                    with st.expander(f"【{name}】解析告警"):
                        if len(warns) ==0:
                            st.success("无解析告警")
                        else:
                            for wtxt in warns:
                                st.info(wtxt)
                        st.markdown("本地解析识别课表结构：")
                        st.code(json.dumps(row.get("local_timetable",[]),ensure_ascii=False,indent=2),language="json")

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
            st.subheader("一键排班（课表冲突+个性化限制+职委均匀轮班）")
            mlist = load_members()
            if st.button("生成值班表"):
                with st.spinner("正在智能排班..."):
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
                st.info("上传你自己的空白值班表PDF模板")
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
            st.info("成员自助页面上传PDF**不会自动解析课表，需要管理员后台补录冲突时段**")
            try:
                base_app_url = str(st.context.url)
            except Exception:
                base_app_url = "http://localhost:8501"
            base_app_url = base_app_url.rstrip("/")
            self_link = f"{base_app_url}?page=self"
            st.text_input("访问链接", value=base_app_url, disabled=True)
            st.code(self_link, language="text")
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
                                save_members(admin_data)
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
    st.info("⚠️自助上传仅保存姓名、手机号、身份、个性化需求；**课表冲突需要管理员后台人工补录**，不会自动解析PDF课表。")
    with st.form("member_form"):
        name = st.text_input("姓名")
        phone = st.text_input("联系电话")
        role_sel = st.radio("身份", ["学员","职委"])
        limit_input = st.text_input("个性化需求（选填，无则留空）")
        pdf_file = st.file_uploader("上传你的课表PDF（仅用于存档，不自动解析）", type="pdf")
        submit = st.form_submit_button("提交")
        if submit:
            if not (name and phone):
                st.warning("姓名、联系电话为必填项")
            else:
                all_mem = load_members()
                exist = False
                for m in all_mem:
                    if m["name"] == name:
                        m["phone"] = phone
                        m["role"] = role_sel
                        m["personal_need"] = safe_str(limit_input)
                        exist = True
                if not exist:
                    all_mem.append({
                        "name":name,
                        "phone":phone,
                        "role":role_sel,
                        "class_conflict":"",
                        "personal_need":safe_str(limit_input)
                    })
                save_members(all_mem)
                st.success("提交成功！请联系管理员补录你的课表冲突时段。")
