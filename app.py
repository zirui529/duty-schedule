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

# 火山方舟配置，从Streamlit Secrets读取
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
WEEK_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TEMPLATE_OPTIONS = ["默认样式", "武汉大学学生会水印模板"]

SHIFT_PERIOD = {0: "上午", 1: "下午", 2: "晚上"}

ALL_SLOTS = ["周一上午","周一下午","周一晚上","周二上午","周二下午","周二晚上","周三上午","周三下午","周三晚上","周四上午","周四下午","周四晚上","周五上午","周五下午","周五晚上","周六上午","周六下午","周六晚上","周日上午","周日下午","周日晚上"]

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

# ========== AI综合解析 ==========
def fast_ai_extract(text:str, personal_need:str=""):
    headers = {"Authorization": f"Bearer {VOLC_API_KEY}", "Content-Type": "application/json"}
    
    def call_ai(prompt_text):
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role":"system","content":"你是大学课表分析专家，擅长从PDF提取的混乱表格文本中准确识别每天的课程安排。你只输出JSON数组，不输出任何解释。"},
                {"role":"user","content": prompt_text}
            ],
            "temperature":0.0,
            "max_tokens":800,
            "service_tier":"default"
        }
        resp = get_session().post(VOLC_ENDPOINT, headers=headers, json=payload, timeout=API_TIMEOUT)
        res = resp.json()
        out = res["choices"][0]["message"]["content"].strip()
        if out.startswith("```"):
            out = re.sub(r"^```[a-z]*\n?", "", out)
            out = re.sub(r"\n?```$", "", out)
        return out

    personal_section = ""
    if personal_need and personal_need != "无":
        personal_section = f"""
同时这个人还有个性化需求：{personal_need}
请仔细理解个性化需求，把所有不能值班的时段都加进数组：
- 如果说"X时间有事/不值班/不能来/要上课/回家/请假" → X时间加进数组
- 如果说"只在X时间有空/只能X时间值班" → 除了X时间以外，其他所有时段都加进数组
- 如果说"周末不值班/周末有事" → 周六上午、周六下午、周六晚上、周日上午、周日下午、周日晚上都加进数组
- 如果说"周X下午有空" → 除了周X下午，其他所有时段都加进数组
- 如果说"晚上不值班" → 周一晚上、周二晚上、周三晚上、周四晚上、周五晚上、周六晚上、周日晚上都加进数组
"""

    prompt1 = f"""请分析下面的大学课表文本，找出这个人所有不能值班的时段。

值班时段共21个，你必须逐个检查，一个都不能漏：
周一上午、周一下午、周一晚上、周二上午、周二下午、周二晚上、周三上午、周三下午、周三晚上、周四上午、周四下午、周四晚上、周五上午、周五下午、周五晚上、周六上午、周六下午、周六晚上、周日上午、周日下午、周日晚上

判断规则（必须严格遵守）：
【上午班冲突】上午有课就算 = 表格里上午区域（一般是前5节课）当天有任何课
【下午班冲突】下午有课就算 = 表格里下午区域（一般是中间几节课）当天有任何课
【晚上班冲突】晚上有课就算 = 表格里下午晚些时候或傍晚/晚上区域（一般是靠后的几节课）当天有任何课

具体节次对应：
- 上午班（9:00-13:00）冲突：第2、3、4、5节有课就算（第1节8:45下课，不算）
- 下午班（13:00-17:00）冲突：第6、7、8、9节有课就算
- 晚上班（17:00-20:00）冲突：第9、10、11、12节有课就算（第13节20:10开始，不算）

⚠️ 特别提醒（非常重要）：
1. 晚上的课最容易被漏掉！请你从头到尾通读课表文本，不要只看前半部分！
2. 很多PDF课表的晚上课排在表格下方，甚至单独一行，很容易被忽略！
3. 对每一天，都要分别确认：上午有没有课？下午有没有课？晚上有没有课？
4. 只要那天下午的课排得比较靠后（超过3节），就要检查晚上是不是也有课！
5. 不要因为"晚上应该没课"就跳过，必须以课表文本为准！

只输出JSON数组格式，不要任何解释文字
{personal_section}

课表文本如下：
{text[:2000]}"""

    try:
        out1 = call_ai(prompt1)
        arr1 = json.loads(out1)
        if len(arr1) < 2:
            prompt2 = f"""重新检查一遍！重点检查每天下午和晚上有没有漏掉的课！
已经识别出的不能值班的时段：{arr1}
请逐天核对周一到周日，每天分别确认：
- 上午班（第2、3、4、5节）有没有课？
- 下午班（第6、7、8、9节）有没有课？
- 晚上班（第9、10、11、12节）有没有课？
不要漏掉任何下午和晚上的课！
还是只输出完整的JSON数组（包含所有不能值班的时段），不要解释。
{personal_section}
课表文本：
{text[:2000]}"""
            out2 = call_ai(prompt2)
            arr2 = json.loads(out2)
            combined = list(set(arr1 + arr2))
            arr1 = combined
        if isinstance(arr1, list) and len(arr1) > 0:
            return "、".join(arr1)
        else:
            return "无"
    except Exception as e:
        st.error(f"AI调用失败: {e}")
        return "无"

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
if "temp_pdf_texts" not in st.session_state:
    st.session_state.temp_pdf_texts = []

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
                        need_ai_texts = []
                        need_ai_personal = []
                        for i, row in enumerate(current_rows):
                            if not safe_str(row.get("class_conflict")):
                                need_ai_idx.append(i)
                                need_ai_texts.append(st.session_state.temp_pdf_texts[i] if i < len(st.session_state.temp_pdf_texts) else "")
                                need_ai_personal.append(safe_str(row.get("personal_need")))
                        if need_ai_texts:
                            with st.spinner(f"AI正在综合分析 {len(need_ai_texts)} 人的课表和个性化需求..."):
                                bar = st.progress(0)
                                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                    futures = {executor.submit(fast_ai_extract, t, p): idx for idx, t, p in zip(need_ai_idx, need_ai_texts, need_ai_personal)}
                                    done = 0
                                    for fut in as_completed(futures):
                                        idx = futures[fut]
                                        try:
                                            current_rows[idx]["class_conflict"] = fut.result()
                                        except Exception as e:
                                            st.error(f"第{idx+1}个人AI识别失败: {e}")
                                            current_rows[idx]["class_conflict"] = "无"
                                        done += 1
                                        bar.progress(done/len(need_ai_texts))
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
            app_url = st.text_input("部署后的网址", value="http://localhost:8501")
            self_link = f"{app_url}?page=self"
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
    with st.form("member_form"):
        name = st.text_input("姓名")
        phone = st.text_input("联系电话")
        role_sel = st.radio("身份", ["学员","职委"])
        limit_input = st.text_input("个性化需求（选填，各种说法都行：周末不值班 / 只在周六下午有空 / 周一上午有课）")
        pdf_file = st.file_uploader("上传你的课表PDF", type="pdf")
        submit = st.form_submit_button("提交")
        if submit:
            if not (name and phone and pdf_file):
                st.warning("姓名、电话、课表都需要填！")
            else:
                with st.spinner("AI正在综合分析你的课表和个性化需求..."):
                    txt = get_pdf_full_text(pdf_file.read())
                    class_conflict = fast_ai_extract(txt, limit_input)
                    personal_need = safe_str(limit_input)
                    all_mem = load_members()
                    exist = False
                    for m in all_mem:
                        if m["name"] == name:
                            m["phone"] = phone
                            m["role"] = role_sel
                            m["class_conflict"] = class_conflict
                            m["personal_need"] = personal_need
                            exist = True
                    if not exist:
                        all_mem.append({"name":name,"phone":phone,"role":role_sel,"class_conflict":class_conflict,"personal_need":personal_need})
                    save_members(all_mem)
                st.success("提交成功！AI已综合识别你所有不能值班的时段。")
