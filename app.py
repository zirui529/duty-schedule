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
# 移除了未使用的 requests.adapters.HTTPAdapter 和 urllib3.util.retry.Retry
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from json import JSONDecodeError # 明确导入 JSONDecodeError

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
    st.warning("⚠️ 缺少火山方舟密钥，请在 Streamlit Secrets 中配置 ARK_API_KEY、ARK_ENDPOINT、ARK_MODEL_ENDPOINT_ID。AI 识别功能将不可用。")

API_TIMEOUT = 30
MAX_WORKERS = 4
SHIFT_LIST = [
    # 统一使用标准 ASCII 连字符 '-'
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
        except (JSONDecodeError, OSError): # 使用导入的 JSONDecodeError
            _write_json_atomically(ADMIN_FILE, {"admins": [{"username": "admin", "password": "123456", "is_super": True}]})

def load_admin():
    init_admin()
    try:
        with open(ADMIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (JSONDecodeError, OSError):
        st.error("管理员文件损坏，已重置为默认账号。")
        init_admin() # 再次尝试修复
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
        except (JSONDecodeError, OSError):
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
        # CRITICAL CHANGE: 彻底移除正则表达式，改为简单的字符串检查
        if len(phone_candidate) == 11 and phone_candidate.isdigit() and phone_candidate.startswith('1'):
            phone = phone_candidate
        else:
            phone = "" # 如果不符合简单规则，则认为不是有效手机号
    return role, phone

# ========== AI 调用（带防幻觉） ==========
def _call_course_ai(prompt_text):
    if not VOLC_API_KEY:
        raise RuntimeError("火山方舟 API Key 未配置。")
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
        raise RuntimeError(f"API 请求失败，状态码 {resp.status_code}: {resp.text[:120]}")
    
    try:
        response_data = resp.json() # 确保这一行是完整的，并处理 JSON 解码错误
    except JSONDecodeError:
        raise ValueError(f"API 返回非 JSON 格式数据，状态码 {resp.status_code}: {resp.text[:120]}")

    out = response_data["choices"][0]["message"]["content"].strip()
    # 重新字面检查 regex 模式，确保无隐形字符
    out = re.sub(r"^```[^\n]*\n", "", out) 
    out = re.sub(r"\n```$", "", out)       
    match = re.search(r"\\[.*\\\]", out, re.DOTALL) 
    if not match:
        raise ValueError(f"模型返回数据不包含合法的 JSON 数组。原始输出：{out[:500]}") # 增加原始输出以便调试
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
        return [] # 如果 API 请求失败，直接返回空列表
    
    try:
        response_data = resp.json()
    except JSONDecodeError:
        return [] # 如果返回非 JSON，也返回空列表

    out = response_data["choices"][0]["message"]["content"].strip()
    # 重新字面检查 regex 模式，确保无隐形字符
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\\[.*\\\]", out, re.DOTALL)
    if not match:
        return []
    arr = json.loads(match.group(0))
    return [s.strip() for s in arr if isinstance(s, str)]

# 节次精确时间（分钟）
SECTION_TIME = {
    1: (480, 525), 2: (530, 575), 3: (590, 635), 4: (640, 685), 5: (690, 735),
    6: (845, 890), 7: (895, 940), 8: (945, 990), 9: (1000, 1045), 10: (1050, 1095),
    11: (1110, 1155), 12: (1160, 1205), 13: (1210, 1255)
}

def calc_course_conflict(ai_course_data):
    conflict = set()
    for item in ai_course_data:
        w = str(item.get("week", "")).strip().replace("星期", "周")
        if w not in WEEK_NAMES:
            continue
        secs = item.get("sections", [])
        if not isinstance(secs, list):
            continue
        for s in secs:
            if not isinstance(s, int) or s not in SECTION_TIME:
                continue
            c_start = SECTION_TIME[s][0]
            if c_start < 13 * 60:
                period = "上午"
            elif c_start < 17 * 60:
                period = "下午"
            else:
                period = "晚上"
            conflict.add(f"{w}{period}")
    return sorted(list(conflict))

def fast_ai_extract(text, personal_need=""):
    if not text or len(text.strip()) < 10:
        return "课表文本提取失败，请手动填写"
    
    # 检查 AI Key 是否可用
    if not VOLC_API_KEY:
        return "AI 识别功能不可用：缺少火山方舟 API Key 配置。"

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
        
        valid_weeks = set(WEEK_NAMES)
        cleaned = {}
        for entry in ai_course_data:
            if not isinstance(entry, dict):
                continue
            w = str(entry.get("week", "")).strip().replace("星期", "周")
            if w not in valid_weeks:
                continue
            secs = [x for x in entry.get("sections", []) if isinstance(x, int) and 1 <= x <= 13]
            if secs:
                cleaned.setdefault(w, []).extend(secs)
        ai_course_data = [{"week": w, "sections": sorted(set(secs))} for w, secs in cleaned.items()]
        
        # 仅在 Streamlit 运行时显示 DEBUG 信息
        if st.runtime.exists():
            with st.expander("🔍 DEBUG AI读取原始课表(仅管理员批量上传可见)"):
                st.code(json.dumps(ai_course_data, ensure_ascii=False, indent=2), language="json")
                
        course_conflict = calc_course_conflict(ai_course_data)
    except Exception as e:
        # 统一捕获并返回错误信息
        return f"课表识别失败:{str(e)[:60]}"

    personal_conflict = []
    if personal_need and personal_need.strip() not in ("", "无"):
        prompt_personal = f"课表冲突：{json.dumps(course_conflict, ensure_ascii=False)}\n用户要求：{personal_need}\n只输出额外禁止时段"
        try:
            personal_conflict = _call_personal_ai(prompt_personal)
        except Exception:
            personal_conflict = []
    total = list(set(course_conflict + personal_conflict))
    return "、".join(sorted(total)) if total else "识别为空，请手动填写"

def is_available(member, week_idx, shift_idx):
    conflict = safe_str(member.get("class_conflict"))
    return not (conflict and conflict != "无" and f"{WEEK_NAMES[week_idx]}{SHIFT_PERIOD[shift_idx]}" in conflict)

# ========== 排班 ==========
def smart_schedule(member_list):
    stu_pool = [m for m in member_list if m.get("role") == "学员"]
    wei_pool = [m for m in member_list if m.get("role") == "职委"]
    st.info(f"📊 学员 {len(stu_pool)} 人，职委 {len(wei_pool)} 人")
    
    if not wei_pool: # 优化：如果没有职委，先给出警告
        st.error("⚠️ 警告：人员库里没有职委！请去「人员库管理」把需要当职委的人身份改成「职委」")
        
    random.shuffle(stu_pool)
    random.shuffle(wei_pool)
    
    stu_count = {id(m): 0 for m in stu_pool}
    wei_count = {id(m): 0 for m in wei_pool}
    schedule = []
    
    for w_idx, week in enumerate(WEEK_NAMES):
        for s_idx, shift in enumerate(SHIFT_LIST):
            slot = {
                "week_idx": w_idx, "shift_idx": s_idx,
                "week": week, "shift": shift["shift_name"], "学员": [], "职委": []
            }
            # 学员排班
            # 优先选择有空且排班次数少于1次的学员
            available_stu = [m for m in stu_pool if is_available(m, w_idx, s_idx) and stu_count[id(m)] < 1]
            # 如果不够，则从所有有空学员中选择（不考虑排班次数限制），并优先选排班最少的
            if len(available_stu) < shift["need_stu"]:
                all_available = [m for m in stu_pool if is_available(m, w_idx, s_idx)]
                all_available.sort(key=lambda m: stu_count[id(m)]) # 优先选排班少的
                # 将已选和新选合并去重
                available_stu = list(set(available_stu + all_available))
                
            chosen_stu = available_stu[:shift["need_stu"]]
            for m in chosen_stu:
                slot["学员"].append(m)
                stu_count[id(m)] += 1
            
            # 职委排班
            if wei_pool:
                available_wei = [m for m in wei_pool if is_available(m, w_idx, s_idx) and wei_count[id(m)] < 1]
                if not available_wei: # 如果没有未排班的职委，则从所有有空职委中选排班最少的
                    all_available_wei = [m for m in wei_pool if is_available(m, w_idx, s_idx)]
                    if all_available_wei:
                        all_available_wei.sort(key=lambda m: wei_count[id(m)])
                        available_wei = [all_available_wei[0]] # 选排班最少的那个
                    else: # 如果所有职委都有冲突，则强制选排班最少的（可能仍有冲突）
                        wei_pool.sort(key=lambda m: wei_count[id(m)])
                        available_wei = [wei_pool[0]]
                        
                chosen_wei = available_wei[:shift["need_wei"]]
                for m in chosen_wei:
                    slot["职委"].append(m)
                    wei_count[id(m)] += 1
            
            schedule.append(slot)
            
    rested = [m["name"] for m in stu_pool if stu_count[id(m)] == 0]
    if rested:
        st.info(f"ℹ️ 本周共 {len(rested)} 名学员轮空：{'、'.join(rested)}")
    if wei_pool:
        st.success(f"✅ 职委排班完成：共 {len(wei_pool)} 名职委，平均每人排 {sum(wei_count.values())/len(wei_pool):.1f} 班")
    return schedule

# ========== PDF 导出 ==========
def export_pdf_auto(schedule_data, template_name):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 14)
    c.drawCentredString(w/2, h-40, "武汉大学学生会值班表")
    y = h - 70
    if template_name == "武汉大学学生会水印模板":
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.setFont(FONT_NAME, 35)
        c.drawCentredString(w/2, h/2, "武汉大学学生会")
        c.setFillColorRGB(0, 0, 0)
    c.setFont(FONT_NAME, 9)
    for row in schedule_data:
        stu = ", ".join([f"{x['name']}({x.get('phone','')})" for x in row.get('学员', [])])
        wei = ", ".join([f"{x['name']}({x.get('phone','')})" for x in row.get('职委', [])])
        line = f"{row['week']} {row['shift']} | 学员：{stu} | 职委：{wei}"
        c.drawString(30, y, line)
        y -= 16
        if y < 40:
            c.showPage()
            y = h - 40
            c.setFont(FONT_NAME, 9)
    c.save()
    buf.seek(0)
    return buf

def fill_user_template_pdf(template_bytes, schedule_data, offset_x=0, offset_y=-30):
    doc = fitz.open(stream=template_bytes, filetype="pdf")
    if not doc or doc.page_count == 0:
        st.error("无法加载PDF模板或模板为空。")
        return None
        
    page = doc[0]
    # 调整文字插入位置的基准点，使其更居中
    start_x = 158 + offset_x
    start_y = 175 + offset_y
    
    font_title = 6
    font_name = 6
    font_phone = 4.5
    
    # 间距调整，确保文字不会重叠
    gap_title = 5
    gap_name = 4.5
    gap_phone = 4
    
    cell_w = 82
    cell_h = 115
    
    cell_dict = {(si, wi): {"stu": [], "wei": []} for si in range(3) for wi in range(7)}
    for entry in schedule_data:
        k = (entry["shift_idx"], entry["week_idx"])
        if k in cell_dict:
            for p in entry.get("学员", []):
                cell_dict[k]["stu"].append((p["name"], p.get("phone", "")))
            for p in entry.get("职委", []):
                cell_dict[k]["wei"].append((p["name"], p.get("phone", "")))
                
    # 动态计算每个单元格的文本块高度，并设置顶部的填充
    # 假设最少一个学员，一个职委。最多2学员1职委。
    # 这里需要更精确的计算，因为高度可能随着内容变化
    
    for si in range(3):
        for wi in range(7):
            cell_data = cell_dict[(si, wi)]
            stu_list = cell_data["stu"][:2] # 最多显示两名学员
            wei_list = cell_data["wei"][:1] # 最多显示一名职委
            
            pos_x = start_x + wi * cell_w + 5
            
            # 计算当前单元格内文字块的实际高度，用于垂直居中
            actual_block_height = 0
            if stu_list:
                actual_block_height += (font_title + gap_title) # "值班学员"
                for _, phone in stu_list:
                    actual_block_height += (font_name + gap_name) # 姓名
                    if phone: actual_block_height += (font_phone + gap_phone) # 电话
                    else: actual_block_height += gap_name # 无电话也预留一点
            if wei_list:
                actual_block_height += (font_title + gap_title) # "轮值职委"
                for _, phone in wei_list:
                    actual_block_height += (font_name + gap_name) # 姓名
                    if phone: actual_block_height += (font_phone + gap_phone) # 电话
                    else: actual_block_height += gap_name # 无电话也预留一点
            
            top_pad = (cell_h - actual_block_height) / 2 # 计算顶部填充，实现垂直居中
            if top_pad < 0: top_pad = 5 # 避免文字溢出到单元格上方

            cy = start_y + si * cell_h + top_pad # 计算该单元格的起始Y坐标
            
            if stu_list:
                page.insert_text((pos_x, cy), "值班学员", fontsize=font_title, fontname="china-s")
                cy += gap_title
                for name, phone in stu_list:
                    page.insert_text((pos_x, cy), name, fontsize=font_name, fontname="china-s")
                    cy += gap_name
                    if phone:
                        page.insert_text((pos_x, cy), str(phone), fontsize=font_phone, fontname="china-s")
                        cy += gap_phone
                    else:
                        cy += gap_name # 即使没有电话也预留间距
            
            if wei_list:
                page.insert_text((pos_x, cy), "轮值职委", fontsize=font_title, fontname="china-s")
                cy += gap_title
                for name, phone in wei_list:
                    page.insert_text((pos_x, cy), name, fontsize=font_name, fontname="china-s")
                    cy += gap_name
                    if phone:
                        page.insert_text((pos_x, cy), str(phone), fontsize=font_phone, fontname="china-s")
                        cy += gap_phone
                    else:
                        cy += gap_name # 即使没有电话也预留间距
    
    out_buf = BytesIO()
    doc.save(out_buf)
    doc.close()
    out_buf.seek(0)
    return out_buf


# ========== 获取页面参数（兼容旧版 Streamlit） ==========
# Streamlit 1.10.0 之前 st.query_params 不存在，使用 st.experimental_get_query_params
try:
    _query_params = st.query_params
except AttributeError:
    _query_params = st.experimental_get_query_params() # 尝试兼容老版本

# 统一处理 _query_params 可能返回列表或字符串的情况
def _get_query_param_value(params_dict, key, default=""):
    value = params_dict.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value

_is_self_page = (_get_query_param_value(_query_params, "page") == "self")


if _is_self_page:
    menu = "成员自助上传"
else:
    menu = st.sidebar.radio("导航", ["管理员后台", "成员自助上传"])

# 初始化 session_state，确保所有键都存在
for key, val in [("login", False), ("current_admin", ""), ("schedule_result", None),
                 ("user_template_bytes", None), ("temp_new_member", []),
                 ("name_to_pdf_text", {})]: # 移除了 temp_pdf_texts，因为其功能已被 name_to_pdf_text 取代
    if key not in st.session_state:
        st.session_state[key] = val

# ========== 主流程 ==========
if menu == "管理员后台":
    if not st.session_state.login:
        st.title("管理员登录")
        admin_user = st.text_input("管理员账号")
        pwd = st.text_input("密码", type="password")
        if st.button("登录"):
            info = verify_admin(admin_user, pwd)
            if info:
                st.session_state.login = True
                st.session_state.current_admin = admin_user
                st.rerun()
            else:
                st.error("账号或密码错误（默认账号：admin，密码：123456）")
    else:
        st.title(f"管理员后台（当前登录：{st.session_state.current_admin}）")
        
        # 简化 Tab 结构，避免过多的嵌套和复杂逻辑
        if is_super_admin(st.session_state.current_admin):
            tab_names = ["批量上传课表", "人员库管理", "自动排班", "导出PDF", "管理员管理"]
            tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names) # 重新编号为5个tab
            tab_admin = tab5 # 将管理员管理指定给tab5
        else:
            tab_names = ["批量上传课表", "人员库管理", "自动排班", "导出PDF"]
            tab1, tab2, tab3, tab4 = st.tabs(tab_names)
            tab_admin = None # 非超级管理员不显示管理员管理tab

        # Tab1: 批量上传课表
        with tab1:
            st.subheader("批量上传PDF课表（文件名格式：身份&姓名&手机号-课表.pdf）")
            files = st.file_uploader("选择课表PDF", type="pdf", accept_multiple_files=True)
            if files and st.button("📂 第一步：本地识别姓名、身份、电话"):
                new_members_extracted = []
                name_to_pdf_text_map = {}
                
                for f in files:
                    pdf_text = get_pdf_full_text(f.read())
                    member_name = extract_name_from_text(pdf_text)
                    
                    if member_name:
                        role, phone = parse_filename_phone_and_role(f.name)
                        # 为每个有效成员存储其数据和对应的PDF文本
                        new_members_extracted.append({
                            "name": member_name,
                            "phone": phone,
                            "role": role,
                            "class_conflict": "",
                            "personal_need": ""
                        })
                        name_to_pdf_text_map[member_name] = pdf_text # 建立姓名到PDF文本的映射

                # 对提取到的成员列表去重，保留第一个
                seen_names = set()
                unique_members_list = []
                for member_data in new_members_extracted:
                    if member_data["name"] not in seen_names:
                        unique_members_list.append(member_data)
                        seen_names.add(member_data["name"])

                st.session_state.temp_new_member = unique_members_list
                st.session_state.name_to_pdf_text = name_to_pdf_text_map
                
                st.success(f"本地识别完成，共 {len(unique_members_list)} 人！请在表格中填写个性化需求后，点击第二步让 AI 综合识别。")
                st.rerun()

            if st.session_state.temp_new_member:
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
                        need_ai_tasks = []
                        
                        for row in current_rows:
                            member_name = safe_str(row.get("name"))
                            # 只有当 class_conflict 为空，且有对应的 PDF 文本，才需要 AI 识别
                            if not safe_str(row.get("class_conflict")) and member_name in st.session_state.name_to_pdf_text:
                                pdf_text_for_ai = st.session_state.name_to_pdf_text[member_name]
                                personal_need_for_ai = safe_str(row.get("personal_need"))
                                need_ai_tasks.append((member_name, pdf_text_for_ai, personal_need_for_ai))
                        
                        if need_ai_tasks:
                            if not VOLC_API_KEY:
                                st.error("AI 密钥未配置，无法进行 AI 识别。请联系管理员。")
                            else:
                                with st.spinner(f"AI 正在综合分析 {len(need_ai_tasks)} 人的课表和个性化需求..."):
                                    bar = st.progress(0)
                                    results_from_ai = {}
                                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                        futures = {
                                            executor.submit(fast_ai_extract, pdf_txt, p_need): name 
                                            for name, pdf_txt, p_need in need_ai_tasks
                                        }
                                        done_count = 0
                                        for future in as_completed(futures):
                                            member_name_from_future = futures[future]
                                            try:
                                                results_from_ai[member_name_from_future] = future.result()
                                            except Exception as e:
                                                results_from_ai[member_name_from_future] = f"识别失败: {str(e)[:60]}"
                                            done_count += 1
                                            bar.progress(done_count / len(need_ai_tasks))
                                
                                # 更新预览表格中的 class_conflict 字段
                                for row_data in current_rows:
                                    name_in_row = safe_str(row_data.get("name"))
                                    if name_in_row in results_from_ai:
                                        row_data["class_conflict"] = results_from_ai[name_in_row]
                                    
                                    # 确保所有字段都是字符串类型
                                    row_data["personal_need"] = safe_str(row_data.get("personal_need"))
                                    row_data["name"] = safe_str(row_data.get("name"))
                                    row_data["phone"] = safe_str(row_data.get("phone"))
                                    row_data["role"] = safe_str(row_data.get("role")) if safe_str(row_data.get("role")) else "学员"

                                st.session_state.temp_new_member = current_rows
                                st.success("AI 综合识别完成！课表冲突和个性化需求已合并计算。")
                                st.rerun()
                        else:
                            st.info("所有人都已识别，或无有效数据需要 AI 处理！")
                with col_b:
                    if st.button("✅ 确认保存入库"):
                        all_members_in_db = load_members()
                        existing_names = {m["name"] for m in all_members_in_db}
                        rows_to_add = edited_temp.to_dict("records")
                        new_members_to_save = []
                        for row_to_save in rows_to_add:
                            name_to_check = safe_str(row_to_save.get("name"))
                            if name_to_check and name_to_check not in existing_names:
                                # 确保所有字段都安全转换为字符串
                                row_to_save["personal_need"] = safe_str(row_to_save.get("personal_need"))
                                row_to_save["class_conflict"] = safe_str(row_to_save.get("class_conflict"))
                                row_to_save["role"] = safe_str(row_to_save.get("role")) if safe_str(row_to_save.get("role")) else "学员"
                                new_members_to_save.append(row_to_save)
                        all_members_in_db.extend(new_members_to_save)
                        save_members(all_members_in_db)
                        st.success(f"成功录入 {len(new_members_to_save)} 人！")
                        st.session_state.temp_new_member = []
                        st.session_state.name_to_pdf_text = {} # 清空不再需要的映射
                        st.rerun()
                with col_c:
                    if st.button("❌ 放弃本批"):
                        st.session_state.temp_new_member = []
                        st.session_state.name_to_pdf_text = {}
                        st.rerun()
        
        # Tab2: 人员库管理
        with tab2:
            st.subheader(f"人员库管理（共 {len(load_members())} 人）")
            members_data = load_members()
            df_members = pd.DataFrame(members_data)
            if not df_members.empty:
                df_members.insert(0, "序号", range(1, len(df_members) + 1))
            
            edit_config = {
                "序号": st.column_config.NumberColumn("序号", disabled=True),
                "role": st.column_config.SelectboxColumn("身份", options=["学员", "职委"], required=True),
                "class_conflict": st.column_config.TextColumn("不能值班时段（课表+需求综合）"),
                "personal_need": st.column_config.TextColumn("个性化需求原文")
            }
            edited_df_members = st.data_editor(
                df_members, use_container_width=True, num_rows="dynamic", column_config=edit_config
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾保存修改"):
                    with st.spinner("正在保存人员数据..."):
                        saved_records = edited_df_members.drop(columns=["序号"], errors="ignore").to_dict("records")
                        for record in saved_records:
                            record["personal_need"] = safe_str(record.get("personal_need"))
                            record["class_conflict"] = safe_str(record.get("class_conflict"))
                            record["role"] = safe_str(record.get("role")) if safe_str(record.get("role")) else "学员"
                        save_members(saved_records)
                    st.success("已保存人员库！")
                    st.rerun()
            with col2:
                if st.button("🗑️一键清空全部人员"):
                    save_members([])
                    st.warning("人员库已清空！")
                    st.rerun()
        
        # Tab3: 自动排班
        with tab3:
            st.subheader("一键排班（AI综合考虑课表冲突+个性化需求+职委均匀轮班）")
            if st.button("生成值班表"):
                with st.spinner("正在智能排班，综合考虑课表冲突和个性化需求..."):
                    schedule_result_data = smart_schedule(load_members())
                    st.session_state.schedule_result = schedule_result_data
                st.success("排班完成！")
            if st.session_state.schedule_result:
                st.dataframe(pd.DataFrame(st.session_state.schedule_result), use_container_width=True)
        
        # Tab4: 导出PDF
        with tab4:
            st.subheader("导出PDF（两种模式任选）")
            mode_selection = st.radio("选择导出模式", ["模式A：系统自动生成值班表", "模式B：上传我的空白模板PDF，自动填充"])
            
            if mode_selection == "模式A：系统自动生成值班表":
                pdf_template_selection = st.selectbox("PDF样式", TEMPLATE_OPTIONS)
                if not st.session_state.schedule_result:
                    st.warning("请先生成排班结果")
                else:
                    if st.button("导出自动生成的值班表PDF"):
                        with st.spinner("正在生成PDF..."):
                            pdf_buffer = export_pdf_auto(st.session_state.schedule_result, pdf_template_selection)
                        st.download_button("下载值班表", pdf_buffer, file_name="值班表.pdf", mime="application/pdf")
            else: # 模式B
                st.info("上传你自己的空白值班表PDF模板，系统将自动填充排班信息。")
                user_uploaded_template = st.file_uploader("上传你的空白值班表PDF模板", type="pdf")
                if user_uploaded_template:
                    st.session_state.user_template_bytes = user_uploaded_template.read()
                    st.success("模板已加载！")
                
                offset_x_input = st.number_input("横向微调偏移（像素）", value=0, help="正值向右，负值向左")
                offset_y_input = st.number_input("纵向微调偏移（像素）", value=-30, help="正值向下，负值向上")
                
                if not st.session_state.schedule_result:
                    st.warning("请先生成排班结果")
                elif not st.session_state.user_template_bytes:
                    st.warning("请先上传你的空白模板PDF")
                else:
                    if st.button("自动填充我的模板并导出"):
                        with st.spinner("正在填充你的模板PDF..."):
                            output_pdf_buffer = fill_user_template_pdf(
                                st.session_state.user_template_bytes,
                                st.session_state.schedule_result,
                                int(offset_x_input), int(offset_y_input)
                            )
                        if output_pdf_buffer: # 确保 PDF 成功生成才提供下载
                            st.download_button("下载填充好的值班表", output_pdf_buffer, file_name="值班表_已填充.pdf", mime="application/pdf")
                        else:
                            st.error("填充模板失败，请检查模板文件或联系技术支持。")
        
        # Tab5: 管理员管理 (仅超级管理员可见)
        if tab_admin is not None:
            with tab_admin: # 使用指定的 tab_admin 变量
                st.subheader("管理员管理（仅超级管理员可见）")
                
                st.markdown("#### 修改超级管理员账号")
                old_pwd_super = st.text_input("超级管理员原密码", type="password", key="old_pwd_super")
                new_username_super = st.text_input("新管理员账号（留空则不修改）", value="", key="new_username_super")
                new_pwd_super = st.text_input("新密码（留空则不修改）", type="password", key="new_pwd_super")
                confirm_pwd_super = st.text_input("确认新密码", type="password", key="confirm_pwd_super")
                if st.button("保存超级管理员修改"):
                    admin_data_current = load_admin()
                    super_admin_entry = None
                    for a_entry in admin_data_current["admins"]:
                        if a_entry.get("is_super", False):
                            super_admin_entry = a_entry
                            break
                    if not super_admin_entry:
                        st.error("未找到超级管理员账号。")
                    elif super_admin_entry["password"] != old_pwd_super:
                        st.error("原密码错误。")
                    elif new_pwd_super and new_pwd_super != confirm_pwd_super:
                        st.error("两次新密码不一致。")
                    else:
                        if new_username_super:
                            super_admin_entry["username"] = new_username_super.strip()
                        if new_pwd_super:
                            super_admin_entry["password"] = new_pwd_super
                        save_admin(admin_data_current)
                        if new_username_super:
                            st.session_state.current_admin = new_username_super.strip()
                        st.success("超级管理员账号信息已更新！")
                        st.rerun()
                
                st.divider()
                st.markdown("#### 修改当前登录账号密码")
                old_pwd_self = st.text_input("原密码", type="password", key="old_pwd_self")
                new_pwd_self = st.text_input("新密码", type="password", key="new_pwd_self")
                confirm_pwd_self = st.text_input("确认新密码", type="password", key="confirm_pwd_self")
                if st.button("修改我的密码"):
                    if new_pwd_self != confirm_pwd_self:
                        st.error("两次新密码不一致。")
                    elif len(new_pwd_self) < 6:
                        st.error("新密码至少需要6位。")
                    else:
                        admin_data_current = load_admin()
                        for a_entry in admin_data_current["admins"]:
                            if a_entry["username"] == st.session_state.current_admin and a_entry["password"] == old_pwd_self:
                                a_entry["password"] = new_pwd_self
                                save_admin(admin_data_current)
                                st.success("密码修改成功！")
                                break
                        else:
                            st.error("原密码错误。")

                st.divider()
                st.markdown("#### 添加新管理员")
                new_admin_user_input = st.text_input("新管理员账号", key="new_admin_user_input")
                new_admin_pwd_input = st.text_input("新管理员密码", type="password", key="new_admin_pwd_input")
                if st.button("添加管理员"):
                    admin_data_current = load_admin()
                    exist_admin = any(a_entry["username"] == new_admin_user_input for a_entry in admin_data_current["admins"])
                    if exist_admin:
                        st.error(f"管理员账号 {new_admin_user_input} 已存在。")
                    elif not new_admin_user_input or not new_admin_pwd_input:
                        st.warning("账号和密码不能为空。")
                    else:
                        admin_data_current["admins"].append({"username": new_admin_user_input, "password": new_admin_pwd_input, "is_super": False})
                        save_admin(admin_data_current)
                        st.success(f"管理员 {new_admin_user_input} 添加成功！")
                        st.rerun()

                st.divider()
                st.markdown("#### 管理员列表")
                admin_data_current = load_admin()
                df_admin_list = pd.DataFrame(admin_data_current["admins"])
                df_admin_list["角色"] = df_admin_list["is_super"].apply(lambda x: "超级管理员" if x else "普通管理员")
                st.dataframe(df_admin_list[["username","角色"]], use_container_width=True)
                
                st.divider()
                st.markdown("#### 删除管理员")
                normal_admins_to_delete = [a_entry["username"] for a_entry in admin_data_current["admins"] if not a_entry.get("is_super", False)]
                if normal_admins_to_delete:
                    delete_user_selection = st.selectbox("选择要删除的普通管理员", normal_admins_to_delete)
                    if st.button("删除该管理员"):
                        admin_data_current["admins"] = [a_entry for a_entry in admin_data_current["admins"] if a_entry["username"] != delete_user_selection]
                        save_admin(admin_data_current)
                        st.success(f"管理员 {delete_user_selection} 已删除！")
                        st.rerun()
                else:
                    st.info("暂无普通管理员可删除。")
        
        # 管理员页面底部的退出登录按钮
        st.divider()
        if st.button("🚪 退出登录"):
            st.session_state.login = False
            st.session_state.current_admin = ""
            st.rerun()

elif menu == "成员自助上传":
    st.title("成员自助提交信息 & 上传课表")
    with st.form("member_self_submit_form"):
        name_input = st.text_input("姓名")
        phone_input = st.text_input("联系电话")
        role_selection = st.radio("身份", ["学员","职委"])
        personal_need_input = st.text_input("个性化需求（选填，例如：周末不值班 / 只在周六下午有空 / 周一上午有课）")
        pdf_file_uploader = st.file_uploader("上传你的课表PDF", type="pdf")
        
        submitted_button = st.form_submit_button("提交")
        
        if submitted_button:
            if not (name_input and phone_input and pdf_file_uploader):
                st.warning("姓名、联系电话和课表 PDF 都需要填写！")
            else:
                if not VOLC_API_KEY:
                    st.error("AI 识别功能不可用：火山方舟 API Key 未配置。请联系管理员。")
                else:
                    with st.spinner("AI 正在综合分析你的课表和个性化需求..."):
                        pdf_raw_text = get_pdf_full_text(pdf_file_uploader.read())
                        ai_class_conflict = fast_ai_extract(pdf_raw_text, personal_need_input)
                        sanitized_personal_need = safe_str(personal_need_input)
                        
                        all_members_data = load_members()
                        member_found_and_updated = False
                        for member_entry in all_members_data:
                            if member_entry["name"] == name_input:
                                member_entry["phone"] = phone_input
                                member_entry["role"] = role_selection
                                member_entry["class_conflict"] = ai_class_conflict
                                member_entry["personal_need"] = sanitized_personal_need
                                member_found_and_updated = True
                                break
                        
                        if not member_found_and_updated:
                            all_members_data.append({
                                "name": name_input,
                                "phone": phone_input,
                                "role": role_selection,
                                "class_conflict": ai_class_conflict,
                                "personal_need": sanitized_personal_need
                            })
                        
                        save_members(all_members_data)
                    st.success("提交成功！AI 已综合识别你所有不能值班的时段。")
