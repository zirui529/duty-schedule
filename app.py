import streamlit as st
import re
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pdfplumber
import fitz
import dashscope
from openai import OpenAI
import numpy as np
from datetime import datetime

# ====================== 页面基础配置 ======================
st.set_page_config(page_title="武大课表智能排班", layout="wide")

# ====================== Secrets 读取 ======================
try:
    qwen_api_key = st.secrets["qwen"]["api_key"]
    volc_api_key = st.secrets["volc"]["api_key"]
    volc_endpoint_id = st.secrets["volc"]["endpoint_id"]
    admin_pwd = st.secrets["auth"]["admin_pwd"]
except Exception as e:
    st.error("secrets读取失败，请检查配置")
    st.stop()

dashscope.api_key = qwen_api_key
# 火山方舟 OpenAI兼容客户端（不再依赖volcenginesdkarkruntime包）
volc_client = OpenAI(
    api_key=volc_api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# ====================== 全局状态初始化 ======================
if "login" not in st.session_state:
    st.session_state.login = False
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "member_list" not in st.session_state:
    st.session_state.member_list = []
if "schedule_result" not in st.session_state:
    st.session_state.schedule_result = None
if "history" not in st.session_state:
    st.session_state.history = []

# ====================== 提示词【千问：解析规整Markdown课表表格】 ======================
EXTRACT_SYS_PROMPT = """
你现在处理一份武汉大学教务课表，已经被整理成固定13节×7天的Markdown表格，空白单元格代表无课。
任务：提取所有有课程的【星期 + 节次】。

硬性铁律：
1. 单元格里面有文字才算有课；空白单元格直接跳过，禁止编造课程。
2. week只能是：["周一","周二","周三","周四","周五","周六","周日"]
3. sections只能是1~13的整数。
4. 输出JSON数组，示例：[{"week":"周一","sections":[1,2,3]},{"week":"周六","sections":[11]}]
5. 如果全部空白，直接返回 []
6. 只输出纯净JSON，不要markdown、注释、额外文字。

节次对照：
1:08:00‑08:45｜2:08:50‑09:35｜3:10:35‑11:20｜4:11:25‑12:10
5:14:00‑14:45｜6:14:50‑15:35｜7:15:40‑16:25｜8:16:30‑17:15
9:17:20‑18:05｜10:18:10‑18:55｜11:19:00‑19:45｜12:19:50‑20:35
13:20:40‑21:25
"""

# ====================== 提示词【火山豆包：个性化需求识别&冲突补充】 ======================
DEMAND_SYS_PROMPT = """
你负责解析用户个性化的不可用时间需求，输出格式和课表结果保持一致。
规则：
1. 用户说哪些时段不能排班，就提取哪些时段。输出结构示例：[{"week":"周一","sections":[1,2]}]
2. 只提取用户明确说明的时段，禁止脑补、额外增加时间。
3. 用户没有额外要求，返回空数组 []
4. 只输出纯JSON，不要多余文字。
"""

# ====================== 核心预处理函数：PDF → 自适应包围盒 + 规整等格Markdown表格 ======================
def get_pdf_full_text(pdf_bytes):
    row_count = 13
    col_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    col_count = len(col_names)
    grid = [["" for _ in range(col_count)] for _ in range(row_count)]
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words or len(words) < 5:
                    continue
                all_y = [w["top"] for w in words]
                ys = np.array(all_y)
                y_low = np.percentile(ys, 10)
                y_high = np.percentile(ys, 90)
                table_words = [w for w in words if (y_low <= w["top"] <= y_high)]
                if len(table_words) < 3:
                    table_words = words

                table_x = [w["x0"] for w in table_words]
                table_y = [w["top"] for w in table_words]
                x_min, x_max = min(table_x), max(table_x)
                y_min, y_max = min(table_y), max(table_y)

                cell_w = (x_max - x_min) / col_count
                cell_h = (y_max - y_min) / row_count

                for w in table_words:
                    wx = w["x0"]
                    wy = w["top"]
                    rel_x = wx - x_min
                    rel_y = wy - y_min
                    col_idx = int(rel_x / cell_w)
                    row_idx = int(rel_y / cell_h)
                    if 0 <= col_idx < col_count and 0 <= row_idx < row_count:
                        grid[row_idx][col_idx] += w["text"] + " "
    except Exception:
        doc = fitz.open("pdf", pdf_bytes)
        raw = ""
        for p in doc:
            raw += p.get_text()
        return raw

    md_lines = ["| 节次 | " + " | ".join(col_names) + " |",
                "| ---- | " + " | ".join(["----"] * col_count) + " |"]
    for r in range(row_count):
        line = f"| {r+1} | " + " | ".join(grid[r]) + " |"
        md_lines.append(line)
    table_text = "\n".join(md_lines)
    table_text = re.sub(r"打印时间[:：].*", "", table_text)
    table_text = re.sub(r"学号[:：].*", "", table_text)
    return table_text

# ====================== AI调用函数：千问解析课表 ======================
def call_qwen_parse_timetable(table_md):
    resp = dashscope.Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": EXTRACT_SYS_PROMPT},
            {"role": "user", "content": table_md}
        ],
        result_format="message"
    )
    content = resp.output.choices[0].message.content
    return json.loads(content)

# ====================== AI调用函数：火山解析个性化需求（OpenAI兼容版） ======================
def call_volc_parse_demand(demand_text):
    if not demand_text.strip():
        return []
    resp = volc_client.chat.completions.create(
        model=volc_endpoint_id,
        messages=[
            {"role": "system", "content": DEMAND_SYS_PROMPT},
            {"role": "user", "content": demand_text}
        ],
        temperature=0.1
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# ====================== 冲突合并工具 ======================
def merge_conflict_periods(timetable_data, demand_data):
    busy_map = {}
    def add_item(item):
        wk = item["week"]
        secs = item["sections"]
        if wk not in busy_map:
            busy_map[wk] = set()
        for s in secs:
            busy_map[wk].add(s)
    for d in timetable_data:
        add_item(d)
    for d in demand_data:
        add_item(d)
    res = []
    for week, sec_set in busy_map.items():
        res.append({"week": week, "sections": sorted(list(sec_set))})
    return res

# ====================== 排班冲突检测 ======================
def check_schedule_conflict(member_busy_list, assign_plan):
    conflict_list = []
    busy_dict = {}
    for mb in member_busy_list:
        name = mb["name"]
        busy_dict[name] = {}
        for item in mb["busy"]:
            busy_dict[name][item["week"]] = set(item["sections"])
    for task in assign_plan:
        wk = task["week"]
        sec = task["section"]
        person = task["person"]
        if person not in busy_dict:
            continue
        if wk in busy_dict[person] and sec in busy_dict[person][wk]:
            conflict_list.append(f"{person} 在 {wk} 第{sec}节有占用，排班冲突")
    return conflict_list

# ====================== 导出图片+武大水印 ======================
def export_schedule_image(plan):
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 16)
        title_font = ImageFont.truetype("simhei.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    draw.text((width-220, height-40), "武汉大学 | 学生会排班", font=font, fill="#cccccc")
    y = 40
    draw.text((40, y), "排班结果表", font=title_font, fill="#000000")
    y += 40
    for item in plan:
        line = f"{item['week']} 第{item['section']}节 → {item['person']}"
        draw.text((40,y), line, font=font, fill="#222222")
        y +=28
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ====================== 登录界面 ======================
if not st.session_state.login:
    st.title("🔐 武大智能排班系统 - 登录")
    pwd_input = st.text_input("管理员密码", type="password")
    if st.button("登录"):
        if pwd_input == admin_pwd:
            st.session_state.login = True
            st.session_state.is_admin = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()

# ====================== 主Tab界面 ======================
tab1, tab2, tab3, tab4 = st.tabs(["📥 课表识别录入", "👥 成员管理", "📅 排班生成", "📜 历史记录"])

# Tab1 课表识别
with tab1:
    st.subheader("上传PDF课表，自动提取不可用时段")
    name_input = st.text_input("成员姓名")
    upload_file = st.file_uploader("上传该成员课表PDF", type="pdf")
    demand_input = st.text_area("个性化额外避坑需求", placeholder="例：周一晚上有事")
    run_btn = st.button("识别课表并生成busy时段")
    if upload_file and run_btn and name_input.strip():
        pdf_bytes = upload_file.read()
        with st.expander("DEBUG调试面板｜规整后Markdown课表【复核用】", expanded=True):
            md_table = get_pdf_full_text(pdf_bytes)
            st.code(md_table, language="markdown")
        try:
            st.info("千问正在解析课表...")
            timetable_result = call_qwen_parse_timetable(md_table)
            st.info("火山豆包正在解析个性化需求...")
            demand_result = call_volc_parse_demand(demand_input)
            final_busy = merge_conflict_periods(timetable_result, demand_result)
            st.success(f"✅ {name_input} 的占用时段识别完成！")
            st.json(final_busy)
            st.session_state.member_list.append({"name": name_input, "busy": final_busy})
        except Exception as err:
            st.error(f"识别失败：{str(err)}")

# Tab2 成员管理
with tab2:
    st.subheader("已录入成员列表")
    st.json(st.session_state.member_list)
    if st.button("清空全部成员"):
        st.session_state.member_list = []
        st.rerun()

# Tab3 排班、冲突检测、导出
with tab3:
    st.subheader("排班任务录入 & 冲突校验")
    raw_plan_text = st.text_area("输入排班计划JSON", placeholder='[{"week":"周一","section":1,"person":"张三"}]')
    col1, col2 = st.columns(2)
    with col1:
        if st.button("校验冲突"):
            try:
                plan = json.loads(raw_plan_text)
                conflicts = check_schedule_conflict(st.session_state.member_list, plan)
                if conflicts:
                    st.error("发现冲突：")
                    for c in conflicts:
                        st.write("- " + c)
                else:
                    st.success("✅ 无冲突，可以排班")
                    st.session_state.schedule_result = plan
            except Exception as e:
                st.error(f"JSON解析错误：{e}")
    with col2:
        if st.button("导出排班图片（带武大水印）"):
            if st.session_state.schedule_result:
                img_buf = export_schedule_image(st.session_state.schedule_result)
                st.download_button("下载PNG排班表", img_buf, file_name="排班表.png", mime="image/png")
                st.session_state.history.append({"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "plan": st.session_state.schedule_result})
            else:
                st.warning("请先校验并生成排班结果")

# Tab4 历史记录
with tab4:
    st.subheader("历史排班记录")
    for h in st.session_state.history:
        st.markdown(f"**{h['time']}**")
        st.json(h["plan"])
