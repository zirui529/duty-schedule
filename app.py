import streamlit as st
import re
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pdfplumber
import fitz
from openai import OpenAI

# ===================== Secrets读取 【严格匹配你后台配置】 =====================
try:
    dashscope_api_key = st.secrets["dashscope"]["api_key"]
    dashscope_endpoint = st.secrets["dashscope"]["endpoint"]
    dashscope_model = st.secrets["dashscope"]["model_name"]

    ark_api_key = st.secrets["ark"]["api_key"]
    ark_endpoint = st.secrets["ark"]["endpoint"]
    ark_model = st.secrets["ark"]["model_name"]

    admin_pwd = st.secrets["auth"]["admin_pwd"]
except Exception as e:
    st.error(f"secrets读取失败，详细错误：{str(e)}")
    st.stop()

# 初始化两个独立AI客户端
client_qwen = OpenAI(
    api_key=dashscope_api_key,
    base_url=dashscope_endpoint
)
client_ark = OpenAI(
    api_key=ark_api_key,
    base_url=ark_endpoint
)

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="武大学生会智能排班系统", layout="wide")

# ===================== Session 状态初始化 =====================
if "login" not in st.session_state:
    st.session_state.login = False
if "member_list" not in st.session_state:
    st.session_state.member_list = []
if "schedule_result" not in st.session_state:
    st.session_state.schedule_result = None
if "history" not in st.session_state:
    st.session_state.history = []

# ===================== Prompt 优化，抑制幻觉，提升识别准确率 =====================
# 千问：PDF课表提取专用
PROMPT_QWEN_EXTRACT = """
你是课表提取专家，输入是武汉大学PDF课表解析出来的文本/表格内容。
任务：只提取所有【有课占用、不能排班】的时间段，严格只输出JSON数组，禁止任何多余文字、解释、备注。
规则：
1. week 取值：周一 / 周二 / 周三 / 周四 / 周五 / 周六 / 周日
2. section：武大课节，数字1~13
3. 空白无课格子直接跳过，不要输出
4. 样例输出：[{"week":"周一","section":[1,2]},{"week":"周三","section":[5,6]}]
5. 绝对禁止编造不存在的课程时段，拿不准的直接忽略，不要猜测！
"""
# 火山豆包：解析个性化需求、补充禁排时段
PROMPT_ARK_DEMAND = """
你负责解析用户排班的个性化约束需求，补充【禁止排班】时段。
输入：已经存在的成员课表占用时段 + 用户文字需求。
只识别用户明确写明的时间限制，禁止联想、脑补、额外增加限制。
输出格式和课表一致：[{"week":"周一","section":[3,4]}]，无额外约束直接返回 []。
只返回JSON，不要任何说明文字。
"""

# ===================== 底层工具函数 =====================
def parse_pdf_to_text(pdf_bytes):
    """读取PDF，优先提取表格，失败则提取纯文本"""
    all_content = ""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    for row in table:
                        row = [cell if cell is not None else "" for cell in row]
                        all_content += " | ".join(row) + "\n"
                else:
                    page_text = page.extract_text()
                    if page_text:
                        all_content += page_text + "\n"
    except Exception:
        pass
    # pdfplumber失败，回退fitz
    if not all_content.strip():
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            all_content += page.get_text()
    return all_content

def call_qwen_extract_timetable(pdf_text):
    """调用千问，提取课表禁排时段"""
    resp = client_qwen.chat.completions.create(
        model=dashscope_model,
        messages=[
            {"role": "system", "content": PROMPT_QWEN_EXTRACT},
            {"role": "user", "content": pdf_text}
        ],
        temperature=0.02
    )
    raw_res = resp.choices[0].message.content.strip()
    return json.loads(raw_res)

def call_ark_parse_requirement(requirement, exist_block_list):
    """调用火山ark豆包，解析自定义需求补充禁排时段"""
    user_msg = f"已有占用时段：{json.dumps(exist_block_list)}\n用户需求：{requirement}"
    resp = client_ark.chat.completions.create(
        model=ark_model,
        messages=[
            {"role": "system", "content": PROMPT_ARK_DEMAND},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.02
    )
    raw_res = resp.choices[0].message.content.strip()
    return json.loads(raw_res)

def merge_block_times(list_a, list_b):
    """合并两组禁排时段，去重"""
    week_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    temp = {}
    for item in list_a + list_b:
        w = item["week"]
        secs = item["section"]
        if w not in temp:
            temp[w] = set()
        for s in secs:
            temp[w].add(int(s))
    out = []
    for w in week_order:
        if w in temp and len(temp[w]) > 0:
            out.append({"week": w, "section": sorted(list(temp[w]))})
    return out

def detect_conflict(member_all, schedule_plan):
    """冲突检测：检查排班计划和所有人禁排时段是否重叠"""
    conflict_msg = []
    for plan_item in schedule_plan:
        p_week, p_sec = plan_item["week"], plan_item["section"]
        for member in member_all:
            for block in member["block"]:
                if block["week"] == p_week:
                    overlap = set(p_sec) & set(block["section"])
                    if overlap:
                        conflict_msg.append(f"{member['name']}：{p_week} 第{list(overlap)}节 时段冲突")
    return conflict_msg

def generate_schedule_image(schedule_data):
    """排班结果导出图片"""
    width, height = 1200, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("simhei.ttf", 28)
        font_text = ImageFont.truetype("simhei.ttf", 19)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
    draw.text((40, 25), "武大学生会排班结果表", font=font_title, fill="#222222")
    y = 80
    for item in schedule_data:
        txt = f"{item['week']}  第 {item['section']} 节，安排：{item['person']}"
        draw.text((40, y), txt, font=font_text, fill="#333333")
        y += 36
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ===================== 登录页面 =====================
if not st.session_state.login:
    st.title("🔐 武大学生会排班系统 - 登录")
    pwd_input = st.text_input("管理员密码", type="password")
    if st.button("登录"):
        if pwd_input == admin_pwd:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("密码错误！")
    st.stop()

# ===================== 主页面（登录成功后） =====================
st.title("📅 武大学生会智能排班系统")
tab1, tab2, tab3, tab4 = st.tabs(["成员管理", "上传课表+录入需求", "排班生成 & 冲突检查", "历史记录"])

# Tab1：成员管理
with tab1:
    st.subheader("添加成员")
    name_in = st.text_input("成员姓名")
    if st.button("添加成员") and name_in.strip():
        st.session_state.member_list.append({"name": name_in.strip(), "block": []})
        st.success(f"已添加：{name_in.strip()}")
        st.rerun()
    st.divider()
    st.subheader("现有成员列表")
    for idx, mem in enumerate(st.session_state.member_list):
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.write(f"✅ {mem['name']}  |  禁排时段：{mem['block']}")
        with col2:
            if st.button(f"删除 {mem['name']}", key=f"del_{idx}"):
                del st.session_state.member_list[idx]
                st.rerun()

# Tab2：上传PDF课表 + 个性化需求
with tab2:
    st.subheader("上传成员课表PDF（千问负责识别解析）")
    select_mem = st.selectbox("选择对应成员", [x["name"] for x in st.session_state.member_list])
    upload_pdf = st.file_uploader("上传PDF课表", type="pdf")
    req_text = st.text_area("个性化额外禁排需求（火山豆包解析），如：周一下午2-4节没空")
    if st.button("解析课表 + 合并需求，保存禁排时段") and upload_pdf is not None:
        with st.spinner("千问正在读取PDF课表..."):
            pdf_bin = upload_pdf.read()
            pdf_txt = parse_pdf_to_text(pdf_bin)
            base_block = call_qwen_extract_timetable(pdf_txt)
        if req_text.strip():
            with st.spinner("豆包解析个性化约束..."):
                add_block = call_ark_parse_requirement(req_text.strip(), base_block)
                final_block = merge_block_times(base_block, add_block)
        else:
            final_block = base_block
        # 写入对应成员
        for m in st.session_state.member_list:
            if m["name"] == select_mem:
                m["block"] = final_block
                st.success(f"{select_mem} 的禁排时段已更新：{final_block}")
        st.rerun()

# Tab3：排班生成、冲突校验、图片导出
with tab3:
    st.subheader("输入本次排班计划")
    plan_input = st.text_area("输入排班安排，示例：周一1-2节安排张三，周三5-6节安排李四")
    if st.button("AI解析排班计划 + 检测冲突") and plan_input.strip():
        # 这里复用ark模型解析排班计划
        plan_prompt = """
        把排班文字转为JSON数组，字段 week、section、person，section是数字数组。
        样例：[{"week":"周一","section":[1,2],"person":"张三"}]，只输出JSON。
        """
        resp_plan = client_ark.chat.completions.create(
            model=ark_model,
            messages=[{"role":"system","content":plan_prompt},{"role":"user","content":plan_input}],
            temperature=0.02
        )
        schedule_plan = json.loads(resp_plan.choices[0].message.content.strip())
        st.session_state.schedule_result = schedule_plan
        conflict_list = detect_conflict(st.session_state.member_list, schedule_plan)
        st.divider()
        st.subheader("排班解析结果")
        st.json(schedule_plan)
        if conflict_list:
            st.error("⚠️ 检测到时段冲突：")
            for c in conflict_list:
                st.write(c)
        else:
            st.success("✅ 无冲突！排班可用")
            # 存入历史
            st.session_state.history.append({
                "time": st.query_params.get("now", ""),
                "plan": schedule_plan
            })
    if st.session_state.schedule_result:
        if st.button("导出排班PNG图片"):
            img_buf = generate_schedule_image(st.session_state.schedule_result)
            st.download_button("下载排班图", img_buf, file_name="排班结果.png", mime="image/png")

# Tab4：历史记录
with tab4:
    st.subheader("排班历史记录")
    for h in st.session_state.history:
        st.json(h["plan"])
        st.divider()
    if st.button("清空全部历史"):
        st.session_state.history = []
        st.rerun()
