import streamlit as st
import requests
import json
import re
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ===================== 基础配置 =====================
st.set_page_config(page_title="排班系统", layout="wide")
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

# 读取密钥
DS_API_KEY = st.secrets["dashscope"]["api_key"]
DS_ENDPOINT = st.secrets["dashscope"]["endpoint"]
DS_MODEL_NAME = st.secrets["dashscope"]["model_name"]

# ===================== 提示词【固定，禁止AI脑补时段】 =====================
EXTRACT_SYS_PROMPT = """
你是PDF课表提取器，严格只提取文档里写明的课程，禁止脑补、禁止额外添加任何不存在课程。
输出硬性规则：
1. 只输出JSON数组，数组内每个课程对象三个字段：weekday（周一/周二…周日），start（起始节次，数字），end（结束节次，数字）
2. 只识别文档里真实存在的课，没有写的课绝对不能加。
3. 一节课跨多节，填正确start、end，不要拆成多条。
4. 不输出上午、下午、晚上，不做任何排班判断，不写解释文字。
5. 禁止markdown、禁止```包裹，直接输出json。

正确示例：
[{"weekday":"周三","start":1,"end":2},{"weekday":"周五","start":5,"end":6}]
严禁输出任何时段文本（周三上午这类一律不允许）
"""

# ===================== 千问调用函数 =====================
def _call_course_ai(prompt_text:str):
    sess = requests.Session()
    headers = {
        "Authorization": f"Bearer {DS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DS_MODEL_NAME,
        "messages": [
            {"role":"system","content":EXTRACT_SYS_PROMPT},
            {"role":"user","content":prompt_text}
        ],
        "temperature": 0
    }
    resp = sess.post(DS_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"].strip()
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        raise ValueError("千问课表解析：未读取到合法课程JSON")
    course_raw = json.loads(match.group(0))
    valid_week = {"周一","周二","周三","周四","周五","周六","周日"}
    clean_course = []
    for item in course_raw:
        wd = item.get("weekday")
        s = item.get("start")
        e = item.get("end")
        if wd in valid_week and isinstance(s, int) and isinstance(e, int) and s <= e:
            clean_course.append({"weekday":wd, "start":s, "end":e})
    return clean_course

# ===================== 本地转换：节次映射时段 =====================
def course_to_slot(course_list):
    slot_set = set()
    slot_map = [
        ("上午", 1, 4),
        ("下午", 5, 8),
        ("晚上", 9, 12),
    ]
    for c in course_list:
        wd = c["weekday"]
        cs = c["start"]
        ce = c["end"]
        for slot_name, low, high in slot_map:
            if not (ce < low or cs > high):
                slot_set.add(f"{wd}{slot_name}")
    return sorted(list(slot_set))

# ===================== 个性化需求文本解析（原有逻辑保留） =====================
def parse_personal_text(text:str):
    if not text.strip():
        return []
    items = [i.strip() for i in re.split(r"[，,、\n]", text.strip()) if i.strip()]
    valid_slots = {
        "周一上午","周一下午","周一晚上",
        "周二上午","周二下午","周二晚上",
        "周三上午","周三下午","周三晚上",
        "周四上午","周四下午","周四晚上",
        "周五上午","周五下午","周五晚上",
        "周六上午","周六下午","周六晚上",
        "周日上午","周日下午","周日晚上"
    }
    return [x for x in items if x in valid_slots]

# ===================== 【你原有排班核心函数，原样保留】 =====================
def schedule_core(people_list, forbid_dict):
    week_slots = [
        "周一上午","周一下午","周一晚上",
        "周二上午","周二下午","周二晚上",
        "周三上午","周三下午","周三晚上",
        "周四上午","周四下午","周四晚上",
        "周五上午","周五下午","周五晚上",
        "周六上午","周六下午","周六晚上",
        "周日上午","周日下午","周日晚上"
    ]
    result = {}
    import random
    random.seed(42)
    for slot in week_slots:
        can = [p for p in people_list if slot not in forbid_dict.get(p, [])]
        if can:
            result[slot] = random.choice(can)
        else:
            result[slot] = "无可用人员"
    return result

# ===================== 页面主体 =====================
def main():
    st.title("智能排班系统")
    tab1, tab2, tab3, tab4 = st.tabs(["人员录入", "课表解析", "排班预览", "导出PDF"])

    with tab1:
        st.subheader("录入人员名单")
        people_text = st.text_area("每行一个名字", height=200)
        if people_text.strip():
            people = [x.strip() for x in people_text.splitlines() if x.strip()]
            st.session_state["people"] = people
            st.success(f"已录入 {len(people)} 人")
        else:
            st.session_state["people"] = []

    with tab2:
        st.subheader("上传PDF课表 + 个性化禁止时段")
        uploaded_pdf = st.file_uploader("上传课表PDF（支持多页/两张表）", type="pdf")
        personal_input = st.text_area("个性化禁止时段（逗号分隔，例：周六晚上、周日上午）", height=120)
        run_btn = st.button("开始AI解析课表")

        if run_btn and uploaded_pdf:
            with st.spinner("千问正在提取课表..."):
                # 这里是PDF文本提取，你原有逻辑，这里简化占位；如果你自己有pdfplumber提取文本直接替换
                import pdfplumber
                full_text = ""
                with pdfplumber.open(uploaded_pdf) as pdf:
                    for page in pdf.pages:
                        page_txt = page.extract_text() or ""
                        full_text += page_txt + "\n"
                try:
                    raw_courses = _call_course_ai(full_text)
                    course_slots = course_to_slot(raw_courses)
                    personal_slots = parse_personal_text(personal_input)
                    final_forbid = list(set(course_slots + personal_slots))
                    st.session_state["current_forbid"] = final_forbid
                    st.session_state["raw_course_data"] = raw_courses
                    st.success("解析完成！")
                    st.write("原始课程（周几+节次）：", raw_courses)
                    st.write("合并后禁止值班时段：", final_forbid)
                except Exception as e:
                    st.error(f"解析失败：{str(e)}")

    with tab3:
        st.subheader("排班预览")
        if "people" not in st.session_state or len(st.session_state["people"]) == 0:
            st.warning("先去【人员录入】添加人员")
            return
        if "current_forbid" not in st.session_state:
            st.warning("先去【课表解析】跑一次识别")
            return
        forbid_all = {}
        for p in st.session_state["people"]:
            forbid_all[p] = st.session_state["current_forbid"]
        schedule_res = schedule_core(st.session_state["people"], forbid_all)
        df = pd.DataFrame(list(schedule_res.items()), columns=["时段", "值班人"])
        st.dataframe(df, use_container_width=True)
        st.session_state["schedule_df"] = df

    with tab4:
        st.subheader("导出排班PDF")
        if "schedule_df" not in st.session_state:
            st.warning("先完成排班预览")
            return
        df = st.session_state["schedule_df"]
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setFont("STSong-Light", 12)
        y = 800
        c.drawString(220, y, "排班表")
        y -= 30
        for _, row in df.iterrows():
            line = f"{row['时段']} —— {row['值班人']}"
            c.drawString(60, y, line)
            y -= 22
            if y < 50:
                c.showPage()
                c.setFont("STSong-Light", 12)
                y = 800
        c.save()
        buf.seek(0)
        st.download_button("下载PDF", data=buf, file_name="排班结果.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
