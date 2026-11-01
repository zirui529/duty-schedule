import streamlit as st
import json
import re
import fitz
import requests
from typing import List, Dict
import time

# ===================== 配置（从secrets读取，不动） =====================
API_TIMEOUT = 60
try:
    DS_API_KEY = st.secrets["dashscope"]["api_key"]
    DS_ENDPOINT = st.secrets["dashscope"]["endpoint"]
    DS_MODEL = st.secrets["dashscope"]["model_name"]
except:
    DS_API_KEY = ""
    DS_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1"
    DS_MODEL = "qwen3.5"

EXTRACT_SYS_PROMPT = """
你是课表解析器，只输出标准JSON数组，禁止额外解释、禁止编造课程、禁止幻觉。
输入是课表文字，提取所有课程，每一条结构：
{"weekday": 周一/周二/周三/周四/周五/周六/周日, "start_time":"HH:MM", "end_time":"HH:MM"}
没有课就返回空数组[]，严格按照课表原文，不存在的时段不要自己生成。
"""

PERSONAL_SYS_PROMPT = """
根据用户个性化需求，输出不能值班的时间段JSON数组，格式同上。
只提取用户明确写出来的限制，不能自己脑补增加限制。
"""

# ===================== 工具函数 =====================
def _make_http_session():
    s = requests.Session()
    return s

def _call_course_ai(prompt_text:str):
    sess = _make_http_session()
    headers = {"Authorization": f"Bearer {DS_API_KEY}", "Content-Type": "application/json"}
    url = DS_ENDPOINT.rstrip("/") + "/chat/completions"
    payload = {
        "model": DS_MODEL,
        "messages": [
            {"role":"system","content":EXTRACT_SYS_PROMPT},
            {"role":"user","content":prompt_text}
        ],
        "temperature":0
    }
    resp = sess.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        raise ValueError("千问：模型没有返回合法JSON数组")
    return json.loads(match.group(0))

def _call_personal_ai(prompt_text:str):
    sess = _make_http_session()
    headers = {"Authorization": f"Bearer {DS_API_KEY}", "Content-Type": "application/json"}
    url = DS_ENDPOINT.rstrip("/") + "/chat/completions"
    payload = {
        "model": DS_MODEL,
        "messages": [
            {"role":"system","content":PERSONAL_SYS_PROMPT},
            {"role":"user","content":prompt_text}
        ],
        "temperature":0
    }
    resp = sess.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        raise ValueError("千问：模型没有返回合法JSON数组")
    return json.loads(match.group(0))

def pdf_extract_text(pdf_bytes):
    doc = fitz.open("pdf", pdf_bytes)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def time_cmp(t1:str, t2:str):
    h1,m1 = map(int, t1.split(":"))
    h2,m2 = map(int, t2.split(":"))
    return h1*60+m1 - (h2*60+m2)

def is_overlap(period_a:Dict, period_b:Dict) -> bool:
    if period_a["weekday"] != period_b["weekday"]:
        return False
    a_s, a_e = period_a["start_time"], period_a["end_time"]
    b_s, b_e = period_b["start_time"], period_b["end_time"]
    if time_cmp(a_e, b_s) <= 0 or time_cmp(b_e, a_s) <=0:
        return False
    return True

# ===================== 页面主体（原有全部保留） =====================
st.set_page_config(page_title="智能排班", layout="wide")
st.title("学生会智能排班系统")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["人员管理", "班次设置", "课表上传", "排班计算", "分享链接"])

with tab1:
    st.header("人员名单")
    if "member_list" not in st.session_state:
        st.session_state["member_list"] = []
    name = st.text_input("姓名")
    note = st.text_area("个性化不可值班备注")
    if st.button("添加人员"):
        st.session_state["member_list"].append({"name":name, "note":note, "block":[]})
    for idx,item in enumerate(st.session_state["member_list"]):
        col1,col2,col3 = st.columns([2,4,1])
        col1.write(item["name"])
        col2.write(item["note"])
        if col3.button("删除", key=f"del_{idx}"):
            del st.session_state["member_list"][idx]
            st.rerun()

with tab2:
    st.header("值班班次")
    if "shift_list" not in st.session_state:
        st.session_state["shift_list"] = []
    wd = st.selectbox("星期", ["周一","周二","周三","周四","周五","周六","周日"])
    s_t = st.text_input("开始时间 HH:MM", value="18:00")
    e_t = st.text_input("结束时间 HH:MM", value="20:00")
    need = st.number_input("需要人数", min_value=1, value=1)
    if st.button("新增班次"):
        st.session_state["shift_list"].append({"weekday":wd, "start_time":s_t, "end_time":e_t, "need":need})
    for idx,item in enumerate(st.session_state["shift_list"]):
        col1,col2,col3,col4 = st.columns([2,2,2,1])
        col1.write(item["weekday"])
        col2.write(f"{item['start_time']}-{item['end_time']}")
        col3.write(f"人数:{item['need']}")
        if col4.button("删班次", key=f"sd_{idx}"):
            del st.session_state["shift_list"][idx]
            st.rerun()

with tab3:
    st.header("上传课表PDF")
    upload_file = st.file_uploader("上传PDF课表，支持多页", type="pdf")
    if upload_file is not None:
        raw_txt = pdf_extract_text(upload_file.read())
        st.text_area("提取原文", raw_txt, height=150)
        if st.button("AI解析课表"):
            try:
                course_blocks = _call_course_ai(raw_txt)
                st.success("课表识别成功")
                st.json(course_blocks)
                st.session_state["course_blocks"] = course_blocks
            except Exception as e:
                st.error(f"课表识别失败:{str(e)}")
    if "course_blocks" in st.session_state:
        st.subheader("已加载禁止时段（课表）")
        st.json(st.session_state["course_blocks"])

with tab4:
    st.header("排班运算")
    if st.button("开始自动排班"):
        final_result = []
        for mem in st.session_state["member_list"]:
            per_block = []
            if mem["note"].strip():
                per_block = _call_personal_ai(mem["note"])
            all_block = []
            if "course_blocks" in st.session_state:
                all_block += st.session_state["course_blocks"]
            all_block += per_block
            mem_res = {"name":mem["name"], "forbidden":all_block, "assign":[]}
            for sh in st.session_state["shift_list"]:
                conflict = False
                for blk in all_block:
                    if is_overlap(sh, blk):
                        conflict=True
                        break
                if not conflict:
                    mem_res["assign"].append(sh)
            final_result.append(mem_res)
        st.subheader("AI识别不能值班时段（课表+需求综合）")
        st.table(final_result)
        st.session_state["last_result"] = final_result

with tab5:
    st.header("分享链接")
    base_url = st.query_params.get("base", st.experimental_get_query_params().get("base", [st.secrets.get("base_url", "")])[0])
    share_link = f"{base_url}"
    st.code(share_link)
    st.markdown("复制链接发给其他人打开")
