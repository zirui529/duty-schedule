import streamlit as st
import json
import requests
import base64
import re
from typing import List, Dict, Any

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="课表排班系统", layout="wide")

# ===================== 服务商自动切换配置（从secrets读取） =====================
PROVIDER = st.secrets["provider"]

# 火山方舟配置
ARK_API_KEY = st.secrets["ark"]["api_key"]
ARK_ENDPOINT = st.secrets["ark"]["endpoint"]
ARK_MODEL = st.secrets["ark"]["model_name"]

# 阿里云DashScope配置
DS_API_KEY = st.secrets["dashscope"]["api_key"]
DS_ENDPOINT = st.secrets["dashscope"]["endpoint"]
DS_MODEL = st.secrets["dashscope"]["model_name"]

# 共用强约束提取提示词，两套模型共用
PROMPT_COURSE_EXTRACT = """
【铁律，违反视为无效输出】
1. 仅提取PDF内明确存在的课程，没有写出来的星期、节次一律禁止输出，严禁脑补、补全、推断、补齐周一到周五。
2. 当天无课，该星期条目直接不出现，不要返回空数组。
3. 禁止同一天生成超过10个节次，原文不存在的节次绝对不能编造。
4. 禁止跨星期拼接碎片、把其他日期课程混入当日。
5. 输出严格JSON，无任何解释、备注、多余文字。
固定结构：[{"week":"周一","sections":[1,2,3]},...]
week只允许：周一、周二、周三、周四、周五、周六、周日
sections是纯数字数组，和原文完全一致。
"""

# ===================== 两个模型独立调用函数 =====================
def call_ark_extract(pdf_bytes):
    b64_data = base64.b64encode(pdf_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": ARK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_COURSE_EXTRACT},
                    {"type": "file_url", "file_url": f"data:application/pdf;base64,{b64_data}"}
                ]
            }
        ]
    }
    resp = requests.post(ARK_ENDPOINT, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["message"]["content"]
    raw_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(raw_text)

def call_dashscope_extract(pdf_bytes):
    b64_data = base64.b64encode(pdf_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {DS_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DS_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_COURSE_EXTRACT},
                    {"type": "file", "file_url": f"data:application/pdf;base64,{b64_data}"}
                ]
            }
        ]
    }
    resp = requests.post(DS_ENDPOINT, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["message"]["content"]
    raw_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(raw_text)

# ===================== 统一入口，原有业务代码无需改动 =====================
def fast_ai_extract(pdf_bytes):
    if PROVIDER == "ark":
        return call_ark_extract(pdf_bytes)
    elif PROVIDER == "dashscope":
        return call_dashscope_extract(pdf_bytes)
    else:
        raise ValueError("provider配置错误，仅支持 ark / dashscope")

# ===================== 风险校验告警逻辑（原样保留） =====================
def check_course_risk(course_data: List[Dict[str, Any]]) -> List[str]:
    alerts = []
    day_sections_limit = 10
    jump_threshold = 4
    for day_item in course_data:
        w = day_item["week"]
        secs = sorted(day_item["sections"])
        if len(secs) > day_sections_limit:
            alerts.append(f"{w}识别到{len(secs)}个节次，数量异常，疑似幻觉")
        has_morning = any(1 <= s <= 4 for s in secs)
        has_afternoon = any(5 <= s <= 8 for s in secs)
        has_night = any(9 <= s <= 12 for s in secs)
        if has_morning and has_afternoon and has_night:
            alerts.append(f"{w}同时出现上午、下午、晚上全套节次，高度怀疑跨星期碎片幻觉")
        for i in range(1, len(secs)):
            diff = secs[i] - secs[i-1]
            if diff > jump_threshold:
                alerts.append(f"{w}节次跳跃过大 {secs[i-1]} → {secs[i]}，疑似幻觉")
    return alerts

# ===================== 下面是你原有全部业务逻辑，无修改 =====================
# 存储初始化
if "submit_queue" not in st.session_state:
    st.session_state["submit_queue"] = []
if "member_lib" not in st.session_state:
    st.session_state["member_lib"] = []

tab1, tab2 = st.tabs(["成员上传课表", "管理员审核后台"])

with tab1:
    st.subheader("提交课表PDF")
    st.info(f"当前使用模型服务商：{PROVIDER}")
    up_file = st.file_uploader("上传课表PDF", type="pdf")
    name = st.text_input("姓名")
    personal_req = st.text_area("个性化需求")
    if st.button("提交解析") and up_file and name:
        with st.spinner("正在解析PDF..."):
            pdf_bin = up_file.read()
            try:
                course_result = fast_ai_extract(pdf_bin)
                risk_list = check_course_risk(course_result)
                queue_item = {
                    "name": name,
                    "personal_need": personal_req,
                    "ai_course_json": course_result,
                    "risk_alerts": risk_list,
                    "status": "待审核",
                    "final_course": None
                }
                st.session_state["submit_queue"].append(queue_item)
                st.success("解析完成，已加入审核队列")
                st.json(course_result)
                if risk_list:
                    st.warning("风险告警列表：")
                    for a in risk_list:
                        st.markdown(f"- {a}")
                else:
                    st.success("✅ 校验通过，无风险")
            except Exception as e:
                st.error(f"解析失败：{str(e)}")

with tab2:
    st.subheader("管理员待审核队列")
    if len(st.session_state["submit_queue"]) == 0:
        st.info("暂无待审核记录")
    else:
        for idx, item in enumerate(st.session_state["submit_queue"]):
            with st.expander(f"{item['name']} | 状态：{item['status']}"):
                st.markdown(f"个性化需求：{item['personal_need']}")
                st.markdown("AI原始解析结果：")
                st.json(item["ai_course_json"])
                st.markdown("风险告警：")
                if item["risk_alerts"]:
                    for alert in item["risk_alerts"]:
                        st.markdown(f"<div style='background:#fff9cc;padding:8px;border-radius:6px;margin:4px 0'>{alert}</div>", unsafe_allow_html=True)
                else:
                    st.success("无风险告警")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ 采用AI结果入库", key=f"use_{idx}"):
                        item["final_course"] = item["ai_course_json"]
                        item["status"] = "已采用"
                        st.session_state["member_lib"].append(item)
                        st.rerun()
                with col2:
                    if st.button("✏️ 人工修正入库", key=f"edit_{idx}"):
                        item["status"] = "待人工修改"
                        st.rerun()
                with col3:
                    if st.button("❌ 退回重传", key=f"reject_{idx}"):
                        item["status"] = "已退回"
                        st.rerun()
