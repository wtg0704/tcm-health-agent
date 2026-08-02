"""Streamlit前端 - 中医AI健康Agent系统 侧边栏常驻修复版"""
import streamlit as st
import requests
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# ==================== 配置 ====================
BACKEND_URL = "http://localhost:8000"
USER_FILE = os.path.join(os.path.dirname(__file__), "..", "user_data.json")

st.set_page_config(
    page_title="中医AI健康Agent系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# CSS：固定侧边栏 + 隐藏多余元素 + 响应式适配 + 紧凑布局
st.markdown("""
<style>
    /* === 隐藏Streamlit默认UI === */
    [data-testid="stToolbar"] {display: none !important;}
    #MainMenu {display: none !important;}
    footer {display: none !important;}

    /* === 侧边栏：永久展开、禁止折叠 === */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        min-width: 260px !important;
        max-width: 280px !important;
    }
    section[data-testid="stSidebar"] {
        transform: none !important;
        transition: none !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapsedControl"],
    button[kind="header"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* === 侧边栏紧凑布局 - 关键修改区域 === */
    /* 1. 减小侧边栏内部边距 */
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* 2. 减小block容器的上下边距 */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.3rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    
    /* 3. 减小各元素的间距 */
    section[data-testid="stSidebar"] .stMarkdown {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    section[data-testid="stSidebar"] .stButton {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    section[data-testid="stSidebar"] .stDivider {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* 4. 减小标题和文本间距 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    section[data-testid="stSidebar"] p {
        margin-top: 0.05rem !important;
        margin-bottom: 0.05rem !important;
    }
    
    /* 5. 减小caption间距 */
    section[data-testid="stSidebar"] .stCaption {
        margin-top: 0.05rem !important;
        margin-bottom: 0.05rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 6. 减小表单元素间距 */
    section[data-testid="stSidebar"] .stTextInput {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    section[data-testid="stSidebar"] .stForm {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* 7. 减小divider线条上下间距 */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* === 按钮圆角 === */
    .stButton > button {border-radius: 8px;}

    /* === 响应式：主区域自适应剩余宽度 === */
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        padding: 2rem 3rem !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    .block-container {
        max-width: 100% !important;
        padding: 2rem 3rem !important;
    }

    /* === 小屏幕(<1024px)：减小内边距 === */
    @media screen and (max-width: 1024px) {
        [data-testid="stSidebar"] {
            min-width: 220px !important;
            max-width: 240px !important;
        }
        .block-container {
            padding: 1rem 1.5rem !important;
        }
    }

    /* === 大屏幕(>1600px)：居中主内容 === */
    @media screen and (min-width: 1600px) {
        .block-container {
            padding: 2.5rem 5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== 数据持久化 ====================
def load_user_id():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("user_id")
        except: pass
    return None

def save_user_id(uid):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump({"user_id": uid}, f)

# ==================== Session初始化 ====================
def init_session():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    if "user_id" not in st.session_state:
        saved = load_user_id()
        st.session_state.user_id = saved
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "constitution_type" not in st.session_state:
        st.session_state.constitution_type = None
    if "user_basic_info" not in st.session_state:
        st.session_state.user_basic_info = {}

init_session()

# ==================== API ====================
def api_call(method, path, data=None):
    url = f"{BACKEND_URL}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, params=data, timeout=30)
        else:
            params = {}
            if data and "user_id" in data:
                params["user_id"] = data["user_id"]
            resp = requests.post(url, json=data, params=params, timeout=60)
        return resp.json() if resp.status_code == 200 else None
    except: return None


# ==================== 可视化 ====================
def render_radar_chart(labels, values):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill='toself', fillcolor='rgba(76, 175, 80, 0.3)',
        line=dict(color='#4CAF50', width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=350,
        margin=dict(l=40, r=40, t=10, b=10),
    )
    return fig

def render_bmi_gauge(bmi, bmi_level):
    color = "#4CAF50" if bmi_level == "正常" else "#FF9800" if bmi_level in ("偏瘦","偏胖") else "#F44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi, title={"text": f"BMI · {bmi_level}"},
        gauge={
            "axis": {"range": [10, 40]}, "bar": {"color": color},
            "steps": [
                {"range": [10, 18.5], "color": "#FFF3E0"},
                {"range": [18.5, 24], "color": "#E8F5E9"},
                {"range": [24, 28], "color": "#FFF3E0"},
                {"range": [28, 40], "color": "#FFEBEE"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig

# ==================== 体质百科数据 ====================
CONSTITUTION_DATA = {
    "平和质": {"emoji":"😊","feature":"阴阳气血调和，体态适中，面色润泽，精力充沛",
               "risk":"较少生病，适应能力强","diet":"营养均衡，顺应四时","exercise":"适度运动","tea":"春茉莉、夏菊花、秋桂花、冬玫瑰"},
    "气虚质": {"emoji":"😮‍💨","feature":"元气不足，疲乏气短，语音低弱，易出虚汗",
               "risk":"易感冒，病后恢复慢","diet":"山药、红枣、小米、鸡肉","exercise":"太极拳、八段锦","tea":"黄芪红枣茶"},
    "阳虚质": {"emoji":"🥶","feature":"阳气不足，畏寒怕冷，手足不温，喜热饮食",
               "risk":"易患水肿、腹泻","diet":"羊肉、韭菜、生姜、核桃","exercise":"阳光下户外活动","tea":"姜枣红糖茶"},
    "阴虚质": {"emoji":"🥵","feature":"阴液亏少，口燥咽干，手足心热，失眠多梦",
               "risk":"易患干咳、便秘","diet":"百合、银耳、梨、鸭肉","exercise":"游泳、瑜伽","tea":"枸杞菊花茶"},
    "痰湿质": {"emoji":"😪","feature":"痰湿凝聚，形体肥胖，腹部肥满，口黏苔腻",
               "risk":"易患高血压、高血脂","diet":"薏米、冬瓜、赤小豆","exercise":"跑步、跳绳有氧","tea":"陈皮薏米茶"},
    "湿热质": {"emoji":"😤","feature":"湿热内蕴，面垢油光，口苦口干，大便黏滞",
               "risk":"易患痤疮、黄疸","diet":"绿豆、苦瓜、黄瓜","exercise":"大强度排汗运动","tea":"绿豆薏米茶"},
    "血瘀质": {"emoji":"😟","feature":"血行不畅，肤色晦暗，易出瘀斑，舌质紫暗",
               "risk":"易患心脑血管病","diet":"山楂、黑豆、茄子","exercise":"避免久坐，常活动","tea":"山楂玫瑰茶"},
    "气郁质": {"emoji":"😔","feature":"气机郁滞，神情抑郁，烦闷不乐，胸胁胀痛",
               "risk":"易患失眠、抑郁","diet":"玫瑰花、柑橘、香菜","exercise":"集体活动、户外","tea":"玫瑰花茶"},
    "特禀质": {"emoji":"🤧","feature":"先天失常，常有过敏反应，对花粉药物敏感",
               "risk":"过敏性疾病","diet":"清淡均衡，避过敏原","exercise":"适度增强免疫","tea":"黄芪防风茶"},
}

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 🌿 中医AI健康Agent系统")
    st.divider()

    # 新对话
    if st.button("➕ 新对话", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_page = "chat"
        st.rerun()
    st.divider()

    # 功能导航
    st.markdown("#### 📄 功能")
    pages = [
        ("💬 智能问答", "chat"),
        ("📊 体质辨识", "constitution"),
        ("🏠 健康画像", "health_profile"),
        ("📖 体质百科", "encyclopedia"),
    ]
    for name, key in pages:
        t = "primary" if st.session_state.current_page == key else "secondary"
        if st.button(name, type=t, use_container_width=True):
            st.session_state.current_page = key
            st.rerun()
    st.divider()

    # 历史对话
    st.markdown("#### 💬 历史对话")
    uid = st.session_state.user_id
    if uid:
        raw = api_call("GET", "/api/chat/history", {"user_id": uid, "limit": 20})
        if raw and raw.get("messages"):
            msgs = [m for m in raw["messages"] if m["role"] == "user"]
            shown = set()
            count = 0
            for m in reversed(msgs):
                txt = m["message"]
                preview = txt[:20] + "..." if len(txt) > 20 else txt
                if preview not in shown and count < 12:
                    shown.add(preview); count += 1
                    if st.button(f"📝 {preview}", key=f"hist_{count}", use_container_width=True):
                        st.session_state.current_page = "chat"
                        st.rerun()
        else:
            st.caption("暂无对话记录")
    else:
        st.caption("登录后查看历史")
    st.divider()

    # 个人中心
    st.markdown("#### 👤 个人中心")
    uid = st.session_state.user_id
    if uid:
        u = api_call("GET", f"/api/users/{uid}")
        if u:
            st.caption(f"账号：{u.get('name', uid[:8])}")
            ct = st.session_state.constitution_type or u.get("constitution_type")
            if ct:
                emoji = CONSTITUTION_DATA.get(ct, {}).get("emoji", "")
                st.caption(f"体质：{emoji} {ct}")
            bi = st.session_state.user_basic_info
            if bi.get("bmi_val"):
                st.caption(f"BMI：{bi['bmi_val']:.1f}（{bi['bmi_level']}）")
        if st.button("退出登录", use_container_width=True):
            for k in ["user_id","chat_history","constitution_type","user_basic_info"]:
                st.session_state[k] = None if k == "user_id" else ([] if k == "chat_history" else ({} if k == "user_basic_info" else None))
            if os.path.exists(USER_FILE): os.remove(USER_FILE)
            st.rerun()
    else:
        with st.form("create_user_sidebar"):
            name = st.text_input("昵称（选填）")
            if st.form_submit_button("创建账号", use_container_width=True):
                res = api_call("POST", "/api/users", {"name": name or None})
                if res:
                    st.session_state.user_id = res["id"]
                    save_user_id(res["id"])
                    st.rerun()
    st.caption("\n⚠️ 养生参考，非医疗诊断")


# ==================== 主区域 ====================
uid = st.session_state.user_id

if not uid:
    st.markdown("""
    <div style="text-align:center; padding:100px 20px;">
        <h1>🌿 中医AI健康Agent系统</h1>
        <p style="color:#666; font-size:18px;">基于 RAG + LangGraph Agent 的中医养生知识助手</p>
        <p style="color:#999;">👈 请在左侧创建账号，解锁全部功能</p>
    </div>
    """, unsafe_allow_html=True)

else:
    page = st.session_state.current_page

    # ==================== 智能问答 ====================
    if page == "chat":
        if len(st.session_state.chat_history) == 0:
            st.markdown("""
            <div style="text-align:center; padding:50px 20px 30px 20px;">
                <h2>🌿 有什么可以帮您的？</h2>
                <p style="color:#999;">可以问我中药功效、方剂组成、食疗方案，或先完成体质辨识获得个性化建议</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**💡 试试这些问题：**")
            qs = [
                "黄芪有什么功效？", "气虚质适合喝什么茶？",
                "红豆薏米粥怎么做？", "四君子汤由哪几味药组成？",
                "什么是湿气重？", "失眠有什么食疗推荐？",
            ]
            cols = st.columns(3)
            for i, q in enumerate(qs):
                with cols[i % 3]:
                    if st.button(q, key=f"quick_{i}", use_container_width=True):
                        st.session_state.chat_history.append({"role": "user", "message": q})
                        with st.spinner("思考中..."):
                            resp = api_call("POST", "/api/chat/query", {
                                "user_id": uid, "message": q, "conversation_history": []
                            })
                        if resp:
                            st.session_state.chat_history.append({
                                "role": "assistant", "answer": resp["answer"],
                                "sources": resp.get("sources", []),
                            })
                        st.rerun()
        else:
            st.caption("💬 智能问答")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["message"])
            else:
                with st.chat_message("assistant", avatar="🌿"):
                    st.write(msg["answer"])
                    if msg.get("sources"):
                        with st.expander("📚 参考来源"):
                            for s in msg["sources"]:
                                st.caption(f"**{s['doc']}**：{s['excerpt'][:200]}...")

        st.divider()
        inp = st.chat_input("输入您的中医养生问题...")
        if inp:
            st.session_state.chat_history.append({"role": "user", "message": inp})
            api_hist = [
                {"role": h["role"], "message": h["message"] if h["role"] == "user" else h["answer"]}
                for h in st.session_state.chat_history[-7:-1]
            ]
            with st.spinner("思考中..."):
                resp = api_call("POST", "/api/chat/query", {
                    "user_id": uid, "message": inp, "conversation_history": api_hist
                })
            if resp:
                st.session_state.chat_history.append({
                    "role": "assistant", "answer": resp["answer"],
                    "sources": resp.get("sources", []),
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "answer": "抱歉，服务暂不可用。请确认后端已启动且API Key已配置。"
                })
            st.rerun()

    # ==================== 体质辨识 ====================
    elif page == "constitution":
        st.header("📊 中医体质辨识问卷")
        st.caption("依据《中医体质分类与判定》标准，共9题，请根据近一年体验作答")
        st.divider()

        qs = api_call("GET", "/api/constitution/questions")
        if qs:
            scores = {}
            for q in qs.get("questions", []):
                scores[f"q{q['id']}"] = st.radio(
                    q["text"], options=[1,2,3,4,5],
                    format_func=lambda x: {1:"1-从不",2:"2-很少",3:"3-有时",4:"4-经常",5:"5-总是"}[x],
                    horizontal=True, key=f"cq_{q['id']}"
                )
            if st.button("提交测评，查看体质报告", type="primary", use_container_width=True):
                ans = [{"question_id": i+1, "score": scores[f"q{i+1}"]} for i in range(9)]
                res = api_call("POST", "/api/constitution/assess", {"user_id": uid, "answers": ans})
                if res:
                    st.success(f"### {res['constitution_type']}")
                    st.info(res["type_description"])
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.plotly_chart(render_radar_chart(res["radar_data"]["labels"], res["radar_data"]["values"]), use_container_width=True)
                    with c2:
                        st.subheader("养生方向建议")
                        st.success(res["health_tips"])
                    st.session_state.constitution_type = res["constitution_type"]

    # ==================== 健康画像 ====================
    elif page == "health_profile":
        st.header("🏠 个人健康画像")
        st.divider()
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("基础健康信息")
            with st.form("profile_form"):
                h = st.number_input("身高(cm)", 50.0, 300.0, 170.0, 0.5)
                w = st.number_input("体重(kg)", 10.0, 500.0, 65.0, 0.5)
                sl = st.slider("睡眠质量", 1, 5, 3, help="1差→5好")
                sp = st.number_input("每周运动次数", 0, 14, 2)
                if st.form_submit_button("保存画像", use_container_width=True, type="primary"):
                    api_call("POST", "/api/profile/update", {
                        "user_id": uid, "height_cm": h, "weight_kg": w,
                        "sleep_quality": sl, "exercise_frequency": sp,
                    })
                    bmi_v = w / ((h/100)**2)
                    if bmi_v < 18.5: lv = "偏瘦"
                    elif bmi_v < 24: lv = "正常"
                    elif bmi_v < 28: lv = "偏胖"
                    else: lv = "肥胖"
                    st.session_state.user_basic_info = {
                        "bmi_val": bmi_v, "bmi_level": lv,
                        "height": h, "weight": w, "sleep": sl, "sport": sp
                    }
                    st.success("已保存")
                    st.rerun()
        with c2:
            st.subheader("BMI体重指标")
            bi = st.session_state.user_basic_info
            if bi.get("bmi_val"):
                st.plotly_chart(render_bmi_gauge(bi["bmi_val"], bi["bmi_level"]), use_container_width=True)
            else:
                st.info("填写左侧信息并保存后查看")
        st.divider()
        st.subheader("💡 今日养生建议")
        ct = st.session_state.constitution_type
        if ct and bi:
            tips = {
                "气虚质": "早餐来碗红枣小米粥，午后散步补充阳气。避免过度劳累。",
                "阳虚质": "晨起喝杯姜枣茶，注意腰腹保暖。晒晒太阳补充阳气。",
                "阴虚质": "泡杯枸杞菊花茶，晚上11点前入睡。少吃辛辣燥热。",
                "痰湿质": "早餐薏米粥，饭后散步排湿。少喝冰饮控甜食。",
                "湿热质": "喝杯绿豆汤清热，晚餐清淡。傍晚跑步排汗。",
                "血瘀质": "山楂片泡水，每小时起身活动5分钟。保持心情舒畅。",
                "气郁质": "泡杯玫瑰花茶，听听音乐。约朋友散散心。",
                "特禀质": "注意避开过敏原，饮食清淡均衡。适当锻炼。",
                "平和质": "保持规律作息，营养均衡。今天也很健康！",
            }
            st.success(tips.get(ct, f"结合您的{ct}体质，保持规律作息和均衡饮食。"))
        else:
            st.info("完成体质测评和健康信息录入后，自动生成每日养生建议")

    # ==================== 体质百科 ====================
    elif page == "encyclopedia":
        st.header("📖 九种中医体质百科")
        st.caption("点击卡片查看每种体质的详细特征、易患疾病、饮食运动建议及推荐茶饮")
        st.divider()

        cols = st.columns(3)
        for i, (name, info) in enumerate(CONSTITUTION_DATA.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(f"{info['emoji']} {name}")
                    st.caption(f"**特征：** {info['feature']}")
                    st.caption(f"**易患：** {info['risk']}")
                    st.caption(f"**饮食：** {info['diet']}")
                    st.caption(f"**运动：** {info['exercise']}")
                    st.caption(f"**茶饮：** {info['tea']}")