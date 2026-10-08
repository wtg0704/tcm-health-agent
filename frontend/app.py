"""Streamlit前端 - 中医AI健康Agent系统 侧边栏常驻修复版"""
import streamlit as st
import requests
import plotly.graph_objects as go
import json
import os
import re
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

    /* === 侧边栏 flex 布局：历史记录占剩余空间，个人中心贴底 === */
    section[data-testid="stSidebar"] .block-container {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 1rem) !important;
    }
    section[data-testid="stSidebar"] .block-container > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 !important;
    }

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
def _load_user_data():
    """从文件加载用户数据（user_id + 会话列表）"""
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def _save_user_data(data: dict):
    """保存用户数据到文件"""
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_user_id():
    return _load_user_data().get("user_id")

def save_user_id(uid):
    d = _load_user_data()
    d["user_id"] = uid
    _save_user_data(d)

def clear_user_id():
    """退出登录：只清 user_id，保留 chat_sessions 归档"""
    d = _load_user_data()
    d.pop("user_id", None)
    _save_user_data(d)

def load_chat_sessions(user_id=None):
    """加载某用户的已归档会话（按 user_id 隔离，换账号互不可见）"""
    if not user_id:
        return []
    return _load_user_data().get("chat_sessions_by_user", {}).get(user_id, [])

def save_chat_sessions(user_id, sessions):
    """持久化某用户的会话列表（按 user_id 隔离存储）"""
    d = _load_user_data()
    by_user = d.get("chat_sessions_by_user", {})
    by_user[user_id] = sessions
    d["chat_sessions_by_user"] = by_user
    _save_user_data(d)

# ==================== Session初始化 ====================
def init_session():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    if "user_id" not in st.session_state:
        saved = load_user_id()
        st.session_state.user_id = saved
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = load_chat_sessions(st.session_state.user_id)
    if "constitution_type" not in st.session_state:
        st.session_state.constitution_type = None
    if "user_basic_info" not in st.session_state:
        st.session_state.user_basic_info = {}

init_session()

# ==================== API ====================
@st.cache_data(ttl=600, show_spinner=False)
def _cached_get(path, params_json=""):
    """缓存 GET 请求——不变的数据不重复调 API"""
    import json as _json
    params = _json.loads(params_json) if params_json else None
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None


def api_call(method, path, data=None):
    """通用 API 调用，GET 请求走缓存"""
    if method == "GET":
        import json as _json
        return _cached_get(path, _json.dumps(data, sort_keys=True) if data else "")
    url = f"{BACKEND_URL}{path}"
    try:
        params = {}
        if data and "user_id" in data:
            params["user_id"] = data["user_id"]
        resp = requests.post(url, json=data, params=params, timeout=60)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None


def api_call_with_error(method, path, data=None):
    """带错误信息的 API 调用，返回 (ok, data_or_error)。登录/注册等需精确提示时用"""
    url = f"{BACKEND_URL}{path}"
    try:
        params = {}
        if data and "user_id" in data:
            params["user_id"] = data["user_id"]
        resp = requests.post(url, json=data, params=params, timeout=60)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            return False, resp.json().get("detail", f"请求失败({resp.status_code})")
        except Exception:
            return False, f"请求失败({resp.status_code})"
    except Exception as e:
        return False, str(e)


def _stream_tokens(user_id, message, history, metadata_container):
    """调用流式API，逐个token产出。metadata_container 用来回传 sources/agent_trace"""
    # 立即给出反馈——避免等待期间的"卡住了"错觉
    yield "⏳ 正在分析您的问题…\n\n"

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat/stream",
            json={
                "user_id": user_id,
                "message": message,
                "conversation_history": history,
            },
            stream=True,
            timeout=120,
        )
        if response.status_code != 200:
            yield f"抱歉，请求失败（{response.status_code}）"
            return

        for line in response.iter_lines():
            if line:
                line_str = line.decode()
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "token":
                        yield data["content"]
                    elif data.get("type") == "status":
                        # 阶段进度反馈（理解/检索），实时渲染，稍后从历史里剥离
                        yield data["content"]
                    elif data.get("type") == "done":
                        metadata_container["sources"] = data.get("sources", [])
                        metadata_container["agent_trace"] = data.get("agent_trace", {})
    except requests.exceptions.ConnectionError:
        yield "抱歉，无法连接后端服务。请确认后端已启动。"
    except Exception as e:
        yield f"抱歉，发生错误：{str(e)}"


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
        # 归档当前会话（非空才保存）
        if st.session_state.chat_history:
            first_user_msg = next(
                (m["message"] for m in st.session_state.chat_history if m["role"] == "user"),
                "空对话"
            )
            # 标题浓缩：去掉 [图片分析 · xx] 前缀，只留正文前 14 字
            first_user_msg = re.sub(r"^\[图片分析[^\]]*\]\s*", "", first_user_msg)
            session_title = first_user_msg[:14] + ("…" if len(first_user_msg) > 14 else "")
            st.session_state.chat_sessions.insert(0, {
                "title": session_title,
                "messages": list(st.session_state.chat_history),  # 深拷贝
            })
            save_chat_sessions(st.session_state.user_id, st.session_state.chat_sessions)
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
    ]
    for name, key in pages:
        t = "primary" if st.session_state.current_page == key else "secondary"
        if st.button(name, type=t, use_container_width=True):
            st.session_state.current_page = key
            st.rerun()
    st.divider()

    # 历史对话（平铺显示最近 6 条，不占大片空白；个人中心由下方占位贴底）
    st.markdown("#### 💬 历史对话")
    sessions = st.session_state.chat_sessions
    if sessions:
        for i, session in enumerate(sessions[:6]):
            preview = session["title"][:10] + ("…" if len(session["title"]) > 10 else "")
            if st.button(f"📝 {preview}", key=f"hist_{i}", use_container_width=True):
                st.session_state.chat_history = list(session["messages"])
                st.session_state.current_page = "chat"
                st.rerun()
        if len(sessions) > 6:
            st.caption(f"更早的 {len(sessions) - 6} 条对话已折叠")
    else:
        st.caption("暂无对话记录")

    # 弹性占位：把个人中心推到侧边栏最底部
    st.markdown('<div style="flex-grow: 1; min-height: 8px;"></div>', unsafe_allow_html=True)

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
            # 清 user_id + 当前会话视图（数据按 user_id 保留，重新登录可找回；换账号互不可见）
            st.session_state.user_id = None
            st.session_state.chat_history = []
            st.session_state.chat_sessions = []
            st.session_state.constitution_type = None
            st.session_state.user_basic_info = {}
            clear_user_id()
            st.rerun()
    else:
        auth_mode = st.radio(
            "", ["登录", "注册"], horizontal=True,
            label_visibility="collapsed", key="auth_mode",
        )
        if auth_mode == "登录":
            with st.form("login_form"):
                login_name = st.text_input("账号")
                login_pw = st.text_input("密码", type="password")
                if st.form_submit_button("登录", use_container_width=True):
                    if not login_name or not login_pw:
                        st.error("请输入账号和密码")
                    else:
                        ok, data = api_call_with_error(
                            "POST", "/api/auth/login",
                            {"name": login_name, "password": login_pw},
                        )
                        if ok:
                            st.session_state.user_id = data["id"]
                            save_user_id(data["id"])
                            st.session_state.constitution_type = data.get("constitution_type")
                            st.session_state.user_basic_info = {}
                            st.session_state.chat_sessions = load_chat_sessions(data["id"])
                            st.rerun()
                        else:
                            st.error(data)
        else:
            with st.form("register_form"):
                reg_name = st.text_input("昵称")
                reg_pw = st.text_input("密码", type="password")
                if st.form_submit_button("注册", use_container_width=True):
                    if not reg_name or not reg_pw:
                        st.error("请填写昵称和密码")
                    else:
                        ok, data = api_call_with_error(
                            "POST", "/api/users",
                            {"name": reg_name, "password": reg_pw},
                        )
                        if ok:
                            st.session_state.user_id = data["id"]
                            save_user_id(data["id"])
                            st.session_state.constitution_type = data.get("constitution_type")
                            st.session_state.user_basic_info = {}
                            st.session_state.chat_sessions = load_chat_sessions(data["id"])
                            st.rerun()
                        else:
                            st.error(data)
    st.caption("⚠️ 养生参考，非医疗诊断")


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

        # ===== 发送栏：文字 + 图片（舌诊/体检报告）融为一体 =====
        # 分析类型不设独立按钮，由文字自然判断：提到「报告/体检」→体检报告，否则默认舌诊
        result = st.chat_input(
            "输入问题，或点 📎 上传图片（舌象直接发；体检报告请说明「体检报告」）…",
            accept_file=True,
        )
        if result:
            # Streamlit 1.60 起 chat_input 返回 ChatInputValue 对象，需用属性访问而非解包
            prompt = result.text
            files = result.files

            # 有附图 → 视觉分析（prompt 作为补充说明 + 类型判断）
            if files:
                import base64
                uploaded = files[0]
                # st.chat_input 不支持按扩展名过滤，手动校验 MIME 类型
                if not (uploaded.type or "").startswith("image/"):
                    st.warning("请上传图片（JPG / PNG），其他文件类型暂不支持。")
                    st.rerun()
                b64 = base64.b64encode(uploaded.getvalue()).decode()
                data_url = f"data:{uploaded.type};base64,{b64}"

                extra = (prompt or "").strip()
                is_report = any(k in extra for k in ("报告", "体检", "化验"))
                task = "report" if is_report else "tongue"
                task_label = "体检报告" if is_report else "舌诊"
                display_msg = f"[图片分析 · {task_label}]"
                if extra:
                    display_msg += f" {extra}"
                st.session_state.chat_history.append({"role": "user", "message": display_msg})

                with st.spinner("AI 正在分析图片…"):
                    resp = api_call("POST", "/api/vision/analyze", {
                        "user_id": uid,
                        "image_data_url": data_url,
                        "task": task,
                        "message": extra,
                    })
                if resp:
                    st.session_state.chat_history.append({
                        "role": "assistant", "answer": resp["answer"], "sources": [],
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "answer": "图片分析失败：请确认后端已启动、`.env` 已配置 `DASHSCOPE_API_KEY`。",
                        "sources": [],
                    })
                st.rerun()

            # 纯文本 → 流式问答
            elif prompt:
                inp = prompt
                st.session_state.chat_history.append({"role": "user", "message": inp})

                api_hist = [
                    {"role": h["role"], "message": h["message"] if h["role"] == "user" else h["answer"]}
                    for h in st.session_state.chat_history[-7:-1]
                ]

                metadata = {}
                with st.chat_message("assistant", avatar="🌿"):
                    full_answer = st.write_stream(_stream_tokens(uid, inp, api_hist, metadata))

                thinking_prefix = "⏳ 正在分析您的问题…\n\n"
                if full_answer.startswith(thinking_prefix):
                    full_answer = full_answer[len(thinking_prefix):]
                # 剥离阶段进度占位（与后端 status 事件文案保持一致）
                for _s in ("🧠 正在理解您的问题…\n\n", "🔍 正在检索中医知识库…\n\n"):
                    full_answer = full_answer.replace(_s, "")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "answer": full_answer,
                    "sources": metadata.get("sources", []),
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
                    ct_name = res['constitution_type']
                    st.success(f"### {ct_name}")
                    st.info(res["type_description"])
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.plotly_chart(render_radar_chart(res["radar_data"]["labels"], res["radar_data"]["values"]), use_container_width=True)
                    with c2:
                        st.subheader("养生方向建议")
                        st.success(res["health_tips"])
                    # 体质详情卡片（原「体质百科」内容并入辨识结果）
                    info = CONSTITUTION_DATA.get(ct_name)
                    if info:
                        st.divider()
                        st.subheader(f"{info['emoji']} {ct_name} · 详细档案")
                        d1, d2 = st.columns(2)
                        with d1:
                            st.caption(f"**特征**：{info['feature']}")
                            st.caption(f"**易患**：{info['risk']}")
                            st.caption(f"**饮食**：{info['diet']}")
                        with d2:
                            st.caption(f"**运动**：{info['exercise']}")
                            st.caption(f"**茶饮**：{info['tea']}")
                    st.session_state.constitution_type = ct_name

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