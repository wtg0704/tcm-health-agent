"""Stage A 测试：医疗风控边界 + 工具构建 + sources 提取（纯逻辑，不依赖 LLM/向量库）"""
import pytest

from backend.safety import is_medical_query, medical_reject_response, wrap_disclaimer
from backend.tools import build_tools
from backend.agent_router import build_messages, extract_sources, _with_system
from langchain_core.messages import ToolMessage, SystemMessage


class TestMedicalGuard:
    """医疗意图拦截边界用例"""

    @pytest.mark.parametrize("text", [
        "我头疼吃什么药",
        "高血压怎么治疗",
        "能开个方子吗",
        "这个病能治好吗",
        "降压药怎么吃",
        "感冒了要打针",
    ])
    def test_medical_positive(self, text):
        """诊疗/用药类请求必须被拦截"""
        assert is_medical_query(text) is True

    @pytest.mark.parametrize("text", [
        "黄芪有什么功效",
        "失眠吃什么好",
        "什么是痰湿质",
        "四君子汤由哪几味药组成",
        "最近容易累怎么调理",
    ])
    def test_medical_negative(self, text):
        """养生/知识类请求不应被误伤"""
        assert is_medical_query(text) is False

    def test_reject_response_guides(self):
        """拒绝回复应含合规引导（转化到养生话题）"""
        resp = medical_reject_response("我头疼吃什么药")
        assert "无法提供医疗建议" in resp
        assert "头痛" in resp


class TestTools:
    """工具构建（Function Calling 的 schema）"""

    def test_build_tools_count(self):
        """4 个检索工具 + 1 个用户画像工具"""
        tools = build_tools("test-user")
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {"search_herbs", "search_formulas", "search_diet",
                         "search_constitution", "get_user_profile"}

    def test_profile_tool_hides_user_id(self):
        """get_user_profile 通过闭包绑定 user_id，schema 不应暴露该参数"""
        tools = build_tools("test-user")
        profile = [t for t in tools if t.name == "get_user_profile"][0]
        assert "user_id" not in profile.args


class TestSources:
    """引用来源提取"""

    def test_extract_sources(self):
        msg = ToolMessage(
            content="[来源1] herbs/黄芪（相关度: 0.85）\n黄芪性温补气。\n\n"
                    "[来源2] diet/红枣茶（相关度: 0.7）\n红枣补中益气。",
            tool_call_id="1", name="search_herbs",
        )
        sources = extract_sources([msg])
        assert len(sources) == 2
        assert sources[0]["doc"] == "herbs/黄芪"
        assert sources[1]["doc"] == "diet/红枣茶"

    def test_wrap_disclaimer(self):
        ans = wrap_disclaimer("黄芪补气")
        assert "不构成医疗诊断" in ans


class TestMessages:
    """消息构建（checkpoint 模式下 system prompt 动态注入）"""

    def test_build_messages_without_system(self):
        msgs = build_messages("黄芪有什么功效")
        # 不含 system prompt（由 call_model 动态注入）
        assert not any(isinstance(m, SystemMessage) for m in msgs)

    def test_with_system_prepends(self):
        msgs = _with_system(build_messages("黄芪有什么功效"))
        assert isinstance(msgs[0], SystemMessage)
