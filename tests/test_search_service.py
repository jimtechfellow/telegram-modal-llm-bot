"""Search orchestration is independent of Telegram, Modal, and real credentials."""

import json

import pytest

from search_bot.service import ResearchError, SearchBot, SearchEvidence, Source, research_topic


class SearchDouble:
    def __init__(self):
        self.queries = []

    async def search(self, topic):
        self.queries.append(topic)
        return SearchEvidence(
            (Source("官方文档", "https://docs.example.org/reference"),),
            "官方资料说明容器可复用。" * 300,
        )


class ModelDouble:
    def __init__(self, text=None):
        self.messages = []
        self.text = text or json.dumps(
            {"findings": [{"text": "资料显示容器可复用。", "sources": [1]}], "limitations": "仅据检索摘要。"}
        )

    async def generate(self, messages):
        self.messages = messages
        return self.text


@pytest.mark.parametrize("text,topic", [("调研 容器", "容器"), (" 调研\nQwen ", "Qwen"), ("调研容器", None),
                                        ("普通聊天", None)])
def test_explicit_research_route(text, topic):
    assert research_topic(text) == topic


async def test_standalone_report_uses_search_urls_and_bounded_context():
    search, model = SearchDouble(), ModelDouble()
    report = await SearchBot(search).research("GPU 复用", model)
    assert search.queries == ["GPU 复用"]
    assert "研究报告" in report and "资料显示容器可复用。[1]" in report
    assert "https://docs.example.org/reference" in report
    assert sum(len(m["content"]) for m in model.messages) <= 3300
    assert "不可信" in model.messages[0]["content"]


async def test_swapping_model_does_not_change_search_logic():
    search = SearchDouble()
    bot = SearchBot(search)

    class OtherModel:
        async def generate(self, messages):
            return json.dumps({"findings": [{"text": "另一个模型的中文结论。", "sources": [1]}],
                               "limitations": "证据有限。"})

    for model in (ModelDouble(), OtherModel()):
        assert "https://docs.example.org/reference" in await bot.research("容器", model)
    assert search.queries == ["容器", "容器"]


async def test_context_budget_counts_json_escaping():
    class EscapedSearch:
        async def search(self, topic):
            return SearchEvidence((Source("官方资料", "https://docs.example.org/reference"),),
                                  "证据\x00\n\t\"" * 10000)

    model = ModelDouble()
    await SearchBot(EscapedSearch()).research("容器", model)
    assert len(model.messages[1]["content"]) <= 2400


@pytest.mark.parametrize("answer", [
    "无法解析", '{"findings":[],"limitations":"无"}',
    '{"findings":[{"text":"结论","sources":[99]}],"limitations":"无"}',
    '{"findings":[{"text":"结论","sources":[true]}],"limitations":"无"}',
    '{"findings":[{"text":"https://fabricated.org","sources":[1]}],"limitations":"无"}',
    '{"findings":[{"text":"结论 [99]","sources":[1]}],"limitations":"无"}',
    '{"findings":[{"text":"结论","sources":[1]}],"limitations":"见 invented.com"}',
    '{"findings":[{"text":"English only","sources":[1]}],"limitations":"Limited evidence"}',
])
async def test_invalid_report_or_fabricated_citation_is_rejected(answer):
    with pytest.raises(ResearchError):
        await SearchBot(SearchDouble()).research("容器", ModelDouble(answer))


@pytest.mark.parametrize("field", ["text", "limitations"])
@pytest.mark.parametrize("citation", ["[1]", "[99]", "[99,100]", "[99, 100]", "[99–100]",
                                      "[99-100]", "[99—100]", "[1,99–100]"])
async def test_model_authored_numeric_citations_are_rejected_in_both_fields(field, citation):
    answer = {"findings": [{"text": "中文结论。", "sources": [1]}], "limitations": "证据有限。"}
    target = answer["findings"][0] if field == "text" else answer
    target[field] += citation
    with pytest.raises(ResearchError, match="引用校验失败"):
        await SearchBot(SearchDouble()).research("容器", ModelDouble(json.dumps(answer)))


async def test_failed_or_empty_search_never_invokes_model():
    class EmptySearch:
        async def search(self, topic):
            return SearchEvidence((), "")

    model = ModelDouble()
    with pytest.raises(ResearchError, match="搜索"):
        await SearchBot(EmptySearch()).research("容器", model)
    assert not model.messages


async def test_unexpected_provider_failure_is_safe_and_never_invokes_model():
    class BrokenSearch:
        async def search(self, topic):
            raise RuntimeError("credential-bearing-provider-error")

    model = ModelDouble()
    with pytest.raises(ResearchError) as caught:
        await SearchBot(BrokenSearch()).research("容器", model)
    assert "credential-bearing" not in str(caught.value) and not model.messages


@pytest.mark.parametrize("topic", ["", "x" * 301])
async def test_invalid_topic_has_no_external_effect(topic):
    search = SearchDouble()
    with pytest.raises(ResearchError):
        await SearchBot(search).research(topic, ModelDouble())
    assert not search.queries
