import pytest

from rag_agent.tools.calculator import CalculatorError, CalculatorTool, safe_eval
from tests.conftest import FakeLLM


def test_basic_math():
    assert safe_eval("15/100*2480") == pytest.approx(372.0)
    assert safe_eval("sqrt(16) + 2**3") == 12
    assert safe_eval("-(3+4)*2") == -14


@pytest.mark.parametrize(
    "bad",
    ["__import__('os').system('ls')", "open('x')", "().__class__", "2**100000", "1/0", "a+1", "lambda: 1", "[1,2]"],
)
def test_rejects_unsafe_or_invalid(bad):
    with pytest.raises(CalculatorError):
        safe_eval(bad)


def test_tool_translates_then_evaluates():
    res = CalculatorTool(FakeLLM()).run("What is 15% of 2480?")
    assert res.ok and res.text.endswith("= 372.0")


def test_tool_reports_errors_instead_of_raising():
    class BadLLM(FakeLLM):
        def invoke(self, messages):
            return "__import__('os')"

    res = CalculatorTool(BadLLM()).run("hack")
    assert not res.ok and "Calculator error" in res.text
