"""Unit tests for the standalone calculator tool."""

from __future__ import annotations

from examples.calculator_tool import (
    ANTHROPIC_TOOL_SCHEMA,
    OPENAI_TOOL_SCHEMA,
    execute_calculator,
)


class TestCalculatorTool:
    def test_addition(self):
        res = execute_calculator("add", 12, 8)
        assert res["result"] == 20

    def test_multiplication(self):
        res = execute_calculator("multiply", 15, 4)
        assert res["result"] == 60

    def test_division(self):
        res = execute_calculator("divide", 100, 4)
        assert res["result"] == 25

    def test_division_by_zero(self):
        res = execute_calculator("divide", 10, 0)
        assert "error" in res

    def test_power(self):
        res = execute_calculator("power", 2, 8)
        assert res["result"] == 256

    def test_unknown_operation(self):
        res = execute_calculator("modulo", 10, 3)
        assert "error" in res

    def test_schema_validity(self):
        assert OPENAI_TOOL_SCHEMA["function"]["name"] == "calculate"
        assert ANTHROPIC_TOOL_SCHEMA["name"] == "calculate"
        assert "operation" in OPENAI_TOOL_SCHEMA["function"]["parameters"]["properties"]
        assert "operation" in ANTHROPIC_TOOL_SCHEMA["input_schema"]["properties"]
