
import ast
import operator
from datetime import datetime
from langchain_core.tools import tool


# Calculator Tool
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):

    def evaluate(node):

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp):
            op = _ALLOWED_OPERATORS.get(type(node.op))

            if op is None:
                raise ValueError("Unsupported operator")

            return op(evaluate(node.left), evaluate(node.right))

        if isinstance(node, ast.UnaryOp):
            op = _ALLOWED_OPERATORS.get(type(node.op))

            if op is None:
                raise ValueError("Unsupported operator")

            return op(evaluate(node.operand))

        raise ValueError("Invalid mathematical expression")

    tree = ast.parse(expression, mode="eval")
    result = evaluate(tree.body)

    if isinstance(result, complex) or abs(result) > 1e100:
        raise ValueError("Result is too large")

    return result


@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        return str(safe_calculate(expression))
    except Exception as e:
        return f"Calculation error: {e}"


# Current Time Tool
@tool
def get_current_time() -> str:
    """Get the current date and time."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
