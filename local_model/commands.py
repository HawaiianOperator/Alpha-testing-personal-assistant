"""Explicit, fast keyword commands; no model download, task history, or shell eval."""
from __future__ import annotations
import argparse
import ast
from decimal import Decimal, DecimalException, localcontext
import json
import re
from urllib.parse import urlencode

KEYWORDS = ("google", "search", "compute")
REJECTION_MESSAGE = "I am sorry the task is too complicated, try an less complicated task"
MAX_REQUEST_BYTES = 256
MAX_EXPRESSION_CHARS = 128


def suggestions(prefix):
    """Complete command keywords from the current prefix only."""
    text = prefix.lstrip().lower()
    if not text:
        return [word + " " for word in KEYWORDS]
    if any(char.isspace() for char in text):
        return []
    return [word + " " for word in KEYWORDS if word.startswith(text)]


def compute(expression):
    """Evaluate bounded decimal arithmetic without eval, exec, names, or calls."""
    expression = expression.strip()
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("Provide an arithmetic expression of 1-128 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("Use numbers, parentheses, and +, -, *, or /.") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > 64:
        raise ValueError("The expression is too complex.")
    pending = [(tree.body, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 16:
            raise ValueError("The expression is nested too deeply.")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    with localcontext() as context:
        context.prec = 28
        context.Emax, context.Emin = 100, -100

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                literal = ast.get_source_segment(expression, node)
                if literal is None or len(literal) > 32:
                    raise ValueError("Numeric literals must be at most 32 characters.")
                try:
                    value = Decimal(literal.replace("_", ""))
                except DecimalException:
                    raise ValueError("Use decimal numbers.") from None
                if not value.is_finite() or not -100 <= value.as_tuple().exponent <= 100 or abs(value) > Decimal("1e50"):
                    raise ValueError("Number exceeds the supported range.")
                return value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ValueError("Division by zero is not supported.")
                    return left / right
            raise ValueError("Only numbers, parentheses, +, -, *, and / are supported.")
        try:
            value = visit(tree.body)
            if not value.is_finite() or abs(value) > Decimal("1e100"):
                raise ValueError("Result exceeds the supported range.")
        except DecimalException:
            raise ValueError("The arithmetic exceeds the supported range.") from None
    if value == 0:
        return "0"
    output = format(value, "f")
    return output.rstrip("0").rstrip(".") if "." in output else output


def reject(command=None):
    """Refuse requests beyond implemented capabilities; never execute a partial plan."""
    return {"status": "rejected", "command": command, "message": REJECTION_MESSAGE}


def route(request):
    if not isinstance(request, str):
        raise TypeError("Request must be text.")
    if len(request) > MAX_REQUEST_BYTES or len(request.encode("utf-8")) > MAX_REQUEST_BYTES:
        return reject()
    text = request.strip()
    if not text:
        return {"status": "needs_clarification", "command": None,
                "question": "What would you like me to do?"}
    match = re.match(r"^([a-zA-Z]+)(?:\s*:\s*|\s+|$)(.*)$", text, re.DOTALL)
    if not match or match[1].lower() not in KEYWORDS:
        return reject()
    command = match[1].lower()
    argument = match[2].strip()
    if command in ("google", "search"):
        if not argument:
            return {"status": "needs_clarification", "command": "search",
                    "question": "What would you like to search for?"}
        return {"status": "ready", "command": "search", "query": argument,
                "url": "https://www.google.com/search?" + urlencode({"q": argument}),
                "privacy": "Opening this search sends the query to Google."}
    if not argument:
        return {"status": "needs_clarification", "command": "compute",
                "question": "What would you like me to calculate?"}
    try:
        value = compute(argument)
    except ValueError:
        return reject("compute")
    return {"status": "completed", "command": "compute", "value": value}


class KeywordSession:
    """Current-task clarification state; stores only a known command label."""
    def __init__(self):
        self.pending_command = None

    def accept(self, text):
        request = ((self.pending_command + " " + text)
                   if self.pending_command else text)
        result = route(request)
        self.pending_command = (result["command"] if result["status"] == "needs_clarification" else None)
        return result

    def cancel(self):
        self.pending_command = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?")
    parser.add_argument("--suggest")
    parser.add_argument("--open", action="store_true",
                        help="Explicitly open a recognized Google search in the local browser.")
    args = parser.parse_args()
    if args.suggest is not None:
        print(json.dumps({"suggestions": suggestions(args.suggest)}))
        return
    if args.request is None:
        parser.error("Provide a quoted request or --suggest.")
    result = route(args.request)
    if args.open and result.get("command") == "search" and result["status"] == "ready":
        import webbrowser
        result["browser_open_requested"] = bool(webbrowser.open(result["url"]))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
