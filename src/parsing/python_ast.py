"""Static Python analysis using ``ast``. Source is parsed, never executed."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

_HTTP = {"get", "post", "put", "delete", "patch", "route", "head", "options"}


@dataclass
class PyFunction:
    name: str
    line: int
    is_async: bool = False
    doc: str = ""


@dataclass
class PyRoute:
    method: str
    path: str
    handler: str
    line: int


@dataclass
class PyModuleInfo:
    functions: list[PyFunction] = field(default_factory=list)
    classes: list[tuple[str, int]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    routes: list[PyRoute] = field(default_factory=list)
    docstring: str = ""
    syntax_error: str | None = None


def _route_from_decorator(dec: ast.expr, handler: str, line: int) -> PyRoute | None:
    """Detect Flask/FastAPI-style ``@app.get('/x')`` / ``@app.route('/x', methods=[...])``."""
    if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
        return None
    attr = dec.func.attr.lower()
    if attr not in _HTTP or not dec.args:
        return None
    first = dec.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    method = attr.upper()
    if attr == "route":
        method = "GET"
        for kw in dec.keywords:
            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                ms = [e.value for e in kw.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                method = "/".join(m.upper() for m in ms) or "GET"
    return PyRoute(method, first.value, handler, line)


def parse_python(source: str) -> PyModuleInfo:
    """Parse Python source into a :class:`PyModuleInfo`. Syntax errors are captured, not raised."""
    info = PyModuleInfo()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        info.syntax_error = f"line {exc.lineno}: {exc.msg}"
        return info
    except (ValueError, RecursionError, MemoryError) as exc:
        info.syntax_error = f"unparseable: {type(exc).__name__}"
        return info
    info.docstring = (ast.get_docstring(tree) or "").strip().split("\n")[0][:200]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(PyFunction(
                node.name, node.lineno, isinstance(node, ast.AsyncFunctionDef),
                (ast.get_docstring(node) or "").strip().split("\n")[0][:160],
            ))
            for dec in node.decorator_list:
                r = _route_from_decorator(dec, node.name, node.lineno)
                if r:
                    info.routes.append(r)
        elif isinstance(node, ast.ClassDef):
            info.classes.append((node.name, node.lineno))
        elif isinstance(node, ast.Import):
            info.imports.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            info.imports.append(node.module.split(".")[0])
    info.imports = sorted(set(info.imports))
    return info
