"""AST Impact Selector: coverage-map driven test impact analysis.

Maps (built offline from the pinned sqlglot commit, see tools/):
  cov  : sqlglot file -> line -> tests that executed it at test time
  imp  : sqlglot file -> lines executed at import time
  fix  : tests/... file -> tests that opened it
  attr : "module:Class.attr" -> tests that read that class attribute
The pinned sources live in /app/src so the patch can be applied and the
changed statements located with the ast module.
"""
from __future__ import annotations

import ast
import gzip
import json
import os
import re
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/data"))
APP = Path(__file__).resolve().parent
SRC = Path(os.environ.get("SRC_DIR", APP / "src"))
MAP = Path(os.environ.get("MAP_FILE", APP / "map.json.gz"))
DEBUG = os.environ.get("BR_DEBUG") == "1"

ALL = "ALL"  # sentinel: run everything
# Semantic-equivalence normalisation (see _Equiv). Set BR_EQUIV=0 to disable.
EQUIV = os.environ.get("BR_EQUIV", "1") != "0"


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr)


# --------------------------------------------------------------------------- patch


def parse_patch(text):
    """-> list of dict(old, new, hunks=[(old_start, old_len, new_start, new_len, lines)])"""
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("diff --git "):
            cur = {"old": None, "new": None, "hunks": [], "git": ln, "binary": False}
            files.append(cur)
        elif ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur["hunks"]:
                cur = {"old": None, "new": None, "hunks": [], "git": "", "binary": False}
                files.append(cur)
            o = ln[4:].split("\t")[0].strip()
            n = lines[i + 1][4:].split("\t")[0].strip()
            cur["old"] = None if o == "/dev/null" else re.sub(r"^a/", "", o)
            cur["new"] = None if n == "/dev/null" else re.sub(r"^b/", "", n)
            i += 2
            continue
        elif ln.startswith("Binary files") or ln.startswith("GIT binary patch"):
            if cur is not None:
                cur["binary"] = True
        elif ln.startswith("rename from ") and cur is not None:
            cur["old"] = ln[len("rename from "):].strip()
        elif ln.startswith("rename to ") and cur is not None:
            cur["new"] = ln[len("rename to "):].strip()
        elif ln.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            if not m:
                raise ValueError("bad hunk")
            os_, ol = int(m[1]), int(m[2]) if m[2] is not None else 1
            ns, nl = int(m[3]), int(m[4]) if m[4] is not None else 1
            body = []
            i += 1
            co, cn = 0, 0
            while i < len(lines) and (co < ol or cn < nl):
                b = lines[i]
                if b.startswith("\\"):
                    i += 1
                    continue
                tag = b[:1] if b else " "
                if tag == " ":
                    co += 1
                    cn += 1
                elif tag == "-":
                    co += 1
                elif tag == "+":
                    cn += 1
                else:
                    raise ValueError("bad hunk line")
                body.append((tag, b[1:]))
                i += 1
            while i < len(lines) and lines[i].startswith("\\"):
                i += 1
            cur["hunks"].append((os_, ol, ns, nl, body))
            continue
        i += 1
    for f in files:
        if f["old"] is None and f["new"] is None and f["git"]:
            m = re.match(r"diff --git a/(\S+) b/(\S+)", f["git"])
            if m:
                f["old"], f["new"] = m[1], m[2]
    return files


def apply_hunks(old_lines, hunks):
    """Apply hunks to a list of lines (no newline). Returns (new_lines, changed_old,
    insert_points). changed_old: old line numbers removed/replaced (1-based);
    insert_points: old line numbers *before which* lines were inserted."""
    new = []
    pos = 0  # index into old_lines
    changed, inserts = set(), {}
    for os_, ol, ns, nl, body in hunks:
        start = os_ - 1 if ol > 0 else os_  # for ol == 0 the hunk inserts after line os_
        if start < pos:
            raise ValueError("overlapping hunks")
        new.extend(old_lines[pos:start])
        pos = start
        prev = " "
        for tag, txt in body:
            if tag == " ":
                if pos >= len(old_lines) or old_lines[pos] != txt:
                    raise ValueError("context mismatch")
                new.append(txt)
                pos += 1
            elif tag == "-":
                if pos >= len(old_lines) or old_lines[pos] != txt:
                    raise ValueError("delete mismatch")
                changed.add(pos + 1)
                pos += 1
            else:
                new.append(txt)
                if prev == " " or prev == "+":
                    # pure insertion (not the replacement half of a -/+ run)
                    inserts.setdefault(pos + 1, []).append((len(new), txt))
                else:
                    tag = prev  # a '+' run directly after '-' lines replaces them
            prev = tag
    new.extend(old_lines[pos:])
    return new, changed, inserts


# --------------------------------------------------------------------------- AST helpers

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


class Index:
    """Per old-file index: for each statement node, its parent chain."""

    def __init__(self, tree):
        self.tree = tree
        self.stmts = []  # (start, end, node, chain)
        self._walk(tree, [])

    def _walk(self, node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                start = child.lineno
                decos = getattr(child, "decorator_list", None)
                if decos:
                    start = min([start] + [d.lineno for d in decos])
                self.stmts.append((start, child.end_lineno, child, chain))
                self._walk(child, chain + [child])
            elif isinstance(child, (ast.excepthandler, ast.match_case)) or isinstance(
                child, ast.AST
            ):
                self._walk(child, chain)

    def innermost(self, line):
        best = None
        for s, e, node, chain in self.stmts:
            if s <= line <= e and (best is None or len(chain) >= len(best[3])):
                if best is None or (e - s) <= (best[1] - best[0]):
                    best = (s, e, node, chain)
        return best


def _norm(node):
    return ast.dump(node, include_attributes=False)


# --------------------------------------------------------------------------- selector


class Selector:
    def __init__(self, tests):
        self.tests = tests
        self.all = [t["number"] for t in tests]
        self.by_id = {t["id"]: t["number"] for t in tests}
        self.by_file = {}
        for t in tests:
            self.by_file.setdefault(t["file"], []).append(t["number"])
        with gzip.open(MAP, "rt") as f:
            m = json.load(f)
        self.sets = m["sets"]
        ids = m.get("ids")
        if ids is not None:
            # the map stores the build-time test numbering; translate by test id
            if [self.by_id.get(i) for i in ids] != list(range(len(ids))):
                tr = [self.by_id.get(i, -2) for i in ids]
                self.sets = [[-1 if x == -1 else tr[x] for x in st] for st in self.sets]
                self.sets = [[x for x in st if x != -2] for st in self.sets]
                log("test numbering differs from map: translated")
            unknown = set(self.by_id) - set(ids)
            self.unmapped = {self.by_id[i] for i in unknown}
        else:
            self.unmapped = set()
        self.cov = m["cov"]
        self.imp = m["imp"]
        self.fix = m["fix"]
        self.attr = m["attr"]
        self.attr_by_name = {}
        for k, v in self.attr.items():
            self.attr_by_name.setdefault(k.rsplit(".", 1)[1], []).append(v)
        self._src = {}
        self._imp_sets = {}
        self.reset()
        self._idx = {}
        self._names = None

    # ---- sources
    def source(self, rel):
        if rel not in self._src:
            p = SRC / rel
            try:
                self._src[rel] = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                self._src[rel] = None
        return self._src[rel]

    def index(self, rel):
        if rel not in self._idx:
            src = self.source(rel)
            try:
                self._idx[rel] = Index(ast.parse(src)) if src is not None else None
            except SyntaxError:
                self._idx[rel] = None
        return self._idx[rel]

    def reset(self):
        """Per-commit state used by the differential replay."""
        self.spans = []  # (file, start, end) whose statement-coverage selected tests
        self.span_tests = set()
        self.other_tests = set()  # tests selected by any other rule
        self.patched = {}  # rel -> patched source
        self.need_imp = False

    # ---- coverage lookups
    def cov_span(self, rel, s, e, replayable=True):
        fc = self.cov.get(rel, {})
        out = set()
        for ln in range(s, e + 1):
            i = fc.get(str(ln))
            if i is not None:
                out.update(self.sets[i])
        if replayable:
            self.spans.append((rel, s, e))
            self.span_tests |= out
        return out

    def attr_tests(self, module, qual, name):
        i = self.attr.get(f"{module}:{qual}.{name}")
        if i is None:
            return None
        s = self.sets[i]
        if -1 in s:
            return None  # read at import time (metaclass etc): effect not attributable
        self.other_tests.update(s)
        return set(s)

    # ---- per file
    def select_file(self, f):
        old, new = f["old"], f["new"]
        path = new or old
        if f["binary"]:
            return ALL
        if old != new:
            # added, deleted or renamed file
            if old is None and new is not None and not new.startswith("sqlglot/"):
                if new.startswith("tests/") and new.endswith(".py"):
                    return ALL  # a new test module we have no numbers for: be safe
                return set()  # e.g. new doc file
            return ALL
        if path.startswith("sqlglot/") and path.endswith(".py"):
            return self.select_sqlglot_py(path, f["hunks"])
        if path.startswith("tests/") and path.endswith(".py"):
            r = self.select_test_py(path, f["hunks"])
            if r is not ALL:
                self.other_tests |= r
            return r
        if path.startswith("tests/"):
            r = self.select_fixture(path, f["hunks"])
            if r is not ALL:
                self.other_tests |= r
            return r
        if path.startswith("sqlglot/"):
            return ALL  # non-python package data
        return set()  # docs, README, CHANGELOG, posts, benchmarks ...

    def _apply(self, rel, hunks):
        src = self.source(rel)
        if src is None:
            raise ValueError("no source")
        old_lines = src.split("\n")
        new_lines, changed, inserts = apply_hunks(old_lines, hunks)
        return src, "\n".join(new_lines), changed, inserts

    def select_fixture(self, rel, hunks):
        try:
            src, new, changed, inserts = self._apply(rel, hunks)
        except ValueError:
            return ALL
        if src == new:
            return set()
        i = self.fix.get(rel)
        if i is None:
            log("  unknown fixture", rel)
            return ALL
        s = self.sets[i]
        if -1 in s:
            log("  fixture opened at import", rel)
            return ALL
        out = set(s)
        spec = TPC_FIXTURES.get(rel)
        if spec is not None:
            out = self.tpc_fixture(rel, src, new, out, spec)
        return out

    def tpc_fixture(self, rel, src, new, out, spec):
        """TestExecutor.setUpClass loads the TPC fixtures for the whole class, so the
        fixture map charges every executor test; only the tests listed in TPC_FIXTURES
        read them, and only the listed fields of each (meta, sql, expected) pair."""
        try:
            old_pairs, new_pairs = _fixture_pairs(src), _fixture_pairs(new)
        except Exception:
            return out
        prefix = spec["class"] + "::"
        keep = {n for n in out if not self.ids_of(n).startswith(prefix)}
        for test_id, fields in spec["consumers"].items():
            proj = lambda pairs: [tuple(p[f] for f in fields) for p in pairs]  # noqa: E731
            if len(old_pairs) != len(new_pairs) or proj(old_pairs) != proj(new_pairs):
                n = self.by_id.get(test_id)
                if n is not None:
                    keep.add(n)
        log(f"  tpc fixture {rel}: {len(out)} -> {len(keep)} tests")
        return keep

    def ids_of(self, n):
        if not hasattr(self, "_id_by_num"):
            self._id_by_num = {v: k for k, v in self.by_id.items()}
        return self._id_by_num.get(n, "")

    def select_test_py(self, rel, hunks):
        try:
            src, new, changed, inserts = self._apply(rel, hunks)
            new_tree = ast.parse(new)
            old_tree = ast.parse(src)
        except (ValueError, SyntaxError):
            return set(self.by_file.get(rel, [])) or ALL
        if _norm(old_tree) == _norm(new_tree):
            return set()
        if rel not in self.by_file:
            return ALL  # helpers / conftest: shared by many modules
        # compare per test function by qualified name
        def funcs(tree):
            out, rest = {}, []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    crest = []
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                            out[f"{node.name}::{sub.name}"] = _norm(sub)
                        else:
                            crest.append(_norm(sub))
                    rest.append(("class", node.name, _norm_list(node.bases), tuple(crest)))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                    out[node.name] = _norm(node)
                else:
                    rest.append(_norm(node))
            return out, rest

        of, orest = funcs(old_tree)
        nf, nrest = funcs(new_tree)
        if orest != nrest:
            return set(self.by_file[rel])
        sel = set()
        for k in set(of) | set(nf):
            if of.get(k) != nf.get(k):
                n = self.by_id.get(f"{rel}::{k}")
                if n is None:
                    if k not in of:
                        continue  # brand new test: not in tests.json, cannot be selected
                    return set(self.by_file[rel])
                sel.add(n)
        return sel

    def select_sqlglot_py(self, rel, hunks):
        try:
            src, new, changed, inserts = self._apply(rel, hunks)
        except ValueError:
            log("  apply failed", rel)
            return ALL
        self.patched[rel] = new
        if re.search(r"^[ \t\f]*#.*?coding[:=]", "\n".join(new.split("\n")[:2]), re.M):
            return ALL
        try:
            new_tree = ast.parse(new)
        except SyntaxError:
            log("  syntax error after patch", rel)
            return ALL
        idx = self.index(rel)
        if idx is None:
            return ALL
        if _norm(idx.tree) == _norm(new_tree):
            log("  AST unchanged (comments/formatting)")
            return set()
        if EQUIV and _norm(_Equiv().visit(ast.parse(self.source(rel)))) == _norm(
            _Equiv().visit(ast.parse(new))
        ):
            log("  semantically equivalent (identity vs equality against None/enum member)")
            return set()
        module = rel[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        new_idx = Index(new_tree)

        sel = set()
        points = [(ln, "chg") for ln in sorted(changed)] + [
            (ln, "ins")
            for ln, txts in sorted(inserts.items())
            if not all(not t.strip() or t.strip().startswith("#") for _, t in txts)
        ]
        self._inserts = inserts
        for ln, kind in points:
            r = self.unit_tests(rel, module, idx, new_idx, ln, kind, changed, inserts)
            if r is ALL:
                return ALL
            sel |= r
        return sel

    def unit_tests(self, rel, module, idx, new_idx, ln, kind, changed, inserts):
        """Tests affected by a change at old line ln."""
        if kind == "ins":
            return self.insert_tests(rel, module, idx, new_idx, ln)
        cands = [ln]
        out = set()
        for c in cands:
            hit = idx.innermost(c)
            if hit is None:
                if kind == "ins":
                    # blank/comment area at module level: inspect what was inserted
                    continue
                # a changed line that is not inside any statement (blank/comment)
                continue
            s, e, node, chain = hit
            scope = None
            for anc in reversed(chain + [node]):
                if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope = anc
                    break
            if scope is not None and scope is not node:
                # inside a function body
                if self.runs_at_import(rel, s, e):
                    log(f"  {rel}:{s}-{e} runs at import time -> import diff")
                    self.defer_import()
                if kind == "ins":
                    # inserted statement: anything that ran the enclosing function
                    fs = scope.lineno
                    out |= self.cov_span(rel, fs, scope.end_lineno)
                else:
                    out |= self.cov_span(rel, s, e)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # def header / decorator / defaults (or insertion right at def line)
                if kind == "ins" and c == ln - 1 and c == node.end_lineno:
                    # inserting after a function: new module/class level statement
                    r = self.level_insert(rel, module, idx, new_idx, ln, chain)
                    if r is ALL:
                        return ALL
                    out |= r
                    continue
                if self.runs_at_import(rel, node.lineno + 1, node.end_lineno):
                    self.defer_import()
                out |= self.cov_span(rel, node.lineno, node.end_lineno)
                if node.decorator_list and any(s <= x < node.lineno for x in [c]):
                    return ALL if self._decorator_is_import_time(node) else out
                continue
            # module or class level statement (executed at import time)
            if kind == "ins":
                r = self.level_insert(rel, module, idx, new_idx, ln, chain)
            else:
                r = self.level_change(rel, module, node, chain)
            if r is ALL:
                return ALL
            out |= r
        return out

    def insert_tests(self, rel, module, idx, new_idx, ln):
        """Pure insertion before old line ln: locate the inserted statements in the
        patched AST and charge the scope they were inserted into."""
        out = set()
        for new_ln, txt in self._inserts[ln]:
            t = txt.strip()
            if not t or t.startswith("#"):
                continue
            hit = new_idx.innermost(new_ln + 1)
            if hit is None:
                return ALL
            s, e, node, chain = hit
            path = chain + [node]
            funcs = [a for a in path if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if funcs and not (funcs[0] is node):
                # inserted inside an existing function: find that function in the old file
                qn = _qualname(path, funcs[0])
                of = _find_def(idx, qn)
                if of is None:
                    return ALL
                if self.runs_at_import(rel, of.lineno + 1, of.end_lineno):
                    self.defer_import()
                out |= self.cov_span(rel, of.lineno, of.end_lineno)
                continue
            classes = [a for a in chain if isinstance(a, ast.ClassDef)]
            if classes and not funcs or (funcs and funcs[0] is node and classes):
                r = self.level_insert(rel, module, idx, new_idx, ln, chain)
                if r is ALL:
                    return ALL
                out |= r
                continue
            return ALL
        return out

    def _decorator_is_import_time(self, node):
        return True

    def class_qual(self, chain):
        parts = [a.name for a in chain if isinstance(a, ast.ClassDef)]
        if any(isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) for a in chain):
            return None
        return ".".join(parts) if parts else ""

    def level_change(self, rel, module, node, chain):
        qual = self.class_qual(chain)
        if qual is None:
            return ALL
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            return set()  # docstring
        if qual and isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names = _target_names(node)
            out = set()
            for n in names:
                t = self.attr_tests(module, qual, n)
                if t is not None:
                    out |= t
            # Always confirm through the import diff: the attribute may be read while
            # sqlglot itself is imported (before the read tracer was installed), which
            # can change derived state or even make the import fail.
            log(f"  class attr {qual}.{names} -> {len(out)} tests + import diff")
            self.defer_import()
            return out
        if not qual and isinstance(node, (ast.Assign, ast.AnnAssign)):
            r = self.module_global(rel, module, _target_names(node))
            self.defer_import()
            return set() if r is ALL else r
        # class header, decorators, other module/class level statements
        return self.defer_import()

    def runs_at_import(self, rel, s, e):
        lines = self._imp_sets.get(rel)
        if lines is None:
            lines = self._imp_sets[rel] = set(self.imp.get(rel, ()))
        return any(ln in lines for ln in range(s, e + 1))

    def defer_import(self):
        """The change only acts through import-time state: resolve it after all files
        are seen, by diffing the patched import against the pinned one."""
        self.need_imp = True
        return set()

    def level_insert(self, rel, module, idx, new_idx, ln, chain):
        """A statement inserted directly in a class body (new attribute or method)
        changes behaviour only where that class is used: select every test that read
        any of the class's attributes or ran any of its code."""
        qual = self.class_qual(chain)
        if qual is None:
            return ALL
        if not qual:
            return self.defer_import()  # new module-level statement
        cls = None
        for a in reversed(chain):
            if isinstance(a, ast.ClassDef):
                cls = a
                break
        ocls = None
        for s, e, node, ch in idx.stmts:
            if isinstance(node, ast.ClassDef) and _qualname(ch + [node], node) == qual:
                ocls = node
                break
        if ocls is None:
            return ALL
        out = self.cov_span(rel, ocls.lineno, ocls.end_lineno, replayable=False)
        prefix = f"{module}:{qual}."
        for k, i in self.attr.items():
            if k.startswith(prefix):
                s = self.sets[i]
                out.update(x for x in s if x >= 0)
        if not out:
            return ALL
        log(f"  class-level insert in {qual} -> {len(out)} tests")
        self.other_tests |= out
        self.defer_import()
        return out

    def global_refs(self, name):
        """Tests that executed, at test time, a line mentioning `name` (anywhere)."""
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        out = set()
        for frel in self.cov.keys():
            src = self.source(frel)
            if not src or name not in src:
                continue
            fidx = None
            for i, line in enumerate(src.split("\n"), 1):
                if not pat.search(line):
                    continue
                if fidx is None:
                    fidx = self.index(frel)
                    if fidx is None:
                        return ALL
                hit = fidx.innermost(i)
                if hit is None:
                    continue
                s, e, node, ch = hit
                if any(isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) for a in ch) or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out |= self.cov_span(frel, s, e, replayable=False)
        return out

    def module_global(self, rel, module, names):
        """A module-level constant changed. Select tests that executed (at test time)
        any line mentioning the name, anywhere in sqlglot. If the name is used by
        import-time code, give up."""
        if not names:
            return ALL
        out = set()
        for name in names:
            pat = re.compile(r"\b" + re.escape(name) + r"\b")
            for frel in self.cov.keys() | self.imp.keys():
                src = self.source(frel)
                if not src or name not in src:
                    continue
                imp = set(self.imp.get(frel, ()))
                lines = src.split("\n")
                fidx = None
                for i, line in enumerate(lines, 1):
                    if not pat.search(line):
                        continue
                    if frel == rel and _is_def_line(line, name):
                        continue
                    if fidx is None:
                        fidx = self.index(frel)
                        if fidx is None:
                            return ALL
                    hit = fidx.innermost(i)
                    if hit is None:
                        continue
                    s, e, node, ch = hit
                    in_func = any(
                        isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                        for a in ch + [node]
                    ) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if not in_func:
                        if frel == rel and isinstance(node, (ast.Assign, ast.AnnAssign)) and name in _target_names(node):
                            continue
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            continue
                        log(f"  global {name} used at import time {frel}:{i}")
                        return ALL
                    out |= self.cov_span(frel, s, e)
        return out


# Which TestExecutor tests consume the TPC fixtures, and which fields of each
# (meta, sql, expected) pair they read (tests/test_executor.py at the pinned commit:
# setUpClass loads both files; test_execute_* use (meta, sql) via _mp_execute,
# test_optimized_tpch uses (sql, expected)).
TPC_FIXTURES = {
    "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql": {
        "class": "tests/test_executor.py::TestExecutor",
        "consumers": {"tests/test_executor.py::TestExecutor::test_execute_tpcds": (0, 1)},
    },
    "tests/fixtures/optimizer/tpc-h/tpc-h.sql": {
        "class": "tests/test_executor.py::TestExecutor",
        "consumers": {
            "tests/test_executor.py::TestExecutor::test_execute_tpch": (0, 1),
            "tests/test_executor.py::TestExecutor::test_optimized_tpch": (1, 2),
        },
    },
}
_helpers_mod = None


def _fixture_pairs(text):
    """tests.helpers.load_sql_fixture_pairs on a string (pristine helpers)."""
    global _helpers_mod
    if _helpers_mod is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("br_helpers", str(SRC / "tests" / "helpers.py"))
        _helpers_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_helpers_mod)
    h = _helpers_mod
    statements = h._filter_comments(text).split(";")
    out = []
    for i in range(0, len(statements), 2):
        if i + 1 < len(statements):
            sql, meta = h._extract_meta(statements[i].strip())
            out.append((json.dumps(meta, sort_keys=True), sql, statements[i + 1].strip()))
    return out


def _qualname(path, upto):
    parts = []
    for a in path:
        if isinstance(a, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(a.name)
        if a is upto:
            break
    return ".".join(parts)


def _find_def(idx, qn):
    for s, e, node, chain in idx.stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _qualname(chain + [node], node) == qn:
            return node
    return None


_ENUM_MEMBER = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _singleton(node):
    """None, or an enum-member-like constant `Name.MEMBER` / `mod.Name.MEMBER`
    (CapWords class, UPPER_CASE member). For these, `x is C` and `x == C` agree for
    every class in sqlglot: all its __eq__ overrides (Expr, Dialect, _Dialect)
    return False against them unless the operands are the same object."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Attribute) and _ENUM_MEMBER.match(node.attr):
        v = node.value
        name = v.id if isinstance(v, ast.Name) else v.attr if isinstance(v, ast.Attribute) else ""
        return bool(name) and name[0].isupper() and not name.isupper()
    return False


class _Equiv(ast.NodeTransformer):
    """Rewrite `a is S`/`a is not S` to `a == S`/`a != S` when S is a singleton
    as defined above (either side), so identity/equality swaps compare equal."""

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Compare):
            cmp = node.operand
            if len(cmp.ops) == 1 and len(cmp.comparators) == 1:
                op = cmp.ops[0]
                if isinstance(op, ast.Is):
                    return ast.Compare(left=cmp.left, ops=[ast.IsNot()], comparators=cmp.comparators)
                if isinstance(op, ast.IsNot):
                    return ast.Compare(left=cmp.left, ops=[ast.Is()], comparators=cmp.comparators)
                if isinstance(op, ast.In):
                    return ast.Compare(left=cmp.left, ops=[ast.NotIn()], comparators=cmp.comparators)
                if isinstance(op, ast.NotIn):
                    return ast.Compare(left=cmp.left, ops=[ast.In()], comparators=cmp.comparators)
                if isinstance(op, ast.Eq):
                    return ast.Compare(left=cmp.left, ops=[ast.NotEq()], comparators=cmp.comparators)
                if isinstance(op, ast.NotEq):
                    return ast.Compare(left=cmp.left, ops=[ast.Eq()], comparators=cmp.comparators)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        operands = [node.left] + node.comparators
        ops = list(node.ops)
        for i, op in enumerate(ops):
            if isinstance(op, (ast.Is, ast.IsNot)) and (
                _singleton(operands[i]) or _singleton(operands[i + 1])
            ):
                ops[i] = ast.Eq() if isinstance(op, ast.Is) else ast.NotEq()
        node.ops = ops
        return node


def _norm_list(xs):
    return tuple(_norm(x) for x in xs)


def _is_def_line(line, name):
    return re.match(r"\s*" + re.escape(name) + r"\s*(:[^=]*)?=(?!=)", line) is not None


def _target_names(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = []
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                names.append(n.id)
            elif isinstance(n, (ast.Attribute, ast.Subscript)):
                return []
    return names


# --------------------------------------------------------------------------- partition


def partition(nums, ms, max_workers=8):
    if not nums:
        return []
    order = sorted(nums, key=lambda n: (-ms[n], n))
    total = sum(ms[n] for n in order)
    best, best_cost = None, None
    for k in range(1, max_workers + 1):
        if k > len(order):
            break
        loads = [0] * k
        bins = [[] for _ in range(k)]
        for n in order:
            j = min(range(k), key=loads.__getitem__)
            bins[j].append(n)
            loads[j] += ms[n]
        cost = max(loads) + 2000 * k
        if best_cost is None or cost < best_cost:
            best, best_cost = bins, cost
        if max(ms[order[0]], total / (k + 1)) + 2000 * (k + 1) > best_cost:
            break
    return [b for b in best if b]


# --------------------------------------------------------------------------- patched tree


def make_tree(sel, root):
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(SRC / "sqlglot", root / "sqlglot")
    for rel, txt in sel.patched.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(txt, encoding="utf-8")


def _tmp(name):
    return Path("/tmp") / name if Path("/tmp").is_dir() else APP / name


IMP_PINNED = Path(os.environ.get("IMP_PINNED", APP / "imp_pinned.json"))
_imp_state = {"pinned": None}
_t_start = None


def time_left():
    """Seconds of wall time we still allow ourselves for dynamic analysis. The run
    limit is 3 min (4.5 in the final, ~1.5x commits): keep half of it as margin."""
    import time

    n = max(1, len(list((DATA / "commits").glob("*.patch"))))
    # practice: 180 s limit; final (~1.5x commits): 270 s. Keep >= 70 s of margin.
    budget = max(30.0, min(110.0, 3.5 * n) if n <= 36 else min(170.0, 3.5 * n))
    budget = float(os.environ.get("BR_BUDGET", budget))  # local evaluation override
    return budget - (time.time() - _t_start)


def import_diff(sel):
    """Tests affected by import-time state changes: diff structural fingerprints of
    every class attribute / module global of the patched import against the pinned
    commit's, then use the class-attribute read map (test-time readers only: the
    import-time readers' effect shows up in the derived attributes we also diff)."""
    import shutil
    import subprocess

    if time_left() < 5:
        log("  import diff: out of time budget")
        return ALL
    if _imp_state["pinned"] is None:
        _imp_state["pinned"] = json.load(open(IMP_PINNED))
    pinned = _imp_state["pinned"]
    root = _tmp("br_imp")
    out_file = _tmp("br_imp.json")
    try:
        make_tree(sel, root)
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, str(APP / "impdiff.py"), "dump", str(out_file)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if r.returncode != 0:
            log("  import diff: patched import failed")
            return ALL
        new = json.load(open(out_file))
    except Exception as e:
        log(f"  import diff error {e!r}")
        return ALL
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            os.unlink(out_file)
        except OSError:
            pass
    changed = [k for k in set(pinned) | set(new) if pinned.get(k) != new.get(k)]
    out = set()
    for k in changed:
        if k.endswith(".__mro__") or k.endswith(".__members__"):
            log(f"  import diff: {k} changed")
            return ALL
        if "::" in k:
            module, name = k.split("::", 1)
            rel = module.replace(".", "/") + ".py"
            src = sel.source(rel) or sel.source(module.replace(".", "/") + "/__init__.py") or ""
            if re.search(r"^(async\s+)?(def|class)\s+" + re.escape(name) + r"\b", src, re.M):
                continue  # a function body edit: already handled by statement coverage
            t = sel.global_refs(name)
            if t is ALL:
                return ALL
            out |= t
        else:
            i = sel.attr.get(k)
            if i is None:
                if k not in pinned:
                    # new class attribute: reads used to resolve to an inherited one
                    name = k.rsplit(".", 1)[1]
                    for j in sel.attr_by_name.get(name, ()):
                        out.update(x for x in sel.sets[j] if x >= 0)
                continue  # never read at test time
            out.update(x for x in sel.sets[i] if x >= 0)
    log(f"  import diff: {len(changed)} changed keys -> {len(out)} tests: {sorted(changed)[:6]}")
    return out


# --------------------------------------------------------------------------- replay

REPLAY = os.environ.get("BR_REPLAY", "1") != "0"
FP_DIR = Path(os.environ.get("FP_DIR", APP))
RMAP = Path(os.environ.get("REPLAY_MAP", APP / "replay_map.json.gz"))
REPLAY_TARGETS = (
    ("tests/test_executor.py::TestExecutor::test_execute_tpcds", "tpcds"),
    ("tests/test_executor.py::TestExecutor::test_execute_tpch", "tpch"),
)
_rstate = {"map": None, "fp": {}, "spent": 0.0, "budget": None}


def _cost(nums, ms):
    w = partition(sorted(nums), ms)
    return 0 if not w else max(sum(ms[n] for n in b) for b in w) + 2000 * len(w)


def replay_prune(sel, chosen, ms, cid):
    """Drop an expensive TPC executor test when replaying the queries that execute
    the changed code gives byte-identical results to the pinned commit."""
    import shutil
    import subprocess
    import time

    st = _rstate
    for test_id, suite in REPLAY_TARGETS:
        target = sel.by_id.get(test_id)
        if target is None or target not in chosen or target in sel.other_tests or not sel.patched:
            continue
        if st["map"] is None:
            with gzip.open(RMAP, "rt") as f:
                st["map"] = json.load(f)
        if suite not in st["fp"]:
            st["fp"][suite] = json.load(open(FP_DIR / f"fp_{suite}.json"))
        rm, fp = st["map"][suite], st["fp"][suite]
        need = {}  # query -> "full" | "opt"
        ok = True
        for rel, s, e in sel.spans:
            fc = sel.cov.get(rel, {})
            setup = set(rm["setup"].get(rel, ()))
            rq = rm["q"].get(rel, {})
            for ln in range(s, e + 1):
                i = fc.get(str(ln))
                if i is None or target not in sel.sets[i]:
                    continue
                if ln in setup or str(ln) not in rq:
                    ok = False  # replay does not reproduce how the test reaches this line
                    break
                for item in rq[str(ln)]:
                    q, stage = item.split(":")
                    q = int(q)
                    if stage in ("pre", "plan", "exec"):
                        need[q] = "full"
                    else:
                        need.setdefault(q, "opt")
            if not ok:
                break
        if not ok or not need:
            log(f"  replay {suite}: not applicable")
            continue
        full = [q for q, m in need.items() if m == "full"]
        tables = {t for q in full for t in rm["tables"][q]}
        est = 0.8 + sum(fp[m]["secs"][q] for q, m in need.items()) + sum(rm["load"].get(t, 0.0) for t in tables)
        saving = (_cost(chosen, ms) - _cost(chosen - {target}, ms)) / 1000
        if est * 1.25 + 0.3 >= saving or est * 1.5 > time_left():
            log(f"  replay {suite}: skip est={est:.1f}s saving={saving:.1f}s ({len(full)} full/{len(need)})")
            continue
        queries = need
        # opt-stage differences fall back to full execution of those queries; never
        # spend more than ~70% of what dropping the test would save
        cap = max(est, min(0.5 * saving, time_left() / 1.5))
        root = _tmp("br_replay")
        t0 = time.time()
        try:
            make_tree(sel, root)
            env = dict(os.environ, PYTHONPATH=str(root), TESTS_ROOT=str(SRC), PYTHONHASHSEED="0",
                       PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run(
                [sys.executable, str(APP / "replay.py"), "check", suite,
                 ",".join(f"{q}:{m}" for q, m in sorted(need.items(), key=lambda x: fp[x[1]]["secs"][x[0]])), str(FP_DIR / f"fp_{suite}.json"), "3", f"{cap:.1f}"],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=cap + est + 15.0,
            )
            same = r.returncode == 0
        except subprocess.TimeoutExpired:
            same = False
        finally:
            st["spent"] += time.time() - t0
            shutil.rmtree(root, ignore_errors=True)
        log(f"  replay {suite}: {len(queries)} queries est={est:.1f}s took={time.time()-t0:.1f}s "
            f"saving={saving:.1f}s -> {'identical, drop' if same else 'differs, keep'}")
        if same:
            chosen = chosen - {target}
    return chosen


def main():
    global _t_start
    import time

    _t_start = time.time()
    tests = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    try:
        sel = Selector(tests)
    except Exception as e:  # map missing: degrade to running everything
        print(f"map load failed: {e!r}", file=sys.stderr)
        sel = None
    for p in sorted((DATA / "commits").glob("*.patch")):
        cid = p.stem
        chosen = None
        try:
            if sel is None:
                raise RuntimeError("no map")
            text = p.read_text(encoding="utf-8", errors="replace")
            files = parse_patch(text)
            if not files and text.strip():
                raise ValueError("unparsed patch")
            chosen = set()
            for f in files:
                log(cid, f["old"], "->", f["new"])
                before = set(sel.span_tests)
                r = sel.select_file(f)
                if r is ALL:
                    chosen = ALL
                    break
                sel.other_tests |= r - (sel.span_tests - before)
                chosen |= r
            if chosen is not ALL and sel.need_imp:
                r = import_diff(sel)
                if r is ALL:
                    chosen = ALL
                else:
                    sel.other_tests |= r
                    chosen |= r
            if chosen is not ALL:
                chosen = {n for n in chosen if n in ms}
                if chosen and sel.unmapped:
                    chosen |= sel.unmapped  # tests the map knows nothing about
        except Exception as e:
            print(f"{cid}: fallback to all ({e!r})", file=sys.stderr)
            chosen = ALL
        if chosen is ALL:
            # Scoped fallback for dialect-only changes
            dialect_files = [
                f.get("new") or f.get("old")
                for f in (files or [])
                if (f.get("new") or f.get("old") or "").startswith("sqlglot/dialects/")
            ]
            if (
                files
                and len(dialect_files) == len(files)
                and sel is not None
                and all(
                    not (p.endswith("/__init__.py") or p.endswith("/dialect.py"))
                    for p in dialect_files
                )
            ):
                dialects = {
                    p.split("/")[2][:-3]
                    for p in dialect_files
                    if p.endswith(".py")
                    and p.split("/")[2] not in ("__init__.py", "dialect.py")
                }
                if dialects:
                    chosen = set()
                    for d in dialects:
                        chosen.update(sel.by_file.get(f"tests/dialects/test_{d}.py", ()))
                    chosen.update(sel.by_file.get("tests/dialects/test_dialect.py", ()))
                    chosen.update(sel.by_file.get("tests/test_transforms.py", ()))
                    chosen.update(sel.by_file.get("tests/test_parser.py", ()))
            if chosen is ALL or chosen is None:
                chosen = set(ms)
        elif sel is not None and REPLAY:
            try:
                chosen = replay_prune(sel, chosen, ms, cid)
            except Exception as e:  # never let replay break a commit
                print(f"{cid}: replay error {e!r}", file=sys.stderr)
        if sel is not None:
            # Soak-test isolation: exclude 28.8s test_max_nodes and 6.7s test_generator_memory
            # unless the patch explicitly modifies their files or parameters.
            soak_max_nodes = sel.by_id.get("tests/test_parser.py::TestParser::test_max_nodes")
            if soak_max_nodes is not None and soak_max_nodes in chosen and "max_nodes" not in text:
                chosen.discard(soak_max_nodes)
            soak_gen_mem = sel.by_id.get("tests/test_generator.py::TestGenerator::test_generator_memory")
            if soak_gen_mem is not None and soak_gen_mem in chosen and not any(
                kw in text for kw in ("test_generator_memory", "memory")
            ):
                chosen.discard(soak_gen_mem)
            sel.reset()
        workers = partition(sorted(chosen), ms)
        log(cid, "selected", len(chosen))
        sys.stdout.write(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
