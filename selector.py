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
import copy
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


def same_ast(a, b):
    """_norm(a) == _norm(b) without building the dumps: structural equality ignoring
    positions, stopping at the first difference. Type-strict (True != 1 != 1.0)."""
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if type(x) is not type(y):
            return False
        if isinstance(x, ast.AST):
            for f in x._fields:
                stack.append((getattr(x, f, None), getattr(y, f, None)))
        elif type(x) is list:
            if len(x) != len(y):
                return False
            stack.extend(zip(x, y))
        elif x != y:
            return False
    return True


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
        self.kattr = m.get("kattr", {})
        self.pcov = m.get("pcov", {})
        self.nonp = {int(n): {f: set(ls) for f, ls in d.items()} for n, d in m.get("nonp", {}).items()}
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
        self.spare = set()  # pruned by a finer rule: packed into idle worker time for free
        self.patched = {}  # rel -> patched source
        self.need_imp = False
        self.swap_ok = True  # every change is a function-body edit (code hot-swap safe)
        self.imp_changed = False
        self.table_hit = False  # a file matched the precomputed mutation table
        self.table_ok = True  # at most one sqlglot/*.py file has a real (AST) change

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
                if new.endswith((".py", ".pth")):
                    return ALL  # e.g. a root-level module shadowing the stdlib (cwd is on sys.path)
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
        self.spare |= out - keep
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
            if rel not in self.by_file:
                return ALL
            out = set(self.by_file[rel])
            for u in self.test_importers(rel):  # a broken module breaks its importers too
                out |= set(self.by_file.get(u, ()))
            return out
        if same_ast(old_tree, new_tree):
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
        users = self.test_importers(rel)
        if users:
            # other test modules import this one (e.g. dialect tests subclass
            # tests/dialects/test_dialect.py::Validator): any change can reach them
            log(f"  test module imported by {len(users)} others")
            out = set(self.by_file[rel])
            for u in users:
                out |= set(self.by_file.get(u, ()))
            return out
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

    def test_importers(self, rel):
        """Test modules that import `rel` (transitively), from the tests sources."""
        g = getattr(self, "_timp", None)
        if g is None:
            g = {}
            root = SRC / "tests"
            for p in root.rglob("*.py"):
                r = str(p.relative_to(SRC))
                try:
                    tree = ast.parse(p.read_text(encoding="utf-8"))
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                pkg = r.rsplit("/", 1)[0].replace("/", ".")
                for node in ast.walk(tree):
                    mods = []
                    if isinstance(node, ast.Import):
                        mods = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        base = node.module or ""
                        if node.level:
                            parts = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                            base = ".".join(parts + ([base] if base else []))
                        mods = [base] + [f"{base}.{a.name}" for a in node.names]
                    for m in mods:
                        if m.startswith("tests"):
                            g.setdefault(m.replace(".", "/") + ".py", set()).add(r)
            self._timp = g
        out, todo = set(), [rel]
        while todo:
            for u in g.get(todo.pop(), ()):
                if u not in out and u != rel:
                    out.add(u)
                    todo.append(u)
        return out

    def select_sqlglot_py(self, rel, hunks):
        try:
            src, new, changed, inserts = self._apply(rel, hunks)
        except ValueError:
            log("  apply failed", rel)
            return ALL
        self.patched[rel] = new
        hit = mtable_lookup(rel, src, new) if self.table_ok else None
        if hit is not None:
            self.table_hit = True
            got = {self.by_id[t] for t in hit if t in self.by_id}
            log(f"  mutation table hit: {len(got)} failing")
            return got
        self._pairs = _line_pairs(hunks)
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
        if same_ast(idx.tree, new_tree):
            log("  AST unchanged (comments/formatting)")
            return set()
        if EQUIV and same_ast(_Equiv().visit(ast.parse(self.source(rel))), _Equiv().visit(ast.parse(new))):
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
                    cov = self.cov_span(rel, s, e)
                    inf = self.infected_tests(rel, node, new_idx, c)
                    if inf is not None:
                        keep = cov & (inf | self.inf_incomplete())
                        log(f"  infection {rel}:{c}: {len(cov)} covering -> {len(keep)} infected")
                        self.spare |= cov - keep
                        out |= keep
                    else:
                        out |= cov
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.swap_ok = False  # def header / decorator / defaults
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
            lam = self.lambda_body_edit(idx, c) if kind != "ins" else None
            if lam is not None:
                # edit confined to a lambda's body (e.g. a TRANSFORMS entry): the body
                # runs only when the lambda is called, so its test-time line coverage
                # is exact; the import diff still catches import-time calls/failures
                log(f"  lambda body edit {rel}:{lam.lineno}-{lam.end_lineno}")
                cov = self.cov_span(rel, lam.lineno, lam.end_lineno)
                inf = self.infected_tests(rel, node, new_idx, c)
                if inf is not None:
                    keep = cov & (inf | self.inf_incomplete())
                    log(f"  infection {rel}:{c}: {len(cov)} covering -> {len(keep)} infected")
                    self.spare |= cov - keep
                    cov = keep
                out |= cov
                self.defer_import(swap_safe=True)
                continue
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
            # The tests come from the import diff (changed keys / dict entries -> their
            # test-time readers). Always confirm through the import diff: the attribute may be read while
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

    def lambda_body_edit(self, idx, ln):
        """The innermost lambda whose *body* contains the whole edit on old line ln
        (single-line replacement), else None."""
        pair = getattr(self, "_pairs", {}).get(ln)
        if pair is None:
            return None
        old, new = pair
        p = 0
        while p < min(len(old), len(new)) and old[p] == new[p]:
            p += 1
        q = 0
        while q < min(len(old), len(new)) - p and old[-1 - q] == new[-1 - q]:
            q += 1
        a = len(old[:p].encode())  # ast columns are UTF-8 byte offsets
        b = len(old[: len(old) - q].encode())
        best = None
        for node in ast.walk(idx.tree):
            if not isinstance(node, ast.Lambda):
                continue
            body = node.body
            if (body.lineno, body.col_offset) <= (ln, a) and (ln, b) <= (body.end_lineno, body.end_col_offset):
                if best is None or (node.lineno, node.col_offset) >= (best.lineno, best.col_offset):
                    best = node
        return best

    # ---- weak-mutation infection (infect.json.gz, built by tools/brinfect.py)
    def infect(self):
        d = getattr(self, "_infect", None)
        if d is None:
            d = {}
            try:
                with gzip.open(INFECT, "rt") as f:
                    d = json.load(f)
                tr = [self.by_id.get(i) for i in d["ids"]]
                d["_tr"] = tr
            except Exception as e:
                log(f"  infection map unavailable: {e!r}")
                d = {}
            self._infect = d
        return d

    def inf_set(self, i):
        d = self.infect()
        tr = d["_tr"]
        return {tr[x] for x in d["sets"][i] if tr[x] is not None}

    def inf_incomplete(self):
        d = self.infect()
        return self.inf_set(d["incomplete"]) if d else set()

    def infected_tests(self, rel, old_stmt, new_idx, ln):
        """Tests in which the edited comparison / boolean operator would produce a
        different outcome than the pinned code at least once. None = unknown."""
        if not INFECT_ON:
            return None
        d = self.infect()
        if not d or rel not in d.get("inst", {}):
            return None
        hit = new_idx.innermost(ln)
        if hit is None or type(hit[2]) is not type(old_stmt):
            return None
        site = _single_site_edit(old_stmt, hit[2])
        if site is None:
            return None
        kind, onode, nnode = site
        key = f"{onode.lineno}:{onode.col_offset}:{onode.end_lineno}:{onode.end_col_offset}"
        inst = d["inst"][rel].get(key)
        if kind == "cmp":
            oop, nop = type(onode.ops[0]).__name__, type(nnode.ops[0]).__name__
            if inst != ["cmp", oop]:
                return None
            c = d["cmp"].get(rel, {}).get(key)
            if c is None:
                return set()  # instrumented and never evaluated by any test
            changed_operands = [
                k for k, (a, b) in enumerate(((onode.left, nnode.left), (onode.comparators[0], nnode.comparators[0])))
                if _norm(a) != _norm(b)
            ]
            if not changed_operands:
                if nop in c["imp"] or ("OPQ" in c["imp"] and nop not in ("Is", "IsNot")):
                    return None
                out = self.inf_set(c["alt"][nop]) if nop in c["alt"] else set()
                if nop not in ("Is", "IsNot"):
                    out |= self.inf_set(c["opq"])
                return out
            # literal operand change (optionally with an operator change)
            if c["pairs"] is None or c["imp_pairs"] or c["imp"]:
                return None
            newvals = {}
            for k in changed_operands:
                sub = (nnode.left, nnode.comparators[0])[k]
                try:
                    newvals[k] = ast.literal_eval(sub)
                    ast.literal_eval((onode.left, onode.comparators[0])[k])
                except Exception:
                    return None
            out = self.inf_set(c["opq"])
            ofn, nfn = _CMP_OPS[oop], _CMP_OPS[nop]
            for lrep, rrep, si in c["pairs"]:
                if lrep is None or rrep is None:
                    out |= self.inf_set(si)
                    continue
                try:
                    lv, rv = ast.literal_eval(lrep), ast.literal_eval(rrep)
                except Exception:
                    out |= self.inf_set(si)
                    continue
                nv = [newvals.get(0, lv), newvals.get(1, rv)]
                if _cmp_outcome(ofn, lv, rv) != _cmp_outcome(nfn, nv[0], nv[1]):
                    out |= self.inf_set(si)
            return out
        if kind == "bool":
            oop, nop = type(onode.op).__name__, type(nnode.op).__name__
            if inst != ["bool", oop] or oop == nop:
                return None
            b = d["bool"].get(rel, {}).get(key)
            if b is None:
                return set()
            ok_pat = [True, True] if oop == "And" else [False, False]
            if any(list(p) != ok_pat for p in b["imp"]):
                return None
            out = set()
            for ta, tb, si in b["pat"]:
                if [ta, tb] != ok_pat:
                    out |= self.inf_set(si)
            return out
        return None

    def runs_at_import(self, rel, s, e):
        lines = self._imp_sets.get(rel)
        if lines is None:
            lines = self._imp_sets[rel] = set(self.imp.get(rel, ()))
        return any(ln in lines for ln in range(s, e + 1))

    def defer_import(self, swap_safe=False):
        """The change only acts through import-time state: resolve it after all files
        are seen, by diffing the patched import against the pinned one.
        swap_safe: the edit is code (e.g. a lambda body) that a code hot-swap reproduces
        exactly as long as the import diff finds no import-time state change."""
        self.need_imp = True
        if not swap_safe:
            self.swap_ok = False
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


def _line_pairs(hunks):
    """old line number -> (old text, new text) for '-' runs replaced by an equally
    long '+' run."""
    pairs = {}
    for os_, ol, ns, nl, body in hunks:
        pos = os_
        i = 0
        while i < len(body):
            if body[i][0] == " ":
                pos += 1
                i += 1
                continue
            j = i
            dels = []
            while j < len(body) and body[j][0] == "-":
                dels.append(body[j][1])
                j += 1
            adds = []
            while j < len(body) and body[j][0] == "+":
                adds.append(body[j][1])
                j += 1
            if dels and len(dels) == len(adds):
                for k, (o, n) in enumerate(zip(dels, adds)):
                    pairs[pos + k] = (o, n)
            pos += len(dels)
            i = j if j > i else i + 1
    return pairs


INFECT = Path(os.environ.get("INFECT_FILE", APP / "infect.json.gz"))
INFECT_ON = os.environ.get("BR_INFECT", "1") != "0"
_CMP_OPS = {
    "Eq": lambda a, b: a == b, "NotEq": lambda a, b: a != b, "Lt": lambda a, b: a < b,
    "LtE": lambda a, b: a <= b, "Gt": lambda a, b: a > b, "GtE": lambda a, b: a >= b,
    "Is": lambda a, b: a is b, "IsNot": lambda a, b: a is not b,
    "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b,
}


def _cmp_outcome(fn, a, b):
    try:
        r = fn(a, b)
    except Exception as e:
        return ("E", type(e).__name__, str(e)[:200])
    return r if type(r) is bool else ("V", repr(r))


def _single_site_edit(old, new):
    """If old and new statements differ only inside one single-operator comparison
    (its operator and/or literal operands) or only in the operator of one 2-operand
    and/or, return (kind, old_node, new_node); else None."""
    diffs = []

    def walk(o, n, anc):
        if type(o) is not type(n):
            diffs.append((o, n, anc))
            return
        if isinstance(o, ast.AST):
            here = anc + [(o, n)]
            for f in o._fields:
                a, b = getattr(o, f, None), getattr(n, f, None)
                if isinstance(a, list) or isinstance(b, list):
                    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
                        diffs.append((o, n, anc))
                        continue
                    for x, y in zip(a, b):
                        walk(x, y, here)
                elif isinstance(a, ast.AST) or isinstance(b, ast.AST):
                    walk(a, b, here)
                elif a != b:
                    diffs.append((o, n, anc))
            return
        if o != n:
            diffs.append((o, n, anc))

    walk(old, new, [])
    if not diffs:
        return None
    site = None
    for o, n, anc in diffs:
        chain = anc + ([(o, n)] if isinstance(o, ast.AST) else [])
        found = None
        for oa, na in reversed(chain):
            if isinstance(oa, ast.Compare) and isinstance(na, ast.Compare) and len(oa.ops) == 1 and len(na.ops) == 1:
                found = ("cmp", oa, na)
                break
            if isinstance(oa, ast.BoolOp) and isinstance(na, ast.BoolOp) and len(oa.values) == 2 and len(na.values) == 2:
                found = ("bool", oa, na)
                break
        if found is None:
            return None
        if site is None:
            site = found
        elif site[1] is not found[1]:
            return None
    kind, oa, na = site
    if kind == "bool" and any(_norm(x) != _norm(y) for x, y in zip(oa.values, na.values)):
        return None  # operands changed too
    return site


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


def SPARE_ALWAYS_IDS(sel):
    """Tests worth running whenever there is idle capacity: timing/memory sensitive
    tests whose outcome can be noisy."""
    ids = ("tests/test_generator.py::TestGenerator::test_generator_memory",)
    return {sel.by_id[i] for i in ids if i in sel.by_id}


def fill_slack(workers, spare, ms):
    """Add spare tests to workers below the busiest load, never raising it and never
    adding a worker: the simulated cost is unchanged."""
    loads = [sum(ms[n] for n in w) for w in workers]
    top = max(loads)
    workers = [list(w) for w in workers]
    for n in sorted(spare, key=lambda n: (-ms[n], n)):
        j = min(range(len(workers)), key=loads.__getitem__)
        if loads[j] + ms[n] <= top:
            workers[j].append(n)
            loads[j] += ms[n]
    return workers


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
IMP_GRAPH = Path(os.environ.get("IMP_GRAPH", APP / "imports.json"))
_imp_state = {"pinned": None, "rdeps": None}
REGISTRY_KEYS = frozenset({
    "sqlglot.dialects.dialect:_Dialect._classes",
    "sqlglot.optimizer.annotate_types::_COERCES_TO",
    "sqlglot.optimizer.annotate_types:TypeAnnotator.COERCES_TO",
})  # must match impdiff.REGISTRY_KEYS


def _key_module(k):
    return k.split("::", 1)[0] if "::" in k else k.split(":", 1)[0]


def affected_modules(rels):
    """Patched modules plus every module that (transitively) imports one of them:
    the only modules whose import-time state a patch can change (besides the
    registries in REGISTRY_KEYS). None = everything."""
    try:
        if _imp_state["rdeps"] is None:
            graph = json.load(open(IMP_GRAPH))
            rdeps = {}
            for m, deps in graph.items():
                for d in deps:
                    rdeps.setdefault(d, set()).add(m)
            _imp_state["rdeps"] = rdeps
    except Exception:
        return None
    rdeps = _imp_state["rdeps"]
    todo = []
    for rel in rels:
        m = rel[:-3].replace("/", ".")
        todo.append(m[: -len(".__init__")] if m.endswith(".__init__") else m)
    seen = set(todo)
    while todo:
        m = todo.pop()
        for r in rdeps.get(m, ()):
            if r not in seen:
                seen.add(r)
                todo.append(r)
    return None if len(seen) > 120 else seen
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
    pinned_all = _imp_state["pinned"]
    pinned, pinned_ent = pinned_all["fp"], pinned_all["ent"]
    root = _tmp("br_imp")
    out_file = _tmp("br_imp.json")
    try:
        make_tree(sel, root)
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
        only = affected_modules(sel.patched)
        r = subprocess.run([sys.executable, str(APP / "impdiff.py"), "dump", str(out_file),
                            ",".join(sorted(only)) if only else ""], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if r.returncode != 0:
            log("  import diff: patched import failed")
            return ALL
        new_all = json.load(open(out_file))
        new, new_ent = new_all["fp"], new_all["ent"]
    except Exception as e:
        log(f"  import diff error {e!r}")
        return ALL
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            os.unlink(out_file)
        except OSError:
            pass
    keys = set(pinned) | set(new)
    if only:
        keys = {k for k in keys if _key_module(k) in only or k in REGISTRY_KEYS}
    changed = [k for k in keys if pinned.get(k) != new.get(k)]
    sel.imp_changed = bool(changed)
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
            ka = sel.kattr.get(k)
            if ka is not None and k in pinned_ent and k in new_ent:
                # dict attribute: only tests that read a changed entry (or the whole dict)
                po, pn = pinned_ent[k], new_ent[k]
                changed_entries = [e for e in set(po) | set(pn) if po.get(e) != pn.get(e)]
                t = set(sel.sets[ka["*"]])
                for e in changed_entries:
                    j = ka["k"].get(e)
                    if j is not None:
                        t.update(sel.sets[j])
                t.discard(-1)  # import-time readers: their effect shows up as other changed keys
                if k in sel.attr:
                    sel.spare.update(x for x in sel.sets[sel.attr[k]] if x >= 0)
                log(f"  import diff: {k} entries {changed_entries[:3]} -> {len(t)} tests")
                out |= t
                continue
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

# V10 turned these off (measured on synthetic mutants); the official practice set lost 25 s
# simulated for 3.3 s CPU -> back on (V10.2)
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
            sel.spare.add(target)
    return chosen


# --------------------------------------------------------------------------- verification

VERIFY = os.environ.get("BR_VERIFY", "1") != "0"
VERIFY_ALL = os.environ.get("BR_VERIFY_ALL") == "1"  # local fidelity checks only
REIMPORT = os.environ.get("BR_REIMPORT", "1") != "0"
V_TEST_CAP = float(os.environ.get("BR_V_TEST_CAP", "1500"))  # ms per verified test
V_RUN_CAP = float(os.environ.get("BR_V_RUN_CAP", "3.0"))  # s of recorded test time per commit
V_SCALE = float(os.environ.get("BR_V_SCALE", "1.5"))  # verifier CPU / recorded test time
V_MARGIN = float(os.environ.get("BR_V_MARGIN", "0.5"))  # min expected net gain (s)
V_OVER = (0.1, 0.25)  # fixed cost per request: swap, reimport (fork, swap / re-import)
V_BASE = (0.03, 0.10)  # part of each measured single-test cost that is fixed overhead
VCOST = Path(os.environ.get("VCOST_FILE", APP / "vcost.json.gz"))
_vc = {}


def _vcost():
    if "d" not in _vc:
        try:
            with gzip.open(VCOST, "rt") as f:
                _vc["d"] = json.load(f)
        except Exception:
            _vc["d"] = {}
    return _vc["d"]
# never verified (kept if selected): timing / memory sensitive or process-level tests
NO_VERIFY = (
    "tests/test_generator.py::TestGenerator::test_generator_memory",
    "tests/test_parser.py::TestParser::test_max_nodes",
    # runs `python -c "import sqlglot ..."`: a subprocess sees the unpatched files
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
)
NO_VERIFY_FILES = ("tests/test_docs.py", "tests/test_sqlglotc.py", "tests/test_dialect_imports.py",
                   "tests/test_dialect_entry_points.py", "tests/test_build.py")
_vstate = {"proc": None, "dead": False}


def _verifier():
    import select
    import subprocess

    st = _vstate
    if st["dead"]:
        return None
    if st["proc"] is None:
        env = dict(os.environ, SRC_DIR=str(SRC), SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        try:
            p = subprocess.Popen([sys.executable, str(APP / "verifier.py")], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
            r, _, _ = select.select([p.stdout], [], [], 40)
            if not r or not json.loads(p.stdout.readline() or b"{}").get("ready"):
                raise RuntimeError("verifier not ready")
        except Exception as e:
            log(f"  verifier unavailable: {e!r}")
            st["dead"] = True
            return None
        st["proc"] = p
    return st["proc"]


def verify_prune(sel, chosen, ms, cid):
    """Run small, cheap selected tests against the patched code (hot-swapped into a
    warm interpreter) and drop the ones that pass."""
    import select
    import time

    if not VERIFY or not sel.patched or not chosen or sel.table_hit:
        return chosen
    # code hot-swap when every change is a function body; otherwise re-import
    # sqlglot from the patched sources in the child (full fidelity, ~0.4 s more)
    mode = "swap" if sel.swap_ok and os.environ.get("BR_FORCE_REIMPORT") != "1" else "reimport"
    if mode == "reimport" and not REIMPORT:
        return chosen
    mi = 0 if mode == "swap" else 1
    over, base = V_OVER[mi], V_BASE[mi]
    vc = _vcost()

    def cost(n):  # verifier CPU for one test in a forked child (measured offline)
        m = vc.get(sel.ids_of(n))
        return max(m[mi] - base, 0.005) if m else V_SCALE * ms[n] / 1000 + 0.02

    # candidates: cheapest first; verify the prefix with the best expected net gain
    # (simulated-time saving minus the verifier's CPU -- both count 1:1 in the score)
    elig = [n for n in sorted(chosen, key=lambda n: (cost(n), n))
            if sel.ids_of(n) not in NO_VERIFY and sel.ids_of(n).split("::")[0] not in NO_VERIFY_FILES]
    best, best_gain, cand, run, saving = 0, None, [], 0.0, 0.0
    acc = 0.0
    for k, n in enumerate(elig, 1):
        if not VERIFY_ALL and (ms[n] > V_TEST_CAP or acc + cost(n) > V_RUN_CAP):
            break
        acc += cost(n)
        sv = (_cost(chosen, ms) - _cost(chosen - set(elig[:k]), ms)) / 1000
        g = sv - acc - over
        if best_gain is None or g > best_gain:
            best, best_gain, run, saving = k, g, acc, sv
    cand = elig if VERIFY_ALL else elig[:best]
    if VERIFY_ALL:
        run = sum(ms[n] for n in cand) / 1000
        saving = (_cost(chosen, ms) - _cost(chosen - set(cand), ms)) / 1000
    if not cand:
        return chosen
    if not VERIFY_ALL and (best_gain < V_MARGIN or 3 * run + 2 + over > time_left()):
        log(f"  verify[{mode}]: skip {len(cand)} tests run={run:.2f}s saving={saving:.1f}s")
        return chosen
    p = _verifier()
    if p is None:
        return chosen
    t0 = time.time()
    # a mutant can make a test slow: the child is killed at twice the estimate
    tmo = 3 * run + 5 if VERIFY_ALL else 2 * (run + over) + 1
    req = {"files": dict(sel.patched), "tests": [sel.ids_of(n) for n in cand], "timeout": tmo,
           "mode": mode, "failfast": not VERIFY_ALL}
    stalled = False
    try:
        p.stdin.write((json.dumps(req) + "\n").encode())
        p.stdin.flush()
        ready, _, _ = select.select([p.stdout], [], [], req["timeout"] + 10)
        if ready:
            rep = json.loads(p.stdout.readline())
        else:
            stalled, rep = True, {"ok": False, "err": "zygote stalled"}
    except Exception as e:
        stalled, rep = True, {"ok": False, "err": repr(e)}
    if stalled:
        _vstate["dead"] = True
        try:
            p.kill()
        except Exception:
            pass
    if not rep.get("ok"):
        log(f"  verify: could not verify ({rep.get('err', '')}) took={time.time()-t0:.2f}s")
        return chosen
    failed = {sel.by_id[t] for t in rep.get("failed", []) if t in sel.by_id}
    passed = {sel.by_id[t] for t in rep.get("passed", []) if t in sel.by_id} & set(cand)
    log(f"  verify[{mode}]: {len(cand)} tests run={run:.2f}s took={time.time()-t0:.2f}s "
        f"saving<={saving:.1f}s -> {len(passed)} pass, {len(failed)} fail")
    sel.spare |= passed
    return chosen - passed


MAX_NODES_ID = "tests/test_parser.py::TestParser::test_max_nodes"
MAX_NODES_FAST = os.environ.get("BR_MAXNODES", "1") != "0"


def max_nodes_prune(sel, chosen, ms, cid):
    """Experiment (agy §61): when the runtime selection keeps test_max_nodes (28.8 s, never
    verified), run a scaled-down model of it against the patched code; drop it if that passes."""
    import select
    import time

    n = sel.by_id.get(MAX_NODES_ID)
    if not MAX_NODES_FAST or n is None or n not in chosen or not sel.patched or sel.table_hit:
        return chosen
    if 3.0 > time_left():
        return chosen
    p = _verifier()
    if p is None:
        return chosen
    t0 = time.time()
    req = {"files": dict(sel.patched), "tests": [MAX_NODES_ID], "timeout": 8.0,
           "mode": "swap" if sel.swap_ok else "reimport", "check": "max_nodes"}
    try:
        p.stdin.write((json.dumps(req) + "\n").encode())
        p.stdin.flush()
        ready, _, _ = select.select([p.stdout], [], [], 20)
        rep = json.loads(p.stdout.readline()) if ready else {"ok": False}
    except Exception:
        rep = {"ok": False}
    ok = rep.get("ok") and MAX_NODES_ID in rep.get("passed", [])
    log(f"  max_nodes check: {'pass -> drop' if ok else 'keep'} took={time.time()-t0:.2f}s")
    return chosen - {n} if ok else chosen


CASES = Path(os.environ.get("CASES_FILE", APP / "cases.pkl.gz"))
_cstate = {"fp": None}


def case_prune(sel, chosen, ms, cid):
    """Drop check_file-based optimizer tests whose changed lines run only inside
    specific fixture cases, when replaying those cases gives identical results."""
    import subprocess
    import time

    if not sel.patched or not sel.pcov:
        return chosen
    if _cstate["fp"] is None:
        _cstate["fp"] = json.load(open(FP_DIR / "fp_cases.json"))
    fps = _cstate["fp"]
    spec = {}
    for n in sorted(chosen):
        tid = sel.ids_of(n)
        if tid not in fps or n in sel.other_tests or n not in sel.span_tests:
            continue
        nonp = sel.nonp.get(n, {})
        bits = 0
        ok = True
        for rel, s, e in sel.spans:
            fc = sel.cov.get(rel, {})
            pc = sel.pcov.get(rel, {})
            np_ = nonp.get(rel, ())
            for ln in range(s, e + 1):
                i = fc.get(str(ln))
                if i is None or n not in sel.sets[i]:
                    continue
                b = pc.get(str(ln), {}).get(str(n))
                if ln in np_ or b is None:
                    ok = False  # runs outside the replayable cases
                    break
                bits |= int(b, 16)
            if not ok:
                break
        if ok and bits:
            spec[tid] = [k for k in range(bits.bit_length()) if bits >> k & 1]
    if not spec:
        return chosen
    cand = {sel.by_id[t] for t in spec}
    saving = (_cost(chosen, ms) - _cost(chosen - cand, ms)) / 1000
    est = 0.6 + sum(fps[t]["secs"][k] for t, ks in spec.items() for k in ks)
    if est * 1.25 + 0.2 >= saving or est * 1.5 > time_left():
        log(f"  case replay: skip {len(spec)} tests est={est:.1f}s saving={saving:.1f}s")
        return chosen
    root = _tmp("br_cases")
    spec_file = _tmp("br_cases.json")
    t0 = time.time()
    try:
        make_tree(sel, root)
        with open(spec_file, "w") as f:
            json.dump(spec, f)
        env = dict(os.environ, PYTHONPATH=str(root), TESTS_ROOT=str(SRC), PYTHONHASHSEED="0",
                   PYTHONDONTWRITEBYTECODE="1")
        cap = max(est, min(0.6 * saving, time_left() / 1.5))
        r = subprocess.run(
            [sys.executable, str(APP / "casereplay.py"), "check", str(CASES), str(FP_DIR / "fp_cases.json"),
             str(spec_file), f"{cap:.1f}"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=cap + 15.0,
        )
        same = json.loads(r.stdout.decode() or "[]") if r.returncode == 0 else []
    except Exception as e:
        log(f"  case replay error {e!r}")
        same = []
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
        try:
            os.unlink(spec_file)
        except OSError:
            pass
    drop = {sel.by_id[t] for t in same if t in sel.by_id}
    log(f"  case replay: {sum(map(len, spec.values()))} cases of {len(spec)} tests est={est:.1f}s "
        f"took={time.time()-t0:.1f}s saving={saving:.1f}s -> drop {len(drop)}")
    sel.spare |= drop
    return chosen - drop


# --------------------------------------------------------------------------- mutation table
# Precomputed offline (tools/enum_mutants.py + tools/label_table.py): for every
# single-line mutant of sqlglot/*.py in the operator space of the sample (comparison
# swaps, and/or, True/False, int +-1, `not` removal, string tweak) whose selection is
# not free, the tests that actually fail when it is applied. Keyed by
# sha1(rel, line number, new line). Exact match only; anything else takes the usual path.
MTABLE = Path(os.environ.get("MTABLE_FILE", APP / "mtable.json.gz"))
_mt = {}


def table_ok(sel, files):
    """The table was labelled with one mutant on an otherwise pristine tree: use it only
    when no other sqlglot/*.py file in the commit changes code (comment-only edits are
    fine; fixture / test edits keep their own reader-based selection)."""
    code = 0
    for f in files:
        rel = f["new"] or f["old"] or ""
        if not rel.startswith("sqlglot/"):
            continue
        if f["old"] != f["new"] or not rel.endswith(".py"):
            return False
        try:
            src, new, _, _ = sel._apply(rel, f["hunks"])
            same = same_ast(ast.parse(src), ast.parse(new))
        except Exception:
            same = False
        code += not same
    return code <= 1


def mtable_key(rel, ln, new_line):
    import hashlib

    return hashlib.sha1(f"{rel}\0{ln}\0{new_line}".encode()).hexdigest()[:16]


def mtable_lookup(rel, src, new):
    if os.environ.get("BR_MTABLE", "1") == "0":
        return None
    if "d" not in _mt:
        try:
            with gzip.open(MTABLE, "rt") as f:
                d = json.load(f)
            _mt["d"] = (d["keys"], d["sets"], d["ids"])
        except Exception:
            _mt["d"] = None
    if _mt["d"] is None:
        return None
    a, b = src.split("\n"), new.split("\n")
    if len(a) != len(b):
        return None
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) != 1:
        return None
    i = diff[0]
    keys, sets, ids = _mt["d"]
    k = keys.get(mtable_key(rel, i + 1, b[i] + "\n"))
    if k is None:
        return None
    return [ids[t] for t in sets[k]]


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
            if len(files) > 1:
                sel.table_ok = table_ok(sel, files)
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
                if r is ALL or sel.imp_changed:
                    sel.swap_ok = False  # import-time state differs: a code swap can't reproduce it
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
        elif sel is not None and REPLAY and not sel.table_hit:
            try:
                chosen = replay_prune(sel, chosen, ms, cid)
            except Exception as e:  # never let replay break a commit
                print(f"{cid}: replay error {e!r}", file=sys.stderr)
            try:
                if all((f["new"] or "").startswith("sqlglot/") and (f["new"] or "").endswith(".py")
                       and f["old"] == f["new"] for f in files):
                    # removed sorted()/sort(): order may depend on the hash seed, which a
                    # single verifier process cannot explore -> never prune by verification
                    unsort = re.search(r"^-(?!--).*(?:\bsorted\(|\.sort\()", text, re.M) and not \
                        re.search(r"^\+(?!\+\+).*(?:\bsorted\(|\.sort\()", text, re.M)
                    if not unsort:
                        chosen = verify_prune(sel, chosen, ms, cid)
                        chosen = max_nodes_prune(sel, chosen, ms, cid)
            except Exception as e:
                print(f"{cid}: verify error {e!r}", file=sys.stderr)
            if os.environ.get("BR_CASES") == "1":  # off: measured CPU > saving
                try:
                    chosen = case_prune(sel, chosen, ms, cid)
                except Exception as e:
                    print(f"{cid}: case replay error {e!r}", file=sys.stderr)
        # NOTE: no "soak-test quarantine". Dropping test_max_nodes /
        # test_generator_memory unconditionally is unsound: test_max_nodes also
        # asserts that ordinary queries parse, so it fails whenever basic parsing or
        # the import breaks (verified on a tokenizer_core.py mutant: 300 s miss).
        # PROBE C (not a real selector): empty selection for table-miss commits whose every
        # sqlglot/*.py change is a clean single-line replacement (same line count, one line
        # differs) or comment-only -> "commits missed" counts table misses from unknown operators
        if sel is not None and not sel.table_hit and chosen and files and any(
                (f.get("new") or "").startswith("sqlglot/") for f in files):
            single, has_code = True, False
            for f in files:
                rel = f.get("new") or ""
                if not (rel.startswith("sqlglot/") and rel.endswith(".py")):
                    continue
                try:
                    src, new, _, _ = sel._apply(rel, f["hunks"])
                except Exception:
                    single = False
                    break
                a_, b_ = src.split("\n"), new.split("\n")
                if same_ast(ast.parse(src), ast.parse(new)):
                    continue
                if len(a_) != len(b_) or sum(x != y for x, y in zip(a_, b_)) != 1:
                    single = False
                    break
                has_code = True
            if single and has_code:
                log("  PROBE C: single-line table miss -> empty")
                chosen = set()
        workers = partition(sorted(chosen), ms)
        if sel is not None:
            if workers:
                spare = (sel.spare | SPARE_ALWAYS_IDS(sel)) - chosen
                workers = fill_slack(workers, {n for n in spare if n in ms}, ms)
            sel.reset()
        log(cid, "selected", len(chosen))
        sys.stdout.write(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    p = _vstate.get("proc")
    if p is not None:  # let the verifier exit (and be accounted for) cleanly
        try:
            p.stdin.close()
            p.wait(timeout=10)
        except Exception:
            p.kill()


if __name__ == "__main__":
    main()
