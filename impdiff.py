"""Differential import: structural fingerprints of every class attribute and module
global of sqlglot after import.

  python impdiff.py dump <out.json>     (sqlglot root on PYTHONPATH; PYTHONHASHSEED=0)

Keys match tools/brattr.py: "<module>:<Class.qualname>.<attr>" for class-body values
and "<module>::<name>" for module globals. Comparing a patched import with the pinned
one tells which import-time state a patch changes, including state derived at import
(e.g. metaclass-built tries), so the class-attribute read map can pick the tests.
Exit code 3 = the patched sqlglot failed to import.
"""
import enum
import functools
import hashlib
import importlib
import inspect
import json
import unittest  # noqa: F401  before sqlglot: expressions.core.UNITTEST must match pytest (strict arg checks)
import pkgutil
import re
import sys
import types

_SKIP_CLASS_TYPES = (
    types.FunctionType,
    staticmethod,
    classmethod,
    property,
    types.MemberDescriptorType,
    types.GetSetDescriptorType,
    type,
)


def kr(k):
    """Canonical dict-entry key representation (must match tools/brattr.py)."""
    t = type(k)
    if k is None or t in (bool, int, float, str, bytes):
        return f"{t.__name__}:{k!r}"
    if isinstance(k, enum.Enum):
        return f"E:{t.__module__}.{t.__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"T:{k.__module__}.{k.__qualname__}"
    if t is tuple:
        return "t(" + ",".join(kr(x) for x in k) + ")"
    return None


def _named_ref(fn):
    """'module.qualname' if fn is the object found at that path, else None
    (closures, lambdas, decorated wrappers)."""
    q = getattr(fn, "__qualname__", "")
    if "<" in q:
        return None
    mod = sys.modules.get(getattr(fn, "__module__", None) or "")
    obj = mod
    for part in q.split("."):
        obj = getattr(obj, part, None) if obj is not None else None
        if obj is None:
            return None
    obj = getattr(obj, "__func__", obj)
    return f"{fn.__module__}.{q}" if obj is fn else None


class FP:
    def __init__(self):
        self.memo = {}
        self.active = set()

    def __call__(self, v, depth=0):
        t = type(v)
        if v is None or t in (bool, int, float, str, bytes, complex):
            return f"{t.__name__}:{v!r}"
        if t is types.FunctionType:
            ref = _named_ref(v)
            if ref is not None:
                # a named module/class-level function: referenced from other values
                # (aliases, RULES tuples, default arguments) by name, not content; its
                # own code is covered by statement coverage (see content())
                return "FR:" + ref
        k = id(v)
        m = self.memo.get(k)
        if m is not None:
            return m[1]
        if k in self.active:
            return "<cycle>"
        if depth > 120:  # deepest sqlglot dict nesting is ~20 (time tries)
            return "<deep>"
        self.active.add(k)
        try:
            r = self._fp(v, t, depth + 1)
        finally:
            self.active.discard(k)
        if len(r) > 64:
            r = "#" + hashlib.sha1(r.encode()).hexdigest()
        self.memo[k] = (v, r)  # keep v alive so its id cannot be reused
        return r

    def _fp(self, v, t, d):
        f = self
        if isinstance(v, enum.Enum):
            return f"E:{t.__module__}.{t.__qualname__}.{v.name}"
        if isinstance(v, type):
            return f"T:{v.__module__}.{v.__qualname__}"
        if isinstance(v, (tuple, list)):
            items = [f(x, d) for x in v]
            if items and all(isinstance(x, type) for x in v):
                items.sort()  # tuples of classes are often built from sets (id-hashed)
            return f"{t.__name__}(" + ",".join(items) + ")"
        if isinstance(v, (set, frozenset)):
            return f"{t.__name__}{{" + ",".join(sorted(f(x, d) for x in v)) + "}"
        if isinstance(v, dict):
            # order-insensitive: several dicts are built by iterating sets of classes
            # (id-hashed), so their insertion order differs from run to run
            return f"{t.__name__}{{" + ",".join(sorted(f"{f(a, d)}:{f(b, d)}" for a, b in v.items())) + "}"
        if isinstance(v, types.FunctionType):
            # No bytecode: what a function's *code* does is the job of statement
            # coverage (its lines run at test time, or at import time and then show
            # up in derived values). Only import-time *data* bound to it counts here.
            parts = [v.__qualname__, ",".join(v.__code__.co_varnames[: v.__code__.co_argcount])]
            if v.__defaults__:
                parts.append(f(v.__defaults__, d))
            if v.__kwdefaults__:
                parts.append(f(v.__kwdefaults__, d))
            if v.__closure__:
                cells = []
                for c in v.__closure__:
                    try:
                        cells.append(f(c.cell_contents, d))
                    except ValueError:
                        cells.append("<empty>")
                parts.append("|".join(cells))
            return "F:" + "/".join(parts)
        if isinstance(v, types.CodeType):
            return self.code(v, d)
        if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            return f"B:{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', repr(v))}"
        if isinstance(v, types.MethodType):
            return f"M:{f(v.__func__, d)}@{type(v.__self__).__qualname__}"
        if isinstance(v, functools.partial):
            return f"P:{f(v.func, d)}:{f(v.args, d)}:{f(v.keywords, d)}"
        if isinstance(v, re.Pattern):
            return f"R:{v.pattern!r}:{v.flags}"
        if isinstance(v, (staticmethod, classmethod)):
            return f"{t.__name__}:{f(v.__func__, d)}"
        if isinstance(v, property):
            return f"prop:{f(v.fget, d)}:{f(v.fset, d)}"
        if isinstance(v, types.ModuleType):
            return f"mod:{v.__name__}"
        state = getattr(v, "__dict__", None)
        out = f"O:{t.__module__}.{t.__qualname__}"
        if state is not None:
            out += f(dict(sorted(state.items())), d)
        slots = []
        for c in t.__mro__:
            s = c.__dict__.get("__slots__", ())
            slots.extend([s] if isinstance(s, str) else s)
        if slots:
            out += "S(" + ",".join(f(getattr(v, s, None), d) for s in slots if s not in ("__dict__", "__weakref__")) + ")"
        if state is None and not slots:
            try:
                out += ":" + repr(v)[:200].split(" at 0x")[0]
            except Exception:
                pass
        return out

    def content(self, fn):
        r = self._fp(fn, type(fn), 1)
        return "#" + hashlib.sha1(r.encode()).hexdigest() if len(r) > 64 else r

    def code(self, c, d):
        consts = ",".join(self.code(x, d) if isinstance(x, types.CodeType) else self(x, d) for x in c.co_consts)
        return f"C:{c.co_code.hex()}:{consts}:{','.join(c.co_names)}"


def _keys_of(mod, fp, seen, want=None, ent=None):
    """Yield (key, fingerprint) for module `mod`'s globals and the classes it
    defines/exposes. want(key) -> bool filters before fingerprinting; dict entries
    of dict-valued class attributes go to `ent`."""
    for name, val in list(vars(mod).items()):
        if name.startswith("__") or isinstance(val, types.ModuleType) or inspect.isclass(val):
            continue
        key = f"{mod.__name__}::{name}"
        if want is not None and not want(key):
            continue
        if isinstance(val, types.FunctionType) and _named_ref(val) == f"{mod.__name__}.{name}":
            yield key, fp.content(val)  # its defining module
        else:
            yield key, fp(val)
    for obj in list(vars(mod).values()):
        if not inspect.isclass(obj) or id(obj) in seen:
            continue
        if not (getattr(obj, "__module__", "") or "").startswith("sqlglot"):
            continue
        seen.add(id(obj))
        if issubclass(obj, enum.Enum):
            key = f"{obj.__module__}:{obj.__qualname__}.__members__"
            if want is None or want(key):
                yield key, fp([(k, fp(v.value)) for k, v in obj.__members__.items()])
            continue
        stack = [obj]
        while stack:
            c = stack.pop()
            key = f"{c.__module__}:{c.__qualname__}.__mro__"
            if want is None or want(key):
                # several classes can share module+qualname (created dynamically):
                # fingerprint the sorted union so enumeration order cannot matter
                mro = ",".join(f"{b.__module__}.{b.__qualname__}" for b in c.__mro__)
                acc = _MRO.setdefault(key, set())
                acc.add(mro)
                yield key, "|".join(sorted(acc))
            for name, val in list(vars(c).items()):
                if inspect.isclass(val) and id(val) not in seen and (
                    getattr(val, "__module__", "") or ""
                ).startswith("sqlglot"):
                    seen.add(id(val))
                    if not issubclass(val, enum.Enum):
                        stack.append(val)
                if name.startswith("__") or isinstance(val, _SKIP_CLASS_TYPES):
                    continue
                if hasattr(type(val), "__get__"):
                    continue
                k = f"{c.__module__}:{c.__qualname__}.{name}"
                if want is not None and not want(k):
                    continue
                yield k, fp(val)
                if ent is not None and type(val) is dict:
                    e = {}
                    for a_, b_ in val.items():
                        ka = kr(a_)
                        if ka is None:
                            e = None
                            break
                        e[ka] = fp(b_)
                    if e is not None:
                        ent[k] = e


# Keys whose values change when *other* modules are imported (found offline with
# tools/registries.py at the pinned commit). Always compared.
REGISTRY_KEYS = frozenset({
    "sqlglot.dialects.dialect:_Dialect._classes",
    "sqlglot.optimizer.annotate_types::_COERCES_TO",
    "sqlglot.optimizer.annotate_types:TypeAnnotator.COERCES_TO",
})


_MRO = {}


def _module_of(key):
    return key.split("::", 1)[0] if "::" in key else key.split(":", 1)[0]


def dump(only=None):
    """Fingerprints after importing sqlglot. only: set of module names to import and
    fingerprint (keys of other modules are skipped); None = everything."""
    import sqlglot

    # always import everything: registries filled at import time (REGISTRY_KEYS)
    # depend on which modules were imported
    names = [m.name for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.") if not m.name.endswith("__main__")]
    for n in names:
        importlib.import_module(n)
    if only is None:
        mods = [sqlglot] + [sys.modules[n] for n in names]
        want = None
    else:
        scan = set(only) | {_module_of(k) for k in REGISTRY_KEYS}
        mods = [sys.modules[n] for n in ["sqlglot"] + names if n in scan and n in sys.modules]
        want = lambda k: _module_of(k) in only or k in REGISTRY_KEYS  # noqa: E731
    fp = FP()
    out, ent, seen = {}, {}, set()
    for mod in mods:
        for k, v in _keys_of(mod, fp, seen, want, ent):
            out[k] = v
    return {"fp": out, "ent": ent}


def import_graph(src_root):
    """module -> sorted list of sqlglot modules it imports (anywhere in the file,
    including function-level imports), resolved statically with ast."""
    import ast
    import os

    mods = {}
    base = os.path.join(src_root, "sqlglot")
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), src_root)[:-3].replace(os.sep, ".")
                mods[rel[: -len(".__init__")] if rel.endswith(".__init__") else rel] = os.path.join(root, f)
    graph = {}
    for name, path in mods.items():
        is_pkg = path.endswith("__init__.py")
        pkg = name if is_pkg else name.rsplit(".", 1)[0]
        deps = set()
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    deps.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = pkg.split(".")
                    parts = parts[: len(parts) - node.level + 1]
                    target = ".".join(parts + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                deps.add(target)
                for a in node.names:
                    deps.add(f"{target}.{a.name}")
        # importing a.b.c also runs a and a.b
        full = set()
        for d in deps:
            parts = d.split(".")
            for i in range(1, len(parts) + 1):
                full.add(".".join(parts[:i]))
        graph[name] = sorted(d for d in full if d in mods and d != name)
    return graph


def main():
    if sys.argv[1] == "graph":
        with open(sys.argv[3], "w") as f:
            json.dump(import_graph(sys.argv[2]), f)
        return
    if sys.argv[1] == "dump":
        only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None
        try:
            out = dump(only)
        except BaseException as e:  # noqa: BLE001
            print(f"impdiff: import failed: {e!r}", file=sys.stderr)
            sys.exit(3)
        with open(sys.argv[2], "w") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
