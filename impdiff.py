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
        if depth > 40:
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
            parts = [v.__qualname__, self.code(v.__code__, d)]
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


def dump():
    import sqlglot

    mods = [sqlglot]
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if m.name.endswith("__main__"):
            continue
        mods.append(importlib.import_module(m.name))
    fp = FP()
    out = {}
    seen = set()
    for mod in mods:
        for name, val in list(vars(mod).items()):
            if name.startswith("__"):
                continue
            if isinstance(val, types.ModuleType):
                continue
            if inspect.isclass(val):
                continue
            if isinstance(val, types.FunctionType) and _named_ref(val) == f"{mod.__name__}.{name}":
                out[f"{mod.__name__}::{name}"] = fp.content(val)  # its defining module
            else:
                out[f"{mod.__name__}::{name}"] = fp(val)
        for obj in list(vars(mod).values()):
            if not inspect.isclass(obj) or id(obj) in seen:
                continue
            if not (getattr(obj, "__module__", "") or "").startswith("sqlglot"):
                continue
            seen.add(id(obj))
            if issubclass(obj, enum.Enum):
                out[f"{obj.__module__}:{obj.__qualname__}.__members__"] = fp(
                    [(k, fp(v.value)) for k, v in obj.__members__.items()]
                )
                continue
            stack = [obj]
            while stack:
                c = stack.pop()
                out[f"{c.__module__}:{c.__qualname__}.__mro__"] = ",".join(
                    f"{b.__module__}.{b.__qualname__}" for b in c.__mro__
                )
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
                    out[f"{c.__module__}:{c.__qualname__}.{name}"] = fp(val)
    return out


def main():
    if sys.argv[1] == "dump":
        try:
            out = dump()
        except BaseException as e:  # noqa: BLE001
            print(f"impdiff: import failed: {e!r}", file=sys.stderr)
            sys.exit(3)
        with open(sys.argv[2], "w") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
