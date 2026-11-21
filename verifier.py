"""In-process test verification with hot-swapped function code.

A zygote imports the pristine sqlglot (and the test dependencies) once. For each
request it forks a child that

  1. replaces the __code__ of every function / closure / lambda whose code the patch
     changes (function-body edits only: the selector only asks for patches whose
     changes are confined to code that does not run at import time),
  2. imports the requested test modules (after the swap, so their import-time code
     runs the patched sqlglot), and
  3. runs the requested tests with unittest (a failing subtest fails the test).

Requests / replies are JSON lines on stdin / stdout:
  {"files": {rel: new_source}, "tests": [pytest ids], "timeout": seconds}
  -> {"failed": [ids], "ok": bool}        ok=false: could not verify (keep all)
"""
import concurrent.futures as cf
import gc
import importlib
import json
import os
import pickle
import pkgutil
import select
import signal
import sys
import types
import unittest

SRC = os.environ.get("SRC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


# ---------------------------------------------------------------- in-process pool
def _rt(obj):
    try:
        return pickle.loads(pickle.dumps(obj))
    except Exception:
        return obj


class SyncExecutor(cf.Executor):
    """Spawned pool workers would re-import the unpatched sqlglot from disk: run
    the work in-process instead, with the same pickling round trips."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **k):
        if initializer is not None:
            initializer(*_rt(initargs))

    def submit(self, fn, /, *args, **kwargs):
        f = cf.Future()
        try:
            fn, args, kwargs = _rt((fn, args, kwargs))
            f.set_result(_rt(fn(*args, **kwargs)))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [_rt(fn(*_rt(a))) for a in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


def _install_shim():
    cf.ProcessPoolExecutor = SyncExecutor
    import concurrent.futures.process as p

    p.ProcessPoolExecutor = SyncExecutor


# ---------------------------------------------------------------- code swapping
class Unswappable(Exception):
    pass


def _codes(co, out):
    out.setdefault(co.co_qualname, []).append(co)
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            _codes(c, out)
    return out


def swap(rel, new_src):
    path = os.path.join(SRC, rel)
    old_src = open(path, encoding="utf-8").read()
    old = _codes(compile(old_src, path, "exec"), {})
    new = _codes(compile(new_src, path, "exec"), {})
    if set(old) != set(new) or any(len(old[q]) != len(new[q]) for q in old):
        raise Unswappable("code structure changed")
    mapping = {}
    for q in old:
        for a, b in zip(old[q], new[q]):
            if a != b:
                if a.co_freevars != b.co_freevars or a.co_cellvars != b.co_cellvars:
                    raise Unswappable(f"closure layout changed in {q}")
                mapping[a] = b
    n = 0
    for f in gc.get_objects():
        if type(f) is types.FunctionType:
            nc = mapping.get(f.__code__)
            if nc is not None:
                f.__code__ = nc
                n += 1
    return n


# ---------------------------------------------------------------- running tests
def _uid(pytest_id):
    path, *rest = pytest_id.split("::")
    return path[:-3].replace("/", "."), rest


def run_tests(ids):
    failed = set()
    by_uid = {}
    suite = unittest.TestSuite()
    for tid in ids:
        mod, rest = _uid(tid)
        m = importlib.import_module(mod)
        obj = m
        for r in rest:
            obj = getattr(obj, r)
        by_uid[".".join([mod] + rest)] = tid
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(".".join([mod] + rest)))
    result = unittest.TestResult()
    suite.run(result)
    for test, _ in result.failures + result.errors + [(t, None) for t in result.unexpectedSuccesses]:
        case = getattr(test, "test_case", test)  # _SubTest -> its TestCase
        tid = getattr(case, "id", lambda: "")()
        hit = by_uid.get(tid)
        if hit is not None:
            failed.add(hit)
        else:
            # setUpClass / module-level errors etc.: be conservative
            failed.update(by_uid.values())
    return sorted(failed)


def handle(req):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)  # tests may print: keep the reply channel clean
        sys.stdout = open(os.devnull, "w")
        gc.unfreeze()
        try:
            for rel, src in req["files"].items():
                swap(rel, src)
            out = {"failed": run_tests(req["tests"]), "ok": True}
        except BaseException as e:  # noqa: BLE001
            out = {"failed": list(req["tests"]), "ok": False, "err": repr(e)[:200]}
        data = json.dumps(out).encode()
        os.write(w, data)
        os._exit(0)
    os.close(w)
    chunks = []
    deadline = req.get("timeout", 10.0)
    while True:
        rl, _, _ = select.select([r], [], [], deadline)
        if not rl:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(r)
            return {"failed": list(req["tests"]), "ok": False, "err": "timeout"}
        b = os.read(r, 1 << 16)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    os.waitpid(pid, 0)
    try:
        return json.loads(b"".join(chunks).decode())
    except Exception:
        return {"failed": list(req["tests"]), "ok": False, "err": "bad child output"}


def main():
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    sys.path.insert(0, SRC)
    _install_shim()
    import sqlglot

    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if not m.name.endswith("__main__"):
            importlib.import_module(m.name)
    for heavy in ("duckdb", "pandas", "numpy", "pandas.testing", "tests", "tests.helpers"):
        try:
            importlib.import_module(heavy)
        except Exception:
            pass
    gc.collect()
    gc.freeze()  # children start from a frozen heap: cheaper forks
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            rep = handle(json.loads(line))
        except Exception as e:  # noqa: BLE001
            rep = {"failed": [], "ok": False, "err": repr(e)[:200]}
        sys.stdout.write(json.dumps(rep) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
