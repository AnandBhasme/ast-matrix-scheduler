"""In-process test verification with hot-swapped function code.

A zygote imports the pristine sqlglot (and the test dependencies) once. For each
request it forks a child that

  1. mode "swap": replaces the __code__ of every function / closure / lambda whose
     code the patch changes (function-body edits only), or
     mode "reimport": forgets sqlglot and the test modules and re-imports them from
     the patched sources (any in-place edit, including import-time state),
  2. imports the requested test modules (after 1., so their import-time code runs
     the patched sqlglot), and
  3. runs the requested tests with unittest (a failing subtest fails the test);
     with "failfast" one by one in the given order, stopping at the first failure.

Requests / replies are JSON lines on stdin / stdout:
  {"files": {rel: new_source}, "tests": [pytest ids], "timeout": s, "mode": m, "failfast": b}
  -> {"failed": [ids], "passed": [ids], "ok": bool}   ok=false: could not verify (keep all)
"""
import concurrent.futures as cf
import gc
import importlib
import io
import importlib.machinery
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
    # exactly what a spawn pool does to arguments and results: a pickling failure
    # must surface (as it would in the real pool), never be papered over
    return pickle.loads(pickle.dumps(obj))


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
    with io.open_code(path) as fh:  # never an overlay: always the pristine file
        old_src = fh.read().decode("utf-8")
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
    if not mapping and old_src != new_src:
        raise Unswappable("source differs but no code object changed")
    n = 0
    for f in gc.get_objects():
        if type(f) is types.FunctionType:
            nc = mapping.get(f.__code__)
            if nc is not None:
                f.__code__ = nc
                n += 1
    return n


# ---------------------------------------------------------------- source overlays
def install_overlays(files):
    """Tests that read sqlglot sources from disk (e.g. test_typing reads
    sqlglot/typing/*.py) or tracebacks (linecache) must see the patched text."""
    import builtins
    import io
    import linecache

    patched = {os.path.realpath(os.path.join(SRC, rel)): src for rel, src in files.items()}
    for path, src in patched.items():
        linecache.cache[path] = (len(src), None, src.splitlines(True), path)
    orig_open = io.open

    def overlay_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
            try:
                p = os.path.realpath(os.fsdecode(file))
            except Exception:
                p = None
            src = patched.get(p)
            if src is not None:
                if "b" in mode:
                    s = io.BytesIO(src.encode("utf-8"))
                else:
                    s = io.StringIO(src)
                s.name = p
                return s
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = overlay_open
    io.open = overlay_open


# ---------------------------------------------------------------- re-importing
_KEEP_ALIVE = []  # old modules stay referenced: no mass deallocation in the child


class _PatchedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, src):
        super().__init__(fullname, path)
        self._src = src

    def get_code(self, fullname):
        return compile(self._src, self.path, "exec", dont_inherit=True)


class _PatchedFinder:
    def __init__(self, files):
        self.files = {os.path.realpath(os.path.join(SRC, rel)): src for rel, src in files.items()}

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] not in ("sqlglot", "tests"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.origin:
            src = self.files.get(os.path.realpath(spec.origin))
            if src is not None:
                spec.loader = _PatchedLoader(name, spec.origin, src)
        return spec


def reimport(files):
    """Full-fidelity mode for edits a code swap cannot reproduce (class attributes,
    defaults, decorators, module globals): forget sqlglot and the test modules and
    import them again (lazily, as a test run would) from the patched sources."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("sqlglot", "tests"):
            _KEEP_ALIVE.append(sys.modules.pop(name))
    importlib.invalidate_caches()
    sys.meta_path.insert(0, _PatchedFinder(files))


# ---------------------------------------------------------------- running tests
def _uid(pytest_id):
    path, *rest = pytest_id.split("::")
    return path[:-3].replace("/", "."), rest


class _SubTestFailFast(unittest.TestResult):
    """failfast=True makes subTest() end a test at its first failing subtest (the
    verdict is settled); stop() is a no-op so the rest of the suite still runs."""

    def __init__(self):
        super().__init__()
        self.failfast = True

    def stop(self):
        pass


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
    result = _SubTestFailFast()
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


def run_failfast(ids):
    """Run the tests one by one in the given order and stop at the first failure.
    Returns (passed, failed); tests after the failure are in neither."""
    passed = []
    for tid in ids:
        if run_tests([tid]):
            return passed, [tid]
        passed.append(tid)
    return passed, []


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
            if req.get("mode") == "reimport":
                reimport(req["files"])
            else:
                for rel, src in req["files"].items():
                    swap(rel, src)
            # after the swap: swap() must read the pristine file, tests the patched one
            install_overlays(req["files"])
            if req.get("failfast"):
                passed, failed = run_failfast(req["tests"])
                out = {"failed": failed, "passed": passed, "ok": True}
            else:
                failed = run_tests(req["tests"])
                out = {"failed": failed, "passed": [t for t in req["tests"] if t not in failed], "ok": True}
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
    # duckdb (0.1 s) is imported by test_optimizer / test_executor at module level:
    # load it once here; pandas (1.7 s) stays lazy (cost model: vcost.json.gz)
    for heavy in ("duckdb", "tests", "tests.helpers"):
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
