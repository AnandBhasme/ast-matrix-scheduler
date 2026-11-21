"""Differential replay of TPC-H / TPC-DS executor queries.

tests/test_executor.py::test_execute_tpc{h,ds} do, per query i:
    expr  = parse_one(sql)                       (parent process)
    table = execute(pickle(expr), schema, tables) (spawned worker; only if meta "execute")
    a     = duckdb(transpile(sql, write="duckdb")[0])   (only if table is not None)
    assert_frame_equal(a, DataFrame(table.rows, table.columns))
Everything sqlglot contributes to that outcome is (transpiled SQL, columns, rows, any
exception). If those are identical to the pinned commit's for every query whose
code a patch touches, the test's outcome cannot change.

execute() itself is staged: ensure_tables/ensure_schema -> optimize (+ the eagerly
evaluated debug `expression.sql(pretty=True)`) -> Plan -> PythonExecutor. Two
fingerprints per query:
  full : runs everything (columns, rows, transpiled SQL)
  opt  : parse + pickle + optimize (+ debug sql) + transpile, pickled ASTs compared.
If a patch's changed lines run, for a query, only in the parse / optimize /
transpile stages, then identical "opt" fingerprints imply Plan / PythonExecutor
receive identical input and run unmodified code: the result cannot change.

  python replay.py fingerprint <suite> <out.json>          (pinned commit, at build)
  python replay.py check <suite> <spec> <fingerprints>     (patched sqlglot on PYTHONPATH)
      spec = "i:full,j:opt,..."; exit 0 = identical, 1 = differs / error
The sqlglot that gets imported is whatever is first on sys.path; tests/ helpers and
fixtures always come from TESTS_ROOT (the pristine checkout).
"""
import csv
import gzip
import hashlib
import json
import os
import pickle
import sys
import time

TESTS_ROOT = os.environ.get("TESTS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def _helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("br_helpers", os.path.join(TESTS_ROOT, "tests", "helpers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SUITES = {
    "tpch": ("optimizer/tpc-h/tpc-h.sql", "optimizer/tpc-h/", "TPCH_SCHEMA"),
    "tpcds": ("optimizer/tpc-ds/tpc-ds.sql", "optimizer/tpc-ds/", "TPCDS_SCHEMA"),
}


def _noop(stage):
    pass


class Runner:
    def __init__(self, suite):
        self.h = _helpers()
        fixture, self.dir, schema_name = SUITES[suite]
        self.schema = getattr(self.h, schema_name)
        self.sqls = list(self.h.load_sql_fixture_pairs(fixture))
        self.tables = {}

    def table(self, name):
        t = self.tables.get(name)
        if t is None:
            t = self.tables[name] = self._load(name)
        return t

    def _load(self, table):
        # mirrors TestExecutor.setUpClass.setup (csv -> executor Table)
        import ast

        from sqlglot.executor.table import Table

        columns = self.schema[table]
        file_name = os.path.join(self.h.FIXTURES_DIR, self.dir, f"{table}.csv.gz")
        with open(file_name, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        fh = gzip.open(file_name, "rt", newline="") if gz else open(file_name, encoding="utf-8", newline="")
        with fh:
            reader = csv.reader(fh, delimiter="|")
            rows, ctypes = [], []
            next(reader)
            for row in reader:
                if not ctypes:
                    for v in row:
                        try:
                            ctypes.append(type(ast.literal_eval(v)))
                        except (ValueError, SyntaxError):
                            ctypes.append(str)
                rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
        return Table(columns=columns, rows=rows)

    def query(self, i, mode="full", hook=_noop):
        """-> fingerprint string for query i (0-based). hook(stage) marks stage starts."""
        from sqlglot import find_tables, parse_one, transpile
        from sqlglot.executor.python import PythonExecutor
        from sqlglot.executor.table import ensure_tables
        from sqlglot.optimizer import optimize
        from sqlglot.planner import Plan
        from sqlglot.schema import ensure_schema

        meta, sql, _ = self.sqls[i]
        h = hashlib.sha1(mode.encode())
        try:
            hook("parse")
            expr = parse_one(sql)
            expr = pickle.loads(pickle.dumps(expr))
            if not meta.get("execute"):
                h.update(b"noexec")
                return h.hexdigest()
            if mode == "opt":
                h.update(pickle.dumps(expr))
            # --- sqlglot.executor.execute(expr, schema=..., tables=...) inlined, staged
            tables_ = None
            if mode == "full":
                names = [t.name for t in find_tables(expr)]
                hook("pre")
                tables = {n: self.table(n) for n in names}
                tables_ = ensure_tables(tables, dialect=None)
            hook("opt")
            schema = ensure_schema(self.schema, dialect=None)  # only feeds optimize()
            if tables_ is not None and tables_.supported_table_args and (
                tables_.supported_table_args != schema.supported_table_args
            ):
                raise RuntimeError("Tables must support the same table args as schema")
            optimized = optimize(expr, schema, leave_tables_isolated=True, dialect=None)
            dbg = optimized.sql(pretty=True)
            if mode == "opt":
                h.update(pickle.dumps(optimized))
                h.update(dbg.encode())
                hook("transpile")
                h.update(transpile(sql, write="duckdb")[0].encode())
                return h.hexdigest()
            hook("plan")
            plan = Plan(optimized)
            hook("exec")
            res = PythonExecutor(tables=tables_).execute(plan)
            res = pickle.loads(pickle.dumps(res))
            if res is None:
                h.update(b"none")
                return h.hexdigest()
            h.update(repr(tuple(res.columns)).encode())
            h.update(repr(res.rows).encode())
            hook("transpile")
            h.update(transpile(sql, write="duckdb")[0].encode())
        except BaseException as e:  # noqa: BLE001
            h.update(f"EXC {type(e).__name__}: {e}".encode())
        return h.hexdigest()


def main():
    cmd = sys.argv[1]
    if cmd == "fingerprint":
        suite, out = sys.argv[2], sys.argv[3]
        r = Runner(suite)
        res = {}
        for mode in ("full", "opt"):
            fps, secs = [], []
            for i in range(len(r.sqls)):
                t = time.process_time()
                fps.append(r.query(i, mode))
                secs.append(round(time.process_time() - t, 4))
            res[mode] = {"fp": fps, "secs": secs}
            print(f"{suite}/{mode}: {len(fps)} queries, {sum(secs):.2f}s", file=sys.stderr)
        with open(out, "w") as f:
            json.dump(res, f)
    elif cmd == "check":
        # check <suite> <spec> <fps> [max_retry] [retry_cpu_cap]
        # Phase 1: every query in its mode; "opt" mismatches are collected (stop as
        # soon as more than max_retry differ). Phase 2: re-run those in full mode
        # (executed results compared) unless that would exceed retry_cpu_cap.
        # exit 0 identical, 1 differs, 2 gave up (cap)
        suite, spec, fpfile = sys.argv[2], sys.argv[3], sys.argv[4]
        max_retry = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        cap = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        ref = json.load(open(fpfile))
        r = Runner(suite)
        retry = []
        for item in (x for x in spec.split(",") if x):
            i, mode = item.split(":")
            i = int(i)
            if r.query(i, mode) != ref[mode]["fp"][i]:
                if mode == "full" or len(retry) >= max_retry:
                    print(f"replay {suite}: query {i} ({mode}) differs", file=sys.stderr)
                    sys.exit(1)
                retry.append(i)
        if retry:
            need = sum(ref["full"]["secs"][q] for q in retry) + 2.0  # + table loading
            if time.process_time() + need > cap:
                sys.exit(2)
            for i in sorted(retry, key=lambda q: ref["full"]["secs"][q]):
                if r.query(i, "full") != ref["full"]["fp"][i]:
                    print(f"replay {suite}: query {i} (full, retry) differs", file=sys.stderr)
                    sys.exit(1)
        print(f"replay {suite}: identical ({len(retry)} retried)", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
