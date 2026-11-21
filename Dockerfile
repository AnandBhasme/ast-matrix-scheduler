FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# sqlglot sources at the pinned commit: the selector applies each patch to them
# and diffs the ASTs. Fetched from sqlglot's own repository at the pinned commit.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz

# Test dependencies (exact versions of the labelling environment) for the in-process
# verifier, which runs a few selected tests against the patched code.
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

WORKDIR /app
RUN mkdir -p /app/src \
    && tar -xzf /tmp/sqlglot.tar.gz -C /app/src --strip-components=1 \
        sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa/sqlglot \
        sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa/tests \
    && rm /tmp/sqlglot.tar.gz

# map.json.gz: per-test coverage / fixture / class-attribute map built offline
# from the pinned commit. replay_map.json.gz: per-query coverage of replay.py.
# wm.json.gz: weak-mutation comparison-swap infection map.
COPY map.json.gz replay_map.json.gz wm.json.gz selector.py replay.py impdiff.py verifier.py /app/

# Pinned-commit fingerprints: every TPC-H / TPC-DS executor query (differential replay)
# and every class attribute / module global after import (differential import).
# Bytecode for the pinned sources: imports in the sandbox (patched copies in /tmp keep
# mtimes, so only patched files recompile) take ~0.5 s instead of ~2.5 s.
RUN python -m compileall -q /app/src/sqlglot /app/src/tests /app/impdiff.py /app/replay.py /app/verifier.py
RUN cd /app && export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 TESTS_ROOT=/app/src PYTHONPATH=/app/src \
    && python replay.py fingerprint tpch /app/fp_tpch.json \
    && python replay.py fingerprint tpcds /app/fp_tpcds.json \
    && python impdiff.py dump /app/imp_pinned.json \
    && python impdiff.py graph /app/src /app/imports.json \
    && chmod -R a+rX /app

ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-S", "/app/selector.py"]
