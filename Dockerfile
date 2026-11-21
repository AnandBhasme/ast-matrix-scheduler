FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# sqlglot sources at the pinned commit: the selector applies each patch to them
# and diffs the ASTs. Fetched from sqlglot's own repository at the pinned commit.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz

WORKDIR /app
RUN mkdir -p /app/src \
    && tar -xzf /tmp/sqlglot.tar.gz -C /app/src --strip-components=1 \
        sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa/sqlglot \
        sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa/tests \
    && rm /tmp/sqlglot.tar.gz

# map.json.gz: per-test coverage / fixture / class-attribute map built offline
# from the pinned commit with tools/ (see progress.md)
COPY map.json.gz selector.py /app/
RUN chmod -R a+rX /app && python -c "import ast,sys; ast.parse(open('/app/selector.py').read())"

ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-S", "/app/selector.py"]
