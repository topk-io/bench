import os

import psycopg
from psycopg_pool import ConnectionPool

from ..topk_bench import Document, Provider

# TopK's SQL API speaks the PostgreSQL wire protocol, so any pg client works.
#   host=<region>.sql.<host> port=5432 user=topk password=<api-key> dbname=topk
#
# Notes that shape this implementation (see docs/sdk/topk-sql/overview.mdx):
#   - `SELECT *` is not supported; columns must be listed explicitly.
#   - Only a single ORDER BY expression is allowed.
#   - INSERT is an upsert: it replaces the row when `_id` already exists.
#   - Vector literals need a TopK cast (`::f32_vector`), not a pg one.
#   - There are no transactions; BEGIN/COMMIT are accepted but writes are not
#     transactional, so we run autocommit.

DIM = 768

# Rows per INSERT statement. 2000 = the harness batch size; measured working at
# ~30 MB/statement against the shim. See upsert() for why this is not executemany.
_MAX_INSERT_ROWS = 2000


class TopKSQLProvider(Provider):
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        dbname: str = "topk",
        pool_size: int = 16,
    ):
        region = os.environ["TOPK_REGION"]
        base = os.environ.get("TOPK_HOST", "topk.io")

        conninfo = " ".join(
            [
                f"host={host or os.environ.get('TOPK_SQL_HOST') or f'{region}.sql.{base}'}",
                f"port={port or int(os.environ.get('TOPK_SQL_PORT', 5432))}",
                "user=topk",
                f"password={api_key or os.environ['TOPK_API_KEY']}",
                f"dbname={dbname}",
            ]
        )

        # The Rust harness drives providers from several blocking threads at
        # once (8 writers during ingest). A single psycopg connection is not
        # safe to share across threads, and sharing one would silently
        # serialize every request — so pool one connection per caller.
        # prepare_threshold=None disables psycopg's automatic prepared
        # statements. Without it, a connection that has prepared anything is
        # reset with `DEALLOCATE ALL` on putback, which the shim rejects:
        #   FeatureNotSupported: Unsupported: statement: Deallocate { .. "ALL" .. }
        # Real PostgreSQL supports DEALLOCATE ALL. Filed 2026-07-31; see
        # stash/backend-errors.md. Costs nothing here -- the upsert path sends
        # one multi-row statement per chunk and the query path is single-shot,
        # so there is no statement worth caching anyway.
        self.pool = ConnectionPool(
            conninfo,
            min_size=min(4, pool_size),
            max_size=pool_size,
            kwargs={"autocommit": True, "prepare_threshold": None},
            open=True,
        )

    def name(self) -> str:
        return "topk-sql"

    def setup(self, collection: str):
        with self.pool.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_ident(collection)} (
                    text            TEXT           NOT NULL,
                    dense_embedding f32_vector({DIM})      INDEX vector_index(metric = 'cosine'),
                    int_filter      INTEGER        NOT NULL,
                    keyword_filter  TEXT           NOT NULL INDEX keyword_index()
                )
                """
            )

    def query_by_id(self, collection: str, id: str):
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT _id, text, int_filter, keyword_filter "
                f"FROM {_ident(collection)} WHERE _id = %s LIMIT 1",
                (id,),
            ).fetchall()
        return [to_document(r) for r in rows]

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None,
        keyword_filter: str | None,
    ) -> list[Document]:
        where, params = [], [_vector_literal(vector)]

        if int_filter is not None:
            where.append("int_filter <= %s")
            params.append(int_filter)

        if keyword_filter is not None:
            # match_all() requires every token to be present, matching the
            # native provider's field("keyword_filter").match_all(...).
            where.append("match_all(keyword_filter, %s)")
            params.append(keyword_filter)

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(top_k)

        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT _id, text, int_filter, keyword_filter,
                       vector_distance(dense_embedding, %s::f32_vector) AS vector_distance
                FROM {_ident(collection)}
                {clause}
                ORDER BY vector_distance {_ORDER}
                LIMIT %s
                """,
                params,
            ).fetchall()

        return [to_document(r) for r in rows]

    def upsert(self, collection: str, docs: list[Document]):
        # INSERT replaces on _id, so this is an upsert.
        #
        # ONE multi-row VALUES statement per chunk -- do NOT "optimise" this back
        # to executemany(). Measured against the shim on 2026-07-31, 100 docs:
        #
        #   executemany()        4.0 docs/s      <- was the implementation here
        #   explicit pipeline    4.0 docs/s
        #   multi-row VALUES   236.9 docs/s      <- 59x, and it keeps scaling:
        #                                           2639 docs/s at 2000 rows
        #   COPY                 unsupported by the shim (parse error)
        #
        # The old comment here reasoned that a 2000-doc multi-row statement would
        # be ~6 MB and therefore bad. It is actually ~30 MB -- and still wins by
        # a factor of 660, because the alternative is 2000 sequential round-trips
        # at ~250 ms each. psycopg pipelines executemany only when the server
        # supports it; this shim does not, so every row costs a full round-trip.
        # Verified with an explicit conn.pipeline() block: no improvement.
        #
        # Chunked at 2000 rows because that is the harness batch size and the
        # largest width measured to work. topk-es has an analogous limit
        # (MAX_BULK_BYTES) where the shim rejects oversized bodies outright.
        for i in range(0, len(docs), _MAX_INSERT_ROWS):
            chunk = docs[i:i + _MAX_INSERT_ROWS]
            values = ",".join(["(%s, %s, %s::f32_vector, %s, %s)"] * len(chunk))
            params = []
            for doc in chunk:
                params += [
                    doc.id,
                    doc.text,
                    _vector_literal(doc.dense_embedding or []),
                    doc.int_filter,
                    doc.keyword_filter,
                ]
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {_ident(collection)} "
                        f"(_id, text, dense_embedding, int_filter, keyword_filter) "
                        f"VALUES {values}",
                        params,
                    )

    def delete_by_id(self, collection: str, ids: list[str]):
        with self.pool.connection() as conn:
            conn.execute(
                f"DELETE FROM {_ident(collection)} WHERE _id = ANY(%s)", (list(ids),)
            )

    def delete_collection(self, collection: str):
        with self.pool.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_ident(collection)}")

    def list_collections(self):
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
            ).fetchall()
        return [r[0] for r in rows]

    def close(self):
        self.pool.close()


# Ranking direction. Under cosine, TopK's `vector_distance` is a similarity —
# higher is better — so results are ordered DESC:
#   - topk_sdk's Query.topk(expr, k, asc=False) defaults to descending, and the
#     native provider calls .topk(field("vector_distance"), k) without `asc`.
#   - The SQL docs example likewise uses `ORDER BY vec_dist DESC`.
# Getting this backwards does not error: it silently returns the *worst*
# matches, i.e. near-zero recall while latency looks excellent. Any change here
# must be checked against the native provider's recall, not just for a clean run.
_ORDER = "DESC"


def _ident(name: str) -> str:
    """Quote an identifier; collection names contain hyphens (e.g. x-100k)."""
    return '"' + name.replace('"', '""') + '"'


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def to_document(row) -> Document:
    return Document(
        id=str(row[0]),
        text=row[1],
        int_filter=row[2],
        keyword_filter=row[3],
    )
