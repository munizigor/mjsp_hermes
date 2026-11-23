from contextlib import contextmanager
import os
import random
import sys
import time
from typing import Any, Iterator, Sequence

import pymysql
from pymysql.connections import Connection


def get_mariadb_config() -> dict[str, Any]:
    """Retorna os parâmetros de conexão do MariaDB a partir das variáveis de ambiente."""
    return {
        "host": os.environ.get("MARIADB_HOST", "mariadb"),
        "port": int(os.environ.get("MARIADB_PORT", "3306")),
        "user": os.environ.get("MARIADB_USER", "hermes_user"),
        "password": os.environ.get("MARIADB_PASSWORD", "hermes_pass"),
        "database": os.environ.get("MARIADB_DATABASE", "hermes_staging"),
        "charset": "utf8mb4",
        "connect_timeout": int(os.environ.get("MARIADB_CONNECT_TIMEOUT", "5")),
        "autocommit": False,
    }


def connect_mariadb_with_backoff(
    max_attempts: int = 25,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    jitter: float = 0.25,
    log_prefix: str = "DB",
) -> Connection:
    """Abre conexão com o MariaDB usando backoff exponencial com jitter e valida com SELECT 1."""
    cfg = get_mariadb_config()
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        conn: Connection | None = None
        try:
            conn = pymysql.connect(**cfg)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return conn
        except Exception as exc:
            last_exc = exc
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if attempt == max_attempts:
                break

            exp_delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_time = max(0.1, exp_delay + random.uniform(-jitter, jitter))
            print(
                f"[{log_prefix}] MariaDB indisponível (tentativa {attempt}/{max_attempts}): {exc}. "
                f"Nova tentativa em {sleep_time:.2f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_time)

    assert last_exc is not None
    raise last_exc


def ensure_connection_healthy(
    conn: Connection | None, log_prefix: str = "DB"
) -> Connection:
    """Garante uma conexão ativa; reconecta com backoff se estiver fechada ou caída."""
    if conn is None:
        return connect_mariadb_with_backoff(log_prefix=log_prefix)
    try:
        conn.ping(reconnect=True)
        return conn
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return connect_mariadb_with_backoff(log_prefix=log_prefix)


@contextmanager
def get_db_connection(log_prefix: str = "DB") -> Iterator[Connection]:
    """Context manager para conexões de curta duração (fecha automaticamente ao sair)."""
    conn = connect_mariadb_with_backoff(log_prefix=log_prefix)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def db_transaction(
    conn: Connection | None = None, log_prefix: str = "DB"
) -> Iterator[Connection]:
    """
    Context manager transacional: faz commit em sucesso e rollback em exceção.
    Se `conn` não for passada, abre e fecha uma conexão automaticamente.
    """
    owns_conn = conn is None
    active_conn = (
        ensure_connection_healthy(conn, log_prefix=log_prefix)
        if not owns_conn
        else connect_mariadb_with_backoff(log_prefix=log_prefix)
    )
    try:
        yield active_conn
        active_conn.commit()
    except Exception:
        try:
            active_conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if owns_conn:
            try:
                active_conn.close()
            except Exception:
                pass


def fetch_all(
    query: str, params: Sequence[Any] = (), log_prefix: str = "DB"
) -> list[tuple[Any, ...]]:
    """Executa consulta SELECT e retorna todas as linhas."""
    with get_db_connection(log_prefix=log_prefix) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return list(rows) if rows is not None else []


def fetch_one(
    query: str, params: Sequence[Any] = (), log_prefix: str = "DB"
) -> tuple[Any, ...] | None:
    """Executa consulta SELECT e retorna uma linha ou None."""
    with get_db_connection(log_prefix=log_prefix) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()


def execute_write(
    query: str, params: Sequence[Any] = (), log_prefix: str = "DB"
) -> int:
    """Executa INSERT/UPDATE/DELETE com commit automático e retorna lastrowid."""
    with db_transaction(log_prefix=log_prefix) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return int(cur.lastrowid or 0)


def execute_many_write(
    query: str, params_seq: Sequence[Sequence[Any]], log_prefix: str = "DB"
) -> int:
    """Executa escrita em lote com commit automático e retorna rowcount."""
    if not params_seq:
        return 0
    with db_transaction(log_prefix=log_prefix) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params_seq)
            return int(cur.rowcount or 0)