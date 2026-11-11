import os
import sqlite3 as sqlite

env_file_path = "/.env" if os.path.exists("/.env") else ".env"
if os.path.exists(env_file_path):
    with open(env_file_path, "r") as f:
        env_vals = {
            rawline.split("=", 1)[0]: rawline.split("=", 1)[1].rstrip("\r\n")
            for rawline in f.read().split("\n")
            if "=" in rawline and not rawline.lstrip().startswith("#")
        }
    for k, v in env_vals.items():
        os.environ[k] = v


def init_db():
    db_path = os.environ["SQLITE_DB_PATH"]
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        print(f"Database already exists at {db_path}, skipping initialization.")
        return

    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    print("Initializing DB at", db_path)
    sqlite_conn = None
    cursor = None
    try:
        sqlite_conn = sqlite.connect(db_path)
        cursor = sqlite_conn.cursor()
        with open("/init.sql", "r") as init_sql_file:
            cursor.executescript(init_sql_file.read())
        sqlite_conn.commit()
        print("DB Initialization success!")
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()


if __name__ == "__main__":
    init_db()

