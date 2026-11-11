import sqlite3
import sys
import json

tables = set(["emergency_transcripts", "resultados_inferencia"])
if len(sys.argv) > 1:
    tables_new = set(sys.argv[1].split(','))
    tables.update(tables_new)

for table in tables:
    print('\n###################################')
    print(f'#### {table.upper()} ####')
    conn = sqlite3.connect('sqlite.db/database.sqlite')
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table}')
    rows = cursor.fetchall()
    print(f"Últimas inferências:\n")
    for row in rows:
        if table == 'resultados_inferencia':
            name = row[4]
            content = json.dumps(json.loads(row[5]), ensure_ascii=False)
            print("\ntipo:", name)
            print(content)
        else:
            line = row[3].replace('\n', '; ')
            print(line+'\n')
    conn.close()