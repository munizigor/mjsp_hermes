import sqlite3
import sys
import json
conn = sqlite3.connect('sqlite.db/database.sqlite')
cursor = conn.cursor()
cursor.execute(f'SELECT * FROM {sys.argv[1]}')
rows = cursor.fetchall()
#n = 7
#rows = rows[-n::]
print(f"Últimas inferências:\n")
for row in rows:
    if sys.argv[1] == 'resultados_inferencia':
        name = row[4]
        content = json.dumps(json.loads(row[5]), ensure_ascii=False)
        print("\ntipo:", name)
        print(content)
    else:
        line = row[3].replace('\n', '; ')
        print(line+'\n')
conn.close()