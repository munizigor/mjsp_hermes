import os
from re import T
import sys
import json
from time import sleep, time
import signal
import sqlite3 as sqlite
import threading

from interpreters import EmergencyInterpreter
from agent import InterpretationAgent
from start import start_and_listen

os.environ["SQLITE_DB_PATH"] = '/test.db'

def populate_db(sqlite_path):
    sqlite_conn = sqlite.connect(sqlite_path)
    contexts_list_path = '/datasets/emergencias-exemplos1.txt'
    transcription_strs = open(contexts_list_path, 'r').readlines()
    transcription_strs = [x for x in transcription_strs if len(x.rstrip('\n')) > 0]
    emergency_ids = []
    for _ in range(len(transcription_strs)):
        cursor = sqlite_conn.cursor()
        cursor.execute("INSERT INTO emergencies (operator_unique_code) VALUES (?)", ('pita_test',))
        emergency_ids.append(cursor.lastrowid)
    sqlite_conn.commit()
    cursor.close()
    
    transcriptions_to_insert = [
        (em_id, t.rstrip('\n').strip(), time()-0.6, int(time()), 'direct_text')
        for em_id, t in zip(emergency_ids, transcription_strs)
    ]
    cursor = sqlite_conn.cursor()
    cursor.executemany('''
        INSERT INTO emergency_transcripts 
        (id_emergencia, part, start_time, horario_de_transcricao, transcription_model) 
        VALUES (?, ?, ?, ?, ?)''', 
        transcriptions_to_insert)
    sqlite_conn.commit()
    cursor.close()

    #View current state of the database
    cursor = sqlite_conn.cursor()
    cursor.execute('''SELECT * FROM emergencies''')
    print('emergencies:')
    for em in cursor.fetchall():
        print(em)
    cursor.execute('''SELECT * FROM emergency_transcripts''')
    print('emergency_transcripts:')
    for em_trans in cursor.fetchall():
        print(em_trans)

    sqlite_conn.close()


def list_inference_results(sqlite_path):
    sqlite_conn = sqlite.connect(sqlite_path)
    cursor = sqlite_conn.cursor()
    cursor.execute('''SELECT * FROM resultados_inferencia''')
    results = cursor.fetchall()
    sqlite_conn.close()
    
    
    if results != None:
        if len(results) > 0:
            print('results_inferencia:')
            for result in results:
                print(result)
        else:
            print('No results found yet')
    else:
        print('No results found yet')

if __name__ == "__main__":
    config_json = json.load(open('/config.json', 'r'))
    print('Config loaded:', config_json)
    interpretation_config = config_json.get('interpretation', {})
    hardware_config = interpretation_config.get('hardware_config', 'cpu')
    delay = interpretation_config.get('delay', 5)
    n_in_batch = interpretation_config.get('n_in_batch', 10)
    print('Interpretation:', hardware_config, delay, n_in_batch)
    
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    cursor.executescript(open('/init.sql', 'r').read())
    sqlite_conn.commit()

    populate_db(os.environ["SQLITE_DB_PATH"])

    # Start monitoring loop in a separate thread
    def monitoring_loop():
        while True:
            sleep(10)
            list_inference_results(os.environ["SQLITE_DB_PATH"])
    
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    print("Started monitoring loop in background thread")

    # Run start_and_listen in the main thread
    start_and_listen(hardware_config, delay=delay, n_in_batch=n_in_batch)