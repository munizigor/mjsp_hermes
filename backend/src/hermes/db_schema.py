def start_ner_database(sqlite_conn):
    
    """
    Initialize the SQLite database for storing NER labels.
    """
    cursor = sqlite_conn.cursor()
    #Database table to registry calls (not their transcripts/entities)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            operator_unique_code TEXT NOT NULL
        )
    ''')

    #Database table to registry transcript parts of emergency calls
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_emergencia INTEGER NOT NULL,
            audio_id TEXT DEFAULT NULL,
            part TEXT NOT NULL,
            start_time TIMESTAMP,
            horario_de_transcricao INTEGER,
            is_analyzed BOOLEAN DEFAULT FALSE,
            transcription_seconds REAL DEFAULT 0.0,
            transcription_model TEXT DEFAULT NULL,
            FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        )
    ''')

    #Database table to registry WAV audio segments of emergency calls
    #transcription_status: 0 -> waiting; 1 -> running; 2 -> done
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_audios (
            id TEXT PRIMARY KEY,
            id_emergencia INTEGER NOT NULL,
            start_time TIMESTAMP,
            transcription_status INTEGER DEFAULT 0,
            sampling_rate INTEGER NOT NULL,
            audio_path TEXT,
            FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        )
    ''')

    #horario_contexto: parte do áudio até a qual o contexto foi considerado
    #tipo_de_inferencia: ner, descricao_e_observacao, lista_de_naturezas, natureza_decisiva
    #resultado: JSON com interpretação/extração
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_inferencia (
            id TEXT PRIMARY KEY,
            id_emergencia INTEGER NOT NULL,
            horario_contexto INTEGER NOT NULL,
            tipo_de_inferencia TEXT,
            resultado TEXT,
            duracao_inferencia REAL,
            duracao_outros_processamentos REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            modelo_utilizado TEXT,
            FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        )
    ''')

    sqlite_conn.commit()
    sqlite_conn.close()