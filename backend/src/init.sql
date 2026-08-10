/**
emergencies.current_status:
0 - Ainda não atendida
1 - Em atendimento
2 - Finalizada
**/

CREATE TABLE IF NOT EXISTS emergencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    current_status INTEGER DEFAULT 0,
    asterisk_id TEXT,
    source_phone_number TEXT DEFAULT NULL, /**Novo Campo**/
    destination_phone_number TEXT DEFAULT NULL, /**Novo Campo**/
    operator_unique_code TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS call_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    id_emergencia INTEGER,
    source_phone_number TEXT DEFAULT NULL, /**Novo Campo**/
    destination_phone_number TEXT DEFAULT NULL, /**Novo Campo**/
    event_type TEXT NOT NULL,
    event_data TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS emergency_transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_emergencia INTEGER NOT NULL,
    audio_id TEXT DEFAULT NULL,
    part TEXT NOT NULL,
    start_time TIMESTAMP,
    horario_de_transcricao REAL,
    is_analyzed BOOLEAN DEFAULT FALSE,
    transcription_seconds REAL DEFAULT 0.0,
    transcription_model TEXT DEFAULT NULL,
    actor TEXT DEFAULT NULL, /**Novo Campo**/
    FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
);

CREATE TABLE IF NOT EXISTS emergency_audios (
    id TEXT PRIMARY KEY,
    id_emergencia INTEGER NOT NULL,
    start_time TIMESTAMP,
    transcription_status INTEGER DEFAULT 0,
    sampling_rate INTEGER NOT NULL,
    audio_length_seconds REAL NOT NULL,
    audio_path TEXT,
    channel TEXT DEFAULT NULL, /**Novo Campo**/
    FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
);


CREATE TABLE IF NOT EXISTS resultados_inferencia (
    id TEXT PRIMARY KEY,
    id_emergencia INTEGER NOT NULL,
    horario_contexto REAL NOT NULL,
    horario_fim REAL NOT NULL,
    tipo_de_inferencia TEXT,
    resultado TEXT,
    duracao_inferencia REAL,
    duracao_outros_processamentos REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    modelo_utilizado TEXT,
    FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    readings_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    hostname TEXT NOT NULL,
    server_type TEXT NOT NULL,
    inference_type TEXT NOT NULL,
    metrics_dict TEXT NOT NULL
);