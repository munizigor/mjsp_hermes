/**
emergencies.current_status:
0 - Ainda não atendida
1 - Em atendimento
2 - Finalizada
**/

CREATE TABLE IF NOT EXISTS emergencies (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    start_time DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    end_time DATETIME(6) DEFAULT NULL,
    current_status INT NOT NULL DEFAULT 0,
    asterisk_id VARCHAR(128) DEFAULT NULL,
    source_phone_number VARCHAR(15) DEFAULT NULL,
    destination_phone_number VARCHAR(15) DEFAULT NULL,
    operator_unique_code VARCHAR(128) DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_emergencies_start_time (start_time),
    KEY idx_emergencies_end_time (end_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS call_logs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    start_time DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    id_emergencia BIGINT UNSIGNED DEFAULT NULL,
    source_phone_number VARCHAR(15) DEFAULT NULL,
    destination_phone_number VARCHAR(15) DEFAULT NULL,
    event_type VARCHAR(128) NOT NULL,
    event_data LONGTEXT DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_call_logs_start_time (start_time),
    KEY idx_call_logs_id_emergencia (id_emergencia),
    CONSTRAINT fk_call_logs_emergencia
        FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS emergency_transcripts (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    id_emergencia BIGINT UNSIGNED NOT NULL,
    audio_id VARCHAR(100) DEFAULT NULL,
    part TEXT NOT NULL,
    start_time DATETIME(6) DEFAULT NULL,
    horario_de_transcricao DOUBLE DEFAULT NULL,
    is_analyzed BOOLEAN NOT NULL DEFAULT FALSE,
    transcription_seconds DOUBLE NOT NULL DEFAULT 0.0,
    transcription_model VARCHAR(191) DEFAULT NULL,
    actor VARCHAR(128) DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_emergency_transcripts_id_emergencia (id_emergencia),
    KEY idx_emergency_transcripts_start_time (start_time),
    CONSTRAINT fk_emergency_transcripts_emergencia
        FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS emergency_audios (
    id VARCHAR(100) NOT NULL,
    id_emergencia BIGINT UNSIGNED NOT NULL,
    start_time DATETIME(6) DEFAULT NULL,
    transcription_status INT NOT NULL DEFAULT 0,
    sampling_rate INT NOT NULL,
    audio_length_seconds DOUBLE NOT NULL,
    audio_path TEXT DEFAULT NULL,
    channel VARCHAR(100) DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_emergency_audios_id_emergencia (id_emergencia),
    KEY idx_emergency_audios_transcription_status (transcription_status),
    KEY idx_emergency_audios_start_time (start_time),
    CONSTRAINT fk_emergency_audios_emergencia
        FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS resultados_inferencia (
    id VARCHAR(191) NOT NULL,
    id_emergencia BIGINT UNSIGNED NOT NULL,
    horario_contexto DOUBLE NOT NULL,
    horario_fim DOUBLE NOT NULL,
    tipo_de_inferencia VARCHAR(128) DEFAULT NULL,
    resultado LONGTEXT DEFAULT NULL,
    duracao_inferencia DOUBLE DEFAULT NULL,
    duracao_outros_processamentos DOUBLE DEFAULT NULL,
    input_tokens INT DEFAULT NULL,
    output_tokens INT DEFAULT NULL,
    modelo_utilizado VARCHAR(191) DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_resultados_inferencia_id_emergencia (id_emergencia),
    KEY idx_resultados_inferencia_tipo_de_inferencia (tipo_de_inferencia),
    CONSTRAINT fk_resultados_inferencia_emergencia
        FOREIGN KEY (id_emergencia) REFERENCES emergencies (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS metrics (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    readings_time DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    hostname VARCHAR(255) NOT NULL,
    server_type VARCHAR(128) NOT NULL,
    inference_type VARCHAR(128) NOT NULL,
    metrics_dict LONGTEXT NOT NULL,
    PRIMARY KEY (id),
    KEY idx_metrics_readings_time (readings_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;