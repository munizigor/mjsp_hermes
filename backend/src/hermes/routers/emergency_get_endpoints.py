import os
import sqlite3 as sqlite
import json

from fastapi.responses import JSONResponse
from fastapi import APIRouter

router = APIRouter()


@router.get("/list_emergencies", response_class=JSONResponse)
def list_emergencies(is_open: bool = None) -> dict:
    """
    List all emergency sessions.

    Returns:
        dict: A dictionary containing a list of emergencies.
            {
                "open_emergencies": [
                    {
                        "id": int, "start_time": float, "end_time": None,
                        "current_status": int,
                        "asterisk_id": str, "operator_unique_code": str,
                        "source_phone_number": str, "destination_phone_number": str,
                    },
                    ...
                ]
            }
    """
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cols = [
        "id",
        "start_time",
        "end_time",
        "current_status",
        "asterisk_id",
        "operator_unique_code",
        "source_phone_number",
        "destination_phone_number",
    ]
    if is_open:
        filter_str = "WHERE end_time IS NULL"
    elif is_open == None:
        filter_str = ""
    elif is_open == False:
        filter_str = "WHERE end_time IS NOT NULL"
    cursor = sqlite_conn.cursor()
    query_results = cursor.execute(
        f"""
        SELECT {', '.join(cols)}
        FROM emergencies
        {filter_str}
    """
    ).fetchall()
    if query_results is not None:
        open_emergencies = [
            {cols[n]: row[n] for n in range(len(cols))} for row in query_results
        ]
    else:
        open_emergencies = []

    sqlite_conn.close()
    open_emergencies.sort(key=lambda e: e["start_time"], reverse=True)
    return {"emergencies": open_emergencies}


@router.get("/get_emergency/", response_class=JSONResponse)
def get_emergency(id_emergencia: int) -> dict:
    """
    List a emergency session.

    Returns:
        dict: A dictionary containing a emergency session.
            {
                "emergency":
                    {
                        "id": int, "start_time": float, "end_time": None,
                        "asterisk_id": str, "operator_unique_code": str,
                        "current_status": int,
                        "source_phone_number": 998310000,
                        "destination_phone_number": 849983158
                    },
            }
    """
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cols = [
        "id",
        "start_time",
        "end_time",
        "asterisk_id",
        "current_status",
        "operator_unique_code",
        "source_phone_number",
        "destination_phone_number",
    ]
    filter_str = f"WHERE id = {id_emergencia}"
    cursor = sqlite_conn.cursor()
    query_results = cursor.execute(
        f"""
        SELECT {', '.join(cols)}
        FROM emergencies
        {filter_str}
    """
    ).fetchall()
    if query_results is not None:
        open_emergencies = [
            {cols[n]: row[n] for n in range(len(cols))} for row in query_results
        ]
        emergency = open_emergencies[0]
        sqlite_conn.close()
        print("Emergency found:", emergency)
        return {"emergency": emergency}
    else:
        sqlite_conn.close()
        print("Emergency not found:", id_emergencia)
        return {"emergency": None}


@router.get("/start_call_interpretation/", response_class=JSONResponse)
def start_call_interpretation(operator_unique_code: str) -> dict:
    """
    Start a new call interpretation session.

    Args:
        operator_unique_code (str): Unique code for the operator handling the call.

    Returns:
        int: The ID of the newly created emergency session.
    """
    try:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """
            INSERT INTO emergencies (operator_unique_code) VALUES (?)
        """,
            (operator_unique_code,),
        )
        new_emergency_id = cursor.lastrowid
        sqlite_conn.commit()

        cursor.execute(
            """
            INSERT INTO call_logs (id_emergencia, event_type, destination_phone_number) VALUES (?,?,?)
        """,
            (
                new_emergency_id,
                "START",
                operator_unique_code,
            ),
        )
        sqlite_conn.commit()

        sqlite_conn.close()
        return {"new_emergency_id": new_emergency_id}
    except Exception as e:
        print("Error starting call interpretation:", e)
        return {"new_emergency_id": None, "error": str(e)}


@router.get("/processing_finished/", response_class=JSONResponse)
def processing_finished(id_emergencia: int) -> dict:
    """
    Check if the processing of the emergency has finished.

    Args:
        id_emergencia (int): The ID of the emergency session.
    Returns:
        dict: A dictionary indicating whether processing is finished and additional details.
            {
                'terminado': bool,
                'descricao': str,
                'delay': int (optional),
                'ultima_transcricao': int (optional),
                'ultima_natureza_decisiva': int (optional)
            }
    """

    print("Status de", id_emergencia)
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()

    # Verifica se todos os áudios de uma emergência específica estão transcritos
    query = """SELECT 
            CASE 
                WHEN COUNT(*) = 0 THEN 'Todos os áudios transcritos'
                ELSE 'Áudios pendentes de transcrição'
            END AS status
        FROM 
            emergency_audios
        WHERE 
            id_emergencia = ? AND transcription_status != 2;"""
    status_audios = cursor.execute(query, (id_emergencia,)).fetchone()[0]
    print(status_audios)
    audios_transcritos = status_audios == "Todos os áudios transcritos"

    if not audios_transcritos:
        sqlite_conn.close()
        return {"terminado": False, "descricao": "Áudios pendentes de transcrição"}

    # Obtém o horario_de_transcricao mais recente para um id_emergencia específico
    query = """
    SELECT 
        MAX(horario_de_transcricao) AS horario_de_transcricao_mais_recente
    FROM 
        emergency_transcripts
    WHERE 
        id_emergencia = ?;    
    """
    transcricao_mais_recente = cursor.execute(query, (id_emergencia,)).fetchone()[0]
    print(transcricao_mais_recente)
    if transcricao_mais_recente == None:
        # Sem transcrições
        sqlite_conn.close()
        return {"terminado": False, "descricao": "Sem transcrições feitas"}
    else:
        transcricao_mais_recente = int(transcricao_mais_recente)

    # Obtém o horario_de_transcricao mais recente para um id_emergencia específico
    query = """
        SELECT 
            MAX(horario_contexto) AS ultimo_contexto_processado
        FROM 
            resultados_inferencia
        WHERE 
            id_emergencia = ?
            AND tipo_de_inferencia = 'natureza_decisiva';    
        """
    ultimo_contexto_processado = cursor.execute(query, (id_emergencia,)).fetchone()[0]
    print(ultimo_contexto_processado)
    if ultimo_contexto_processado == None:
        # Sem transcrições
        sqlite_conn.close()
        return {"terminado": False, "descricao": "Sem natureza_decisiva criada"}

    diff = transcricao_mais_recente - ultimo_contexto_processado
    sqlite_conn.close()
    if diff > 1:
        return {
            "terminado": False,
            "delay": diff,
            "descricao": "Transcrição mais recente que último contexto interpretado",
            "ultima_transcricao": transcricao_mais_recente,
            "ultima_natureza_decisiva": ultimo_contexto_processado,
        }
    else:
        return {
            "terminado": True,
            "delay": diff,
            "descricao": "Interpretação decisiva de transcrição mais recente",
            "ultima_transcricao": transcricao_mais_recente,
            "ultima_natureza_decisiva": ultimo_contexto_processado,
        }
