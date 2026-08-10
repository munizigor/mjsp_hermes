import os
import sqlite3 as sqlite
import json

from fastapi.responses import JSONResponse
from fastapi import APIRouter

from util import summarize_context

router = APIRouter()


@router.get("/get_transcriptions/", response_class=JSONResponse)
def get_transcriptions(id_emergencia: int) -> dict:
    """
    Get the full transcription for a specific emergency.

    Args:
        id_emergencia (int): The ID of the emergency session.

    Returns:
        dict: A dictionary containing the full transcription data.

    Example response:
    {
        "transcripts": [
            {
                "actor": str,
                "text": str,
                "transcription_time": float,
                "audio_time": float,
            },
            ...
        ]
    }
    """
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()

    # Get all transcript parts for the emergency, ordered by start_time
    cursor.execute(
        """
        SELECT id, part, start_time, horario_de_transcricao, is_analyzed, 
                transcription_seconds, transcription_model
        FROM emergency_transcripts 
        WHERE id_emergencia = ? 
        ORDER BY start_time ASC
    """,
        (id_emergencia,),
    )

    transcripts = []
    lines = cursor.fetchall()
    if lines != None:
        for row in lines:
            transcripts.append(
                {
                    "id": row[0],
                    "part": row[1],
                    "start_time": row[2],
                    "horario_de_transcricao": row[3],
                    "is_analyzed": bool(row[4]),
                    "transcription_seconds": row[5],
                    "transcription_model": row[6],
                }
            )

    cursor.execute(
        """
        SELECT et.part, et.horario_de_transcricao, et.actor, et.start_time
        FROM emergency_transcripts et
        WHERE et.id_emergencia = ?
        ORDER BY et.start_time ASC
    """,
        (id_emergencia,),
    )
    transcript_parts = cursor.fetchall()
    sqlite_conn.close()
    full_transcription = []
    if len(transcript_parts) > 0:
        for cols in transcript_parts:
            text = cols[0]
            actor = cols[2]
            transcription_time = cols[1]
            audio_time = cols[3]
            full_transcription.append(
                {
                    "actor": actor,
                    "text": text,
                    "transcription_time": transcription_time,
                    "audio_time": audio_time,
                }
            )

    sqlite_conn.close()
    return {"transcripts": full_transcription}


@router.get("/get_all_inference_results/", response_class=JSONResponse)
def get_all_inference_results(id_emergencia: int) -> dict:
    """
    Get all inference results for a specific emergency.

    Args:
        id_emergencia (int): The ID of the emergency session.

    Returns:
        dict: A dictionary containing all inference results.

    Example response (formatted JSON):
    If no valid suggestion has been found, the keys may contain an empty list
    {
      "rua_ou_logradouro": [
        { "value": "BR-100", "points": 34.0105 }
      ],
      "bairro": [
        { "value": "Jardim Maravilha", "points": 44.9395 }
      ],
      "municipio": [
        { "value": "Rio de Janeiro", "points": 44.9395 }
      ],
      "ponto_de_referencia":
        [{"value":"Igreja São Miguel","points":30.4605},
        {"value":"Igreja Católica","points":27.034}],
      "nome_do_solicitante": [
        { "value": "Pedro Paulo", "points": 34.2685 }
      ],
      "pessoa": [
        { "value": "Vicente", "points": 31.8185 },
        { "value": "motoqueiro", "points": 31.3235 },
        { "value": "Bombeiro, soldado Vicente", "points": 25.9265 }
      ],
      "numero": [
        { "value": "677", "points": 33.3295 }
      ],
      "complemento": [
        { "value": "Apartamento 27", "points": 29.937 }
      ],
      "descricao_breve": [
        {
            "value": "Motoqueiro sofreu queda sozinha na BR-100, está consciente, respira e responde, porém com dor intensa; capacete foi removido na queda.",
            "points": 12.5
        }
      ],
      "outras_observacoes": [
        {
            "value": "Solicitante registra informações inconsistentes sobre quem chegou ao local (Bombeiro/PRF). Pessoa chamada [...]",
            "points": 12.5
        }
      ],
      "natureza_inicial": [
        { "value": "Acidente de Trânsito Com Vítima", "points": 37.5 },
        { "value": "Queda de Motocicleta", "points": 25.233131735414357 },
        { "value": "Morte Acidental Provocada Por Eletroplessão (choque Elétrico)", "points": 25.192473850780626 },
        { "value": "Acidente de Trânsito Com Vítima", "points": 25.176785965549797 }
      ],
      "transcricao": "Bombeiro, soldado Vicente É, boa noite Um motoqueiro caiu aqui no
        Jardim Maravilha, do lado do posto É, em Jardim Maravilha? Isso Ele tá registrando
        aqui, tá bom? Tá, jóia Ele tá respirando? Tá, até respirando Mas tá bem, complicada
        a situação dele aqui Viu, moço? [...]"
    }
    """

    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()

    """cols = ['id', 'horario_contexto', 'horario_fim', 'tipo_de_inferencia', 'resultado', 'duracao_inferencia',
        'duracao_outros_processamentos', 'input_tokens', 'output_tokens', 'modelo_utilizado']"""

    inference_rows = cursor.execute(
        "SELECT * FROM resultados_inferencia WHERE id_emergencia = ?", (id_emergencia,)
    ).fetchall()
    cursor.execute("SELECT * FROM resultados_inferencia LIMIT 1")
    inference_columns = [desc[0] for desc in cursor.description]
    inference_results = []
    for row in inference_rows:
        inference_results.append(
            {inference_columns[n]: row[n] for n in range(len(inference_columns))}
        )

    cursor.execute(
        """
        SELECT et.part, et.horario_de_transcricao, et.actor 
        FROM emergency_transcripts et
        WHERE et.id_emergencia = ?
        ORDER BY et.start_time ASC
    """,
        (id_emergencia,),
    )
    transcript_parts = cursor.fetchall()
    sqlite_conn.close()
    if transcript_parts == None:
        transcript_parts = []
    if len(transcript_parts) > 0:
        full_transcription = "\n".join(
            [
                part[0] if type(part[2]) != str else part[2] + ": " + part[0]
                for part in transcript_parts
            ]
        )
    else:
        full_transcription = ""

    sqlite_conn.close()

    # print('Resultados da emergencia', id_emergencia, file=sys.stderr)
    # print(len(inference_results), 'resultados de inferencia', file=sys.stderr)
    inference_sorted = summarize_context(inference_results)
    # print(inference_sorted, file=sys.stderr)
    inference_sorted["transcricao"] = full_transcription

    nat_decisivas = inference_sorted["classificacao_decisiva"]
    nat_prob = inference_sorted["classificacoes_provaveis"]

    if len(nat_decisivas) > 0:
        smallest_dec_point = min([nat["points"] for nat in nat_decisivas])
    else:
        smallest_dec_point = 100.0

    for nat in nat_prob:
        nat["points"] = (nat["points"] / 100) * smallest_dec_point
    inference_sorted["natureza_inicial"] = nat_decisivas + nat_prob
    inference_sorted["natureza_inicial"] = sorted(
        inference_sorted["natureza_inicial"],
        key=lambda nat: nat["points"],
        reverse=True,
    )
    del inference_sorted["classificacao_decisiva"]
    del inference_sorted["classificacoes_provaveis"]

    return inference_sorted


@router.get("/get_cluster_metrics/", response_class=JSONResponse)
def get_cluster_metrics() -> dict:
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT * FROM metrics")
    columns = [description[0] for description in cursor.description]
    metrics = cursor.fetchall()
    sqlite_conn.close()

    result = [{c: m[i] for i, c in enumerate(columns)} for m in metrics]
    for r in result:
        r["readings_time"] = float(r["readings_time"])
        r["metrics_dict"] = {
            k: float(v) for k, v in json.loads(r["metrics_dict"]).items()
        }
    most_recents = {}
    for r in result:
        if r["hostname"] not in most_recents:
            most_recents[r["hostname"]] = r
        elif r["readings_time"] > most_recents[r["hostname"]]["readings_time"]:
            most_recents[r["hostname"]] = r
    return {"history": result, "most_recents": most_recents}
