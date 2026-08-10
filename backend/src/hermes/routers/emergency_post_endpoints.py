import os

from fastapi.responses import JSONResponse
from fastapi import APIRouter

import sqlite3 as sqlite

router = APIRouter()


@router.post("/start_call/", response_class=JSONResponse)
def start_call(
    source_number: str, destination_number: str, call_timestamp: str
) -> dict:
    """
    Start a new call.

    Args:
        source_number (str): The source phone number.
        destination_number (str): The destination phone number.
        call_timestamp (str): The timestamp of the call.

    Returns:
        int: The ID of the newly created emergency session.

    EXAMPLE: curl -i -X POST "http://127.0.0.1:8001/start_call/?source_number=107&destination_number=108&call_timestamp=1768416225.58"
    """
    if source_number.isdigit() and destination_number.isdigit():
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """
            INSERT INTO emergencies (source_phone_number, destination_phone_number, asterisk_id) 
            VALUES (?, ?, ?)
        """,
            (source_number, destination_number, call_timestamp),
        )
        new_emergency_id = cursor.lastrowid
        sqlite_conn.commit()

        cursor.execute(
            """
            INSERT INTO call_logs (id_emergencia, event_type, source_phone_number, destination_phone_number) 
            VALUES (?, ?, ?, ?)
        """,
            (
                new_emergency_id,
                "START",
                source_number,
                destination_number,
            ),
        )
        sqlite_conn.commit()

        sqlite_conn.close()
        print("Nova emergencia entre numeros", source_number, destination_number)
        return {"new_emergency_id": new_emergency_id}
    else:
        return {
            "new_emergency_id": None,
            "error": "non-digits at source or destination number",
        }


@router.post("/end_call/")
def end_call(id_emergencia: int) -> str:
    """
    End the call interpretation session.

    Args:
        id_emergencia (int): The ID of the emergency session to end.

    Returns:
        str: A message indicating the session has ended.
    """
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    cursor.execute(
        """
        UPDATE emergencies 
        SET end_time = CURRENT_TIMESTAMP, current_status = 2
        WHERE id = ?
    """,
        (id_emergencia,),
    )
    sqlite_conn.commit()

    cols = ["id", "source_phone_number", "destination_phone_number"]
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
        emergencies = [
            {cols[n]: row[n] for n in range(len(cols))} for row in query_results
        ]

        emergencia = emergencies[0]
        cursor.execute(
            """
            INSERT INTO call_logs (id_emergencia, event_type, source_phone_number, destination_phone_number) VALUES (?, ?, ?, ?)
        """,
            (
                id_emergencia,
                "END",
                emergencia["source_phone_number"],
                emergencia["destination_phone_number"],
            ),
        )
        sqlite_conn.commit()

    sqlite_conn.close()

    return f"Call with ID {id_emergencia} has ended."


@router.post("/set_as_answered_by_number/")
def set_as_answered_by_number(source_number: str, destination_number: str) -> str:
    """
    Set call as answered, based on phone numbers.

    Args:
        source_number (str): The source phone number.
        destination_number (str): The destination phone number.

    Returns:
        str: A message indicating the session has been answered.
    """
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    cursor.execute(
        """
        UPDATE emergencies
        current_status = 1
        WHERE source_phone_number = ? AND destination_phone_number = ?
    """,
        (source_number, destination_number),
    )
    sqlite_conn.commit()

    cursor.execute(
        """
        INSERT INTO call_logs (event_type, source_phone_number, destination_phone_number) VALUES (?, ?, ?)
    """,
        (
            "ANSWERED",
            source_number,
            destination_number,
        ),
    )
    sqlite_conn.commit()

    sqlite_conn.close()

    print("Emergency ended", source_number, destination_number)

    return f"Call with source number {source_number} and destination number {destination_number} has ended."


@router.post("/end_call_by_number/")
def end_call_by_number(source_number: str, destination_number: str) -> str:
    """
    End the last open call interpretation session, based on phone numbers.

    Args:
        source_number (str): The source phone number.
        destination_number (str): The destination phone number.

    Returns:
        str: A message indicating the session has ended.
    """
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    cursor.execute(
        """
        UPDATE emergencies
        SET end_time = CURRENT_TIMESTAMP, current_status = 2
        WHERE end_time IS NULL 
            AND source_phone_number = ? 
            AND destination_phone_number = ?
    """,
        (source_number, destination_number),
    )
    sqlite_conn.commit()

    cursor.execute(
        """
        INSERT INTO call_logs (event_type, source_phone_number, destination_phone_number) VALUES (?, ?, ?)
    """,
        (
            "END",
            source_number,
            destination_number,
        ),
    )
    sqlite_conn.commit()

    sqlite_conn.close()

    print("Emergency ended", source_number, destination_number)

    return f"Call with source number {source_number} and destination number {destination_number} has ended."
