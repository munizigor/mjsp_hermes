import os
import json

from fastapi.responses import JSONResponse
from fastapi import APIRouter

import sqlite3 as sqlite

router = APIRouter()

# Call starters

@router.post("/start_call/", response_class=JSONResponse)
def start_call(
    source_number: str, destination_number: str,
) -> dict:
    """
    Start a new call, not associated with an ID from a VOIP provider.

    Args:
        source_number (str): The source phone number.
        destination_number (str): The destination phone number.

    Returns:
        int: The ID of the newly created emergency session.

    EXAMPLE: curl -i -X POST "http://127.0.0.1:8001/start_call/?source_number=107&destination_number=108"
    """
    if source_number.isdigit() and destination_number.isdigit():
        sqlite_conn = None
        cursor = None
        try:
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            cursor = sqlite_conn.cursor()
            cursor.execute(
                """
                INSERT INTO emergencies (source_phone_number, destination_phone_number) 
                VALUES (?, ?)
            """,
                (source_number, destination_number),
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
        finally:
            if cursor is not None:
                cursor.close()
            if sqlite_conn is not None:
                sqlite_conn.close()

        print("Nova emergencia entre numeros", source_number, destination_number)
        return {"new_emergency_id": new_emergency_id}
    else:
        return {
            "new_emergency_id": None,
            "error": "non-digits at source or destination number",
        }

@router.post("/start_call_asterisk/", response_class=JSONResponse)
def start_call_asterisk(
    source_number: str, destination_number: str, asterisk_id: str
) -> dict:
    """
    Start a new call, associated with an ID from a VOIP provider.

    Args:
        source_number (str): The source phone number.
        destination_number (str): The destination phone number.
        asterisk_id (str): The LinkedID of the asterisk call.

    Returns:
        int: The ID of the newly created emergency session.

    EXAMPLE: curl -i -X POST "http://127.0.0.1:8001/start_call_asterisk/?source_number=107&destination_number=108&asterisk_id=1768416225.58"
    """
    if source_number.isdigit() and destination_number.isdigit():
        sqlite_conn = None
        cursor = None
        try:
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            cursor = sqlite_conn.cursor()
            cursor.execute(
                """
                INSERT INTO emergencies (source_phone_number, destination_phone_number, asterisk_id) 
                VALUES (?, ?, ?)
            """,
                (source_number, destination_number, asterisk_id),
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
        finally:
            if cursor is not None:
                cursor.close()
            if sqlite_conn is not None:
                sqlite_conn.close()

        print("Nova emergencia entre numeros", source_number, destination_number)
        return {"new_emergency_id": new_emergency_id}
    else:
        return {
            "new_emergency_id": None,
            "error": "non-digits at source or destination number",
        }

@router.post("/start_call_for_operator/", response_class=JSONResponse)
def start_call_for_operator(operator_unique_code: str) -> dict:
    """
    Start a new call session for an operator, without a specific source number. Used for front-ends
        that send operator audio only, without access to source number.

    Args:
        operator_unique_code (str): Unique code for the operator handling the call.

    Returns:
        int: The ID of the newly created emergency session.
    """
    sqlite_conn = None
    cursor = None
    try:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """
            INSERT INTO emergencies (destination_phone_number, operator_unique_code) VALUES (?,?)
        """,
            (operator_unique_code, operator_unique_code),
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
        return {"new_emergency_id": new_emergency_id}
    except Exception as e:
        print("Error starting call interpretation:", e)
        return {"new_emergency_id": None, "error": str(e)}
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

# Call status changers

@router.post("/set_as_answered/")
def set_as_answered_by_number(id_emergencia: int) -> str:
    """
    Set call as answered, based on id_emergencia. This is for the front-end
    to signal that someone has opened the interface for this emergency.

    Args:
        source_number (str): The source phone number.
        destination_number (str): The destination phone number.

    Returns:
        str: A message indicating the session has been answered.
    """
    sqlite_conn = None
    cursor = None
    try:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """
            UPDATE emergencies
            SET current_status = 1
            WHERE id_emergencia = ?
        """,
            (id_emergencia),
        )
        sqlite_conn.commit()

        cursor.execute(
            """
            INSERT INTO call_logs (id_emergencia, event_type) VALUES (?, ?)
        """,
            (
                id_emergencia,
                "ANSWERED",
            ),
        )
        sqlite_conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

    return f"Emergency with ID {id_emergencia} has been answered."

@router.post("/change_destination_number/")
def change_destination_number(asterisk_id: str, new_destination_number: str) -> str:
    """
    Change the destination phone number of a asterisk ID, to reflect new routing.
    Launchs a event "TRANSFER" for this session.

    Args:
        asterisk_id (str): The LinkedID of the asterisk call.
        new_destination_number (str): The new destination phone number.

    Returns:
        str: A message indicating the session has been updated.
    """
    sqlite_conn = None
    cursor = None
    try:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()

        # Get source_phone_number and destination_phone_number from open call for log latter
        cols = ["id", "source_phone_number", "destination_phone_number"]
        cols_str = ','.join(cols)
        query_results = cursor.execute(
            f"""
            SELECT {cols_str}
            FROM emergencies
            WHERE asterisk_id = ? AND end_time IS NULL
            """,
            (asterisk_id,),
        ).fetchall()
        if query_results:
            emergencies = [
                {cols[n]: row[n] for n in range(len(cols))} for row in query_results
            ]

            emergencia = emergencies[0]
            source_number = emergencia["source_phone_number"]
            old_destination_number = emergencia["destination_phone_number"]

            cursor.execute(
                """
                UPDATE emergencies
                SET destination_phone_number = ?
                WHERE asterisk_id = ? AND end_time IS NULL
            """,
                (new_destination_number, asterisk_id),
            )
            sqlite_conn.commit()

            data = json.dumps({
                "from": old_destination_number, "to": new_destination_number
            })

            cursor.execute(
                """
                INSERT INTO call_logs (id_emergencia, event_type, source_phone_number, destination_phone_number, event_data) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    emergencia["id"],
                    "TRANSFER",
                    source_number,
                    new_destination_number,
                    data,
                ),
            )
            sqlite_conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

    print("Destination number changed", asterisk_id, new_destination_number)

    return f"Destination number for call with asterisk ID {asterisk_id} has been changed to {new_destination_number}."

# Call enders

@router.post("/end_call/")
def end_call(id_emergencia: int) -> str:
    """
    End the call interpretation session.

    Args:
        id_emergencia (int): The ID of the emergency session to end.

    Returns:
        str: A message indicating the session has ended.
    """
    sqlite_conn = None
    cursor = None
    try:
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
        query_results = cursor.execute(
            f"""
            SELECT {', '.join(cols)}
            FROM emergencies
            {filter_str}
        """
        ).fetchall()
        if query_results:
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
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

    return f"Call with ID {id_emergencia} has ended."

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
    sqlite_conn = None
    cursor = None
    try:
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
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

    print("Emergency ended", source_number, destination_number)

    return f"Call with source number {source_number} and destination number {destination_number} has ended."

@router.post("/end_call_asterisk/")
def end_call_asterisk(asterisk_id: str) -> str:
    """
    End the last open call interpretation session, based on asterisk ID.

    Args:
        asterisk_id (str): The LinkedID of the asterisk call.

    Returns:
        str: A message indicating the session has ended.
    """
    sqlite_conn = None
    cursor = None
    try:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()

        #Get source_phone_number and destination_phone_number from open call for log latter
        cols = ["source_phone_number", "destination_phone_number"]
        query_results = cursor.execute(
            """
            SELECT {', '.join(cols)}
            FROM emergencies
            WHERE asterisk_id = ? AND end_time IS NULL
        """,
            (asterisk_id,),
        ).fetchall()
        if query_results:
            emergencies = [
                {cols[n]: row[n] for n in range(len(cols))} for row in query_results
            ]

            emergencia = emergencies[0]
            source_number = emergencia["source_phone_number"]
            destination_number = emergencia["destination_phone_number"]
        
            #Set as ended
            cursor.execute(
                """
                UPDATE emergencies
                SET end_time = CURRENT_TIMESTAMP, current_status = 2
                WHERE end_time IS NULL 
                    AND asterisk_id = ?
            """,
                (asterisk_id,),
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
        else:
            return f"No open emergency found with asterisk ID {asterisk_id}"
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

    print("Emergency ended", source_number, destination_number)

    return f"Call with source number {source_number} and destination number {destination_number} has ended."
