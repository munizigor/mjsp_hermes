import os
from time import time
from uuid import uuid1

from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi import APIRouter

import sqlite3 as sqlite
import soundfile as sf

router = APIRouter()


# Create endpoint that receives call transcript and return entities extracted from it.
@router.post("/send_transcript/")
def store_transcript(id_emergencia: str, transcript: str) -> str:
    """
    Extract entities from the given transcript.

    Args:
        id_emergencia (str): ID of an existing emergency.
        transcript (str): The call transcript from which to extract entities.

    Returns:
        dict: The ID for the stored transcript
    """

    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    insert_str = f"""
        INSERT INTO emergency_transcripts (id_emergencia, part, start_time, horario_de_transcricao, transcription_model, actor) 
            VALUES ({id_emergencia},'{transcript}',{time()}, {time()}, 'direct_text', 'transcricao')
    """
    print(insert_str)
    cursor.execute(insert_str)
    new_transcript_id = cursor.lastrowid
    sqlite_conn.commit()
    sqlite_conn.close()

    return str(new_transcript_id)


@router.post("/send_audio/")
async def receive_audio(
    id_emergencia: str, sp_rate: int, audio_data: UploadFile
) -> str:
    """
    Extract entities from the given transcript.
    Args:
        id_emergencia (str): ID of an existing emergency.
        sp_rate (int): Desired sampling rate for the audio.
        audio_data (UploadFile): The audio file to be processed.
    Returns:
        The ID for the stored audio
    """
    audio_id = str(uuid1())
    audio_path = os.path.join(
        os.path.dirname(os.environ["SQLITE_DB_PATH"]), id_emergencia, audio_id + ".wav"
    )
    print("Saving", audio_data.filename, "audio to", audio_path)
    if not os.path.exists(os.path.dirname(audio_path)):
        os.mkdir(os.path.dirname(audio_path))
    with open(audio_path, "wb") as tosave:
        # audio_bytes = audio_data.file.read()
        tosave.write(audio_data.file.read())
    print("Saved", audio_path)

    print("Loading", audio_path)
    y, sr = sf.read(audio_path)
    audio_duration = len(y) / sr

    print("Loaded", audio_path)
    print("Original audio shape:", y.shape, "Sample rate:", sr, "Data type:", y.dtype)

    # If stereo, convert to mono first
    if len(y.shape) > 1 and y.shape[1] > 1:
        print("Converting stereo to mono")
        y = y.mean(axis=1)  # Average the channels
        print("After mono conversion:", y.shape)

    if sr > sp_rate:
        print("Sample rate is", sr, "need to resample to", sp_rate)

        # Calculate expected number of samples after resampling
        expected_samples = int(len(y) * sp_rate / sr)
        print("Expected samples after resampling:", expected_samples)

        y_16k = librosa.resample(y, orig_sr=sr, target_sr=sp_rate, res_type="soxr_lq")
        print("Resampled", audio_path)
        print("Old shape:", y.shape, "Old duration:", len(y) / sr, "seconds")
        print(
            "New shape:", y_16k.shape, "New duration:", len(y_16k) / sp_rate, "seconds"
        )

        # Verify resampling worked correctly
        expected_duration = len(y) / sr
        audio_duration = len(y_16k) / sp_rate
        print(
            f"Duration verification: Expected {expected_duration:.2f}s, Got {audio_duration:.2f}s"
        )

        if abs(expected_duration - audio_duration) > 0.1:  # Allow 0.1s tolerance
            print(
                "WARNING: Duration mismatch! Resampling may not have worked correctly."
            )
        else:
            print("Resampling successful - durations match!")

        # Save original before overwriting
        # open(audio_path.replace('.wav', '_original.wav'), 'wb').write(open(audio_path, 'rb').read())
        sf.write(audio_path, y_16k, sp_rate)
        print("Saved resampled audio to", audio_path)
    else:
        print("Sample rate is", sr, "no need to resample")
    """except Exception as err:
        print(err)
        print('Problem found while resampling', audio_path)
        os.remove(audio_path)
        raise err"""

    print("Saving in audios list")
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    insert_str = f"""
        INSERT INTO emergency_audios (id, id_emergencia, start_time, transcription_status, sampling_rate, audio_length_seconds, audio_path) 
            VALUES ('{audio_id}', {id_emergencia}, {time()}, {0}, {sp_rate}, {audio_duration}, '{audio_path}')
    """
    print(insert_str)
    cursor.execute(insert_str)
    new_audio_id = cursor.lastrowid
    sqlite_conn.commit()
    sqlite_conn.close()

    return str(new_audio_id)
