import tempfile
import io
import os
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence


def array_to_audiosegment(audio_array, sr):
    """Converte um numpy array para um objeto AudioSegment do pydub em memória."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return AudioSegment.from_wav(buffer)


def get_atomic_chunks(audio: AudioSegment, max_len_ms=29900):
    """
    Fase 1: Quebra o áudio no maior número possível de pedaços pequenos (frases)
    baseados em silêncio. Se algum pedaço for teimoso e maior que o limite,
    quebra forçadamente.
    """
    min_silence = 500
    thresh_offset = -14

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence,
        silence_thresh=audio.dBFS + thresh_offset,
        keep_silence=True,
    )

    # Fallback 1: Tenta achar silêncios mais sutis se a quebra falhou num áudio longo
    if len(chunks) <= 1 and len(audio) > max_len_ms:
        chunks = split_on_silence(
            audio,
            min_silence_len=200,
            silence_thresh=audio.dBFS - 10,
            keep_silence=True,
        )

    # Fallback 2: Hard cut (corte no meio) se não houver NENHUM silêncio (raro)
    if len(chunks) <= 1:
        if len(audio) > max_len_ms:
            print(
                f"[WARN] Áudio denso de {len(audio)}ms sem silêncios. Forçando corte."
            )
            half = len(audio) // 2
            return get_atomic_chunks(audio[:half], max_len_ms) + get_atomic_chunks(
                audio[half:], max_len_ms
            )
        else:
            return [audio]

    # Garante que todos os pedaços atômicos são estritamente <= max_len_ms
    final_atomic = []
    for c in chunks:
        if len(c) > max_len_ms:
            final_atomic.extend(get_atomic_chunks(c, max_len_ms))
        else:
            final_atomic.append(c)

    return final_atomic


def pack_audio_chunks(atomic_chunks, max_len_ms=29900, min_len_ms=2000):
    """
    Fase 2: Empacota os pedaços atômicos de volta em buffers gigantes que cheguem
    o mais perto possível do max_len_ms para economizar requisições à API.
    Lida com sobras muito curtas (< min_len_ms) pegando emprestado do chunk anterior.
    """
    if not atomic_chunks:
        return [], []

    bins = []
    current_bin = atomic_chunks[0]

    for chunk in atomic_chunks[1:]:
        # Se cabe no pacote atual, adiciona!
        if len(current_bin) + len(chunk) <= max_len_ms:
            current_bin += chunk
        else:
            # Não cabe. Fecha o pacote atual, guarda na lista, e começa um novo
            bins.append(current_bin)
            current_bin = chunk

    # Tratamento do último pacote (para evitar áudios de 1 segundo sem contexto)
    if len(current_bin) > 0:
        if len(current_bin) < min_len_ms and len(bins) > 0:
            prev_bin = bins.pop()

            # Se por acaso o penúltimo pacote tinha muito espaço, só junta
            if len(prev_bin) + len(current_bin) <= max_len_ms:
                bins.append(prev_bin + current_bin)
            else:
                # Rouba o final do prev_bin para engordar o current_bin até o min_len_ms
                # Ex: current_bin tem 1s, min_len_ms é 2s. Roubamos 1s do pacote anterior.
                needed = min_len_ms - len(current_bin)

                # Segurança para não bugar se o prev_bin também for minúsculo
                if needed < len(prev_bin):
                    cut_point = len(prev_bin) - needed
                    new_prev = prev_bin[:cut_point]
                    new_curr = prev_bin[cut_point:] + current_bin
                    bins.append(new_prev)
                    bins.append(new_curr)
                else:
                    bins.append(prev_bin + current_bin)
        else:
            bins.append(current_bin)
    lens_ms = [len(b) for b in bins]
    lens_sec = [l / 1000 for l in lens_ms]
    return bins, lens_sec


def prepare_audio_for_apis(audio_array, sr, max_len_sec=12, verbose=False):
    """
    Recebe o numpy array do dataset e devolve uma lista de byte streams
    (altamente otimizados) prontos para enviar para Azure/GCP.
    """
    max_len_ms = int(max_len_sec * 1000)

    # 1. Array -> Pydub
    audio_segment = array_to_audiosegment(audio_array, sr)

    # 2. Fase Atômica (Quebra nos silêncios)
    atomic_chunks = get_atomic_chunks(audio_segment, max_len_ms=max_len_ms)

    if verbose:
        print(f"Atomic chunk lengths: {[len(c) for c in atomic_chunks]}")

    # 3. Fase Empacotamento (Junta até 29.9s)
    final_segments, lens_sec = pack_audio_chunks(
        atomic_chunks, max_len_ms=max_len_ms, min_len_ms=2000
    )

    if verbose:
        print(f"Final segment lengths: {[len(c) for c in final_segments]}")
    return final_segments, lens_sec


def prepare_api_audio_bytes(audio_array, sr, max_len_sec=12, verbose=False):
    final_segments, lens_sec = prepare_audio_for_apis(
        audio_array, sr, max_len_sec=max_len_sec, verbose=verbose
    )
    byte_chunks = []
    for seg in final_segments:
        buffer = io.BytesIO()
        seg.export(buffer, format="wav")
        byte_chunks.append(buffer.getvalue())

    return byte_chunks, lens_sec


def prepare_api_audio_files(audio_array, sr, max_len_sec=12, verbose=False):
    final_segments, lens_sec = prepare_audio_for_apis(
        audio_array, sr, max_len_sec=max_len_sec, verbose=verbose
    )

    file_names = []
    for seg in final_segments:
        # 1. Ask OS for a temp file, getting both the descriptor and the path
        fd, temp_name = tempfile.mkstemp(suffix=".wav")

        # 2. Immediately close the OS file descriptor so it doesn't leak!
        os.close(fd)

        # 3. Let pydub safely open, write, and close the file
        seg.export(temp_name, format="wav")
        file_names.append(temp_name)

    return file_names, lens_sec
