import io
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence

MAX_CHUNK_DURATION_MS = 58000  # 58 seconds (Safety margin for Google's 60s limit)

def get_audio_duration(file_path):
    """
    Calcula duração usando soundfile (sf.info lê apenas o header, muito rápido).
    """
    try:
        # sf.info retorna um objeto com metadados sem carregar o áudio
        return sf.info(file_path).duration
    except Exception as e:
        print(f"Erro ao calcular duração com soundfile: {e}")
        return 0.0

def generate_safe_chunks(audio_path):
    """
    Splits audio ensuring NO chunk is > 60 seconds.
    1. Tries to split by silence.
    2. Merges small chunks to maximize throughput (fewer API calls).
    3. If a single chunk is still > 60s (rare continuous noise), force splits it.
    
    Returns: List of io.BytesIO objects (WAV format)
    """
    try:
        sound = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio with Pydub: {e}")
        # Fallback: Read raw bytes if pydub fails (risk of being > 60s, but better than crash)
        with open(audio_path, "rb") as f:
            return [io.BytesIO(f.read())]

    # If audio is already short, return it immediately
    if len(sound) <= MAX_CHUNK_DURATION_MS:
        out_buffer = io.BytesIO()
        sound.export(out_buffer, format="wav")
        return [out_buffer]

    print(f"Audio is long ({len(sound)/1000}s). Splitting...")

    # 1. Split on silence
    # Adjust silence_thresh based on your environment noise. -30dB is usually safe for phone calls.
    raw_chunks = split_on_silence(
        sound, 
        min_silence_len=500, 
        silence_thresh=-30, 
        keep_silence=200
    )

    # If silence splitting failed (e.g., constant noise), raw_chunks might be empty or contain 1 huge chunk
    if not raw_chunks:
        raw_chunks = [sound]

    final_buffers = []
    current_segment = AudioSegment.empty()

    for chunk in raw_chunks:
        # Check if adding this chunk exceeds the limit
        if len(current_segment) + len(chunk) < MAX_CHUNK_DURATION_MS:
            current_segment += chunk
        else:
            # The current segment is full, export it
            if len(current_segment) > 0:
                buf = io.BytesIO()
                current_segment.export(buf, format="wav")
                final_buffers.append(buf)
            
            # Start new segment
            # Edge case: If the *single* new chunk is huge (> 60s), we must force split it
            if len(chunk) > MAX_CHUNK_DURATION_MS:
                # Force split big chunk into 58s blocks
                for i in range(0, len(chunk), MAX_CHUNK_DURATION_MS):
                    sub_chunk = chunk[i : i + MAX_CHUNK_DURATION_MS]
                    sub_buf = io.BytesIO()
                    sub_chunk.export(sub_buf, format="wav")
                    final_buffers.append(sub_buf)
                current_segment = AudioSegment.empty()
            else:
                current_segment = chunk

    # Append the remainder
    if len(current_segment) > 0:
        buf = io.BytesIO()
        current_segment.export(buf, format="wav")
        final_buffers.append(buf)

    print(f"Generated {len(final_buffers)} chunks for API processing.")
    return final_buffers

def generate_safe_chunks_iterative(audio_path):
    """
    Splits audio ensuring NO chunk is > 58 seconds.
    Iteratively relaxes silence detection parameters until the audio breaks apart.
    """
    try:
        sound = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio with Pydub: {e}")
        # Panic fallback: Read raw bytes
        with open(audio_path, "rb") as f:
            return [io.BytesIO(f.read())]

    total_len = len(sound)
    # If audio is already short, return it immediately
    if total_len <= MAX_CHUNK_DURATION_MS:
        out_buffer = io.BytesIO()
        sound.export(out_buffer, format="wav")
        return [out_buffer]

    print(f"Audio is long ({total_len/1000}s). Starting iterative split...")

    # Initial parameters (Conservative -> Aggressive)
    min_silence_len = 1000  # Start looking for 700ms pauses
    silence_thresh = -32   # Start assuming silence is -30dB
    
    # Step adjustments
    step_len = 100
    step_thresh = 4
    
    raw_chunks = []
    success = False

    # Retry loop: Keep splitting until every single chunk is smaller than the limit
    # We limit iterations to prevent infinite loops
    for i in range(10):
        print(f"Attempt {i+1}: min_silence={min_silence_len}ms, thresh={silence_thresh}dB")
        
        raw_chunks = split_on_silence(
            sound, 
            min_silence_len=max(100, min_silence_len), # Never go below 100ms
            silence_thresh=silence_thresh, 
            keep_silence=True
        )
        
        # Validation: Are ALL chunks smaller than the limit?
        if not raw_chunks:
            # Nothing found, treat whole audio as one chunk
            longest_chunk = total_len
        else:
            longest_chunk = max(len(c) for c in raw_chunks)

        if longest_chunk < MAX_CHUNK_DURATION_MS:
            print(f"Success! Found split configuration. Longest atomic chunk: {longest_chunk/1000}s")
            success = True
            break
        else:
            print(f"Split failed. Found a chunk of {longest_chunk/1000}s > {MAX_CHUNK_DURATION_MS/1000}s. Relaxing params...")
            # Make it easier to split:
            min_silence_len -= step_len     # Require less silence time
            silence_thresh += step_thresh   # Consider louder noises as "silence"

    # --- Re-Merge Logic ---
    # Now we have many small chunks (hopefully). We merge them back into buckets of ~58s
    # to minimize API calls.
    
    final_buffers = []
    
    # If Iterative split completely failed (e.g. continuous noise), we fall back to hard slicing
    if not success or not raw_chunks:
        print("Iterative split failed (Continuous noise?). Switching to Hard Slicing.")
        raw_chunks = []
        for i in range(0, total_len, MAX_CHUNK_DURATION_MS):
            raw_chunks.append(sound[i : i + MAX_CHUNK_DURATION_MS])

    current_segment = AudioSegment.empty()

    for chunk in raw_chunks:
        # If adding this chunk fits in the bucket
        if len(current_segment) + len(chunk) < MAX_CHUNK_DURATION_MS:
            current_segment += chunk
        else:
            # Bucket full. Export current segment.
            if len(current_segment) > 0:
                buf = io.BytesIO()
                current_segment.export(buf, format="wav")
                final_buffers.append(buf)
            
            # Start new bucket
            # Edge case check: If the single atomic chunk is somehow STILL huge (should be caught by fallback above)
            if len(chunk) > MAX_CHUNK_DURATION_MS:
                # Force slice
                chunk.export(io.BytesIO(), format="wav") # Just strictly for logic correctness
                # Actually if we are here, we just take the big chunk or hard slice it
                sub_buf = io.BytesIO()
                chunk[:MAX_CHUNK_DURATION_MS].export(sub_buf, format="wav")
                final_buffers.append(sub_buf)
                current_segment = chunk[MAX_CHUNK_DURATION_MS:]
            else:
                current_segment = chunk

    # Append remainder
    if len(current_segment) > 0:
        buf = io.BytesIO()
        current_segment.export(buf, format="wav")
        final_buffers.append(buf)

    print(f"Final: Generated {len(final_buffers)} API-ready chunks.")
    return final_buffers

def get_audio_content_and_duration(audio_path, max_seconds=59):
    """
    Lê o arquivo de áudio. Se exceder max_seconds, recorta os últimos segundos.
    Retorna uma tupla (bytes_content, duration_float).
    """
    try:
        # Lê apenas metadados (rápido)
        info = sf.info(audio_path)
        
        if info.duration <= max_seconds:
            # Arquivo curto: leitura direta do disco é mais eficiente
            with open(audio_path, "rb") as f:
                return f.read(), info.duration
        
        # Arquivo longo: Recorta o final na memória
        # print(f"Slicing audio: {info.duration}s -> last {max_seconds}s")
        frames_to_read = int(max_seconds * info.samplerate)
        start_frame = max(0, info.frames - frames_to_read)
        
        # Lê numpy array do trecho final
        data, samplerate = sf.read(audio_path, start=start_frame)
        
        # Re-encoda para WAV em buffer de memória
        mem_buffer = io.BytesIO()
        sf.write(mem_buffer, data, samplerate, format='WAV')
        
        return mem_buffer.getvalue(), float(max_seconds)

    except Exception as e:
        print(f"Erro no processamento de áudio: {e}")
        # Fallback: tenta ler o arquivo inteiro e retornar duração 0 em caso de erro crítico
        # para que o erro real apareça na API do Google ou seja tratado depois
        with open(audio_path, "rb") as f:
            return f.read(), 0.0