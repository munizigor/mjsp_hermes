
import re
import string
import time
import os
from glob import glob

from rapidfuzz import fuzz
from datasets import load_dataset, Audio, load_from_disk
import pandas as pd
import unidecode

colunas = [
    "ID",
    "audio",
    "modelo_audio",
    "roteiro_segmentado",
]

hf_dataset = "pitagoras-alves/fake-emergencies-br"

def get_inputs(max_n=4):
    #Only load if a file with the correct sampling has not been sampled and saved yet
    df_path = f"inputs/df_{max_n}.parquet"
    if os.path.exists(df_path):
        print(f"Loading from {df_path}")
        ds = load_from_disk(df_path)
        return ds
    else:
        print("Downloading dataset...")
        ds = load_dataset(
            hf_dataset,
            split="train",
            streaming=False,
            columns=colunas,
        )#.take(max_n)
        ds = ds.shuffle(seed=1337)
        ds = ds.select(range(max_n))
        print("Casting audio column...")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        sample = next(iter(ds))["audio"]["array"]
        print("Sample:", sample.shape)
        os.makedirs("inputs", exist_ok=True)
        ds.save_to_disk(df_path)
        return ds

def eval_transcriptions(pred, true):
    from evaluate import load
    wer_metric = load("wer")
    return wer_metric.compute(references=true, predictions=pred)

def normalize_light(text):
    text = text.lower()
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_smart_light(text):
    """
    Simplifies structural punctuation and ASR formatting artifacts 
    without losing accents, questions, or exclamations.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # 1. Remove parentheses and brackets
    # The original scripts have stage directions like "(Choro fraco)".
    # The ASR transcribes the words but omits the brackets. 
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)
    
    # 2. Replace ellipses and hyphens with spaces
    # ASR models usually ignore hyphens ("vira-lata" -> "vira lata", "MG-230" -> "mg 230")
    # Ellipses in the reference text usually indicate pauses, which ASR interprets as space.
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'-', ' ', text)
    
    # 3. Strip quotation marks and apostrophes
    # These rarely impact spoken-word semantics in this context.
    text = re.sub(r'[\'\"”"“‘`]', '', text)
    
    # 4. Collapse repeated terminal punctuation
    # Converts panicked typing like "Ajudar!!!" into "Ajudar!"
    text = re.sub(r'\!+', '!', text)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r',+', ',', text)
    
    # 5. Fix spacing around punctuation
    # Prevents WER from penalizing "ajuda ?" vs "ajuda?"
    text = re.sub(r'\s+([.,?!;:>])', r'\1', text)
    
    # 6. Clean up any resulting double spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def normalize_special(text):
    text = text.lower()
    # Remover caracteres especiais
    text = unidecode.unidecode(text)
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_punct(text):
    text = text.lower()
    
    # FIX: Em vez de apenas remover a pontuação, substitua por espaço!
    # Isso impede que falhas de espaçamento como "mordido?nao" virem "mordidonao"
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def intelligent_chunk_merge(chunks):
    if not chunks:
        return ""
    
    merged_text = chunks[0].strip()
    
    for i in range(1, len(chunks)):
        prev = merged_text.strip()
        curr = chunks[i].strip()
        
        # 1. Remove pontuações geradas por cortes abruptos nas bordas
        prev_clean = re.sub(r'[\.\,\;\:\-\…]+$', '', prev)
        curr_clean = re.sub(r'^[\.\,\;\:\-\…]+', '', curr)
        
        prev_words = prev_clean.split()
        curr_words = curr_clean.split()
        
        if prev_words and curr_words:
            last_word = prev_words[-1].lower()
            first_word = curr_words[0].lower()
            
            # 2. Trata duplicações exatas
            if last_word == first_word:
                curr_words.pop(0)
                
            # 3. Trata palavras fragmentadas pelo corte
            elif first_word.startswith(last_word) or last_word.startswith(first_word):
                if len(first_word) > len(last_word):
                    prev_words.pop()
                else:
                    curr_words.pop(0)
                    
        # Junta o texto limpo
        prev_str = " ".join(prev_words)
        curr_str = " ".join(curr_words)
        
        # 4. Ajusta a capitalização para decidir se foi uma quebra de frase
        if curr_str:
            if curr_str[0].islower():
                merged_text = prev_str + " " + curr_str
            else:
                if prev and prev[-1] in ".!?":
                    merged_text = prev_str + prev[-1] + " " + curr_str
                else:
                    merged_text = prev_str + ". " + curr_str
        else:
            merged_text = prev_str

    # 5. Varredura final de limpeza
    # Remove espaço antes de pontuação
    merged_text = re.sub(r'\s+([.,?!])', r'\1', merged_text) 
    
    # FIX: Garante que toda pontuação seja seguida de espaço se a próxima letra for texto
    # Resolve os casos como: "acontecendo?Ajudar!" -> "acontecendo? Ajudar!"
    merged_text = re.sub(r'([.!?,\:])([a-zA-ZÀ-ÿ])', r'\1 \2', merged_text)
    
    # FIX: Separa palavras coladas alucinadas pelo modelo em CamelCase 
    # Resolve os casos como: "seVocê" -> "se Você"
    merged_text = re.sub(r'([a-zà-ÿ])([A-ZÀ-Ÿ])', r'\1 \2', merged_text)
    
    # Limpa espaços extras que possam ter sido gerados nas etapas acima
    merged_text = re.sub(r'\s+', ' ', merged_text)
    
    return merged_text.strip()

def clean_reference_text(text):
    # 1. Remove o artefato de junção " . " (espaço, ponto, espaço) ou pontos isolados no final
    clean_text = re.sub(r'\s+\.\s+', ' ', text)
    clean_text = re.sub(r'\s+\.$', '', clean_text)
    
    # 2. Corrige pontuações duplas (ex: "! ." vira apenas "!")
    clean_text = re.sub(r'([?!,])\s*\.', r'\1', clean_text)
    
    # 3. Remove as reticências (...) que marcam hesitação
    # Substituímos por um espaço para evitar colar palavras (ex: "A...a gente" -> "A a gente")
    clean_text = re.sub(r'\.{2,}', ' ', clean_text)
    
    # 4. Remove múltiplos espaços gerados pelas substituições anteriores
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text.strip()

def remove_duplicates_regex(text: str, max_ngram: int = 5) -> str:
    """
    Remove repetições consecutivas de n-gramas (1..max_ngram).
    Ex.: "fez a de fez a de fez a de" -> "fez a de"
    Mantém separadores (pontuação/espacos) na medida do possível.
    """

    if not text or text.strip() == "":
        return text

    # Tokeniza em palavras (\w+) e não-palavras (separadores)
    tokens = re.findall(r'\w+|\W+', text, flags=re.UNICODE)
    word_indices = [i for i, tok in enumerate(tokens) if re.match(r'\w+', tok, flags=re.UNICODE)]
    words = [tokens[i] for i in word_indices]
    lower_words = [w.lower() for w in words]

    if not words:
        return text

    keep_word = [True] * len(words)
    i = 0
    L = len(words)

    while i < L:
        matched = False
        # tenta maiores n-grams primeiro
        max_n = min(max_ngram, L - i)
        for n in range(max_n, 0, -1):
            seq = tuple(lower_words[i:i + n])
            j = i + n
            # conta quantas vezes a seq se repete consecutivamente
            while j + n <= L and tuple(lower_words[j:j + n]) == seq:
                j += n
            if j > i + n:
                # houve repetição: marca palavras repetidas para remoção
                for k in range(i + n, j):
                    keep_word[k] = False
                i = j  # pula bloco repetido
                matched = True
                break
        if not matched:
            i += 1

    # Reconstrói texto: mantém separadores e apenas palavras marcadas
    out = []
    widx = 0
    for idx, tok in enumerate(tokens):
        if re.match(r'\w+', tok, flags=re.UNICODE):
            if keep_word[widx]:
                out.append(tok)
            # se palavra removida, não append; mantemos separadores seguintes normalmente
            widx += 1
        else:
            out.append(tok)

    result = ''.join(out)
    # Normaliza espaços extras introduzidos pela remoção
    result = re.sub(r'\s{2,}', ' ', result)
    # Remove espaço antes de pontuação (opcional, melhora saída)
    result = re.sub(r'\s+([,.;:!?])', r'\1', result)
    return result.strip()

def normalize_repeats(seq):
    no_dups1 = remove_duplicates_regex(seq)
    no_dups2 = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', no_dups1, flags=re.IGNORECASE)
    return no_dups2

def remove_duplicates_fuzzy(text: str, max_ngram: int = 5, sim_threshold=90):
    if not text.strip():
        return text

    tokens = re.findall(r'\w+|\W+', text, flags=re.UNICODE)
    word_indices = [i for i, tok in enumerate(tokens) if re.match(r'\w+', tok)]
    words = [tokens[i] for i in word_indices]

    keep = [True] * len(words)
    i = 0
    L = len(words)

    def similar(a, b):
        return fuzz.partial_ratio(" ".join(a), " ".join(b)) >= sim_threshold

    while i < L:
        matched = False
        for n in range(min(max_ngram, L - i), 0, -1):
            seq = words[i:i+n]
            j = i + n

            while j + n <= L and similar(seq, words[j:j+n]):
                j += n

            if j > i + n:
                for k in range(i + n, j):
                    keep[k] = False
                i = j
                matched = True
                break

        if not matched:
            i += 1

    # reconstruct
    out = []
    widx = 0
    for tok in tokens:
        if re.match(r'\w+', tok):
            if keep[widx]:
                out.append(tok)
            widx += 1
        else:
            out.append(tok)

    result = ''.join(out)
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+([,.;:!?])', r'\1', result)
    return result.strip()

import string

def remove_asr_hallucination_loops(text, max_phrase_len=10, loop_threshold=2):
    """
    Removes repeating n-grams only if they repeat more than `loop_threshold` times.
    Preserves natural human stutters (e.g., "ele ele") but catches infinite ASR loops.
    """
    tokens = text.split()
    if not tokens:
        return text

    i = 0
    result = []
    
    while i < len(tokens):
        best_match_len = 0
        best_match_count = 0

        # Look for repeating phrases, starting from the largest allowed window down to 1
        for phrase_len in range(max_phrase_len, 0, -1):
            if i + phrase_len * 2 > len(tokens): 
                continue # Not enough words left to form a repeat

            # Clean tokens for comparison (ignore case and punctuation)
            current_phrase = [t.lower().strip(string.punctuation) for t in tokens[i : i+phrase_len]]

            count = 1
            curr_idx = i + phrase_len
            
            # Count how many times this exact phrase repeats immediately after
            while curr_idx + phrase_len <= len(tokens):
                next_phrase = [t.lower().strip(string.punctuation) for t in tokens[curr_idx : curr_idx+phrase_len]]
                if current_phrase == next_phrase:
                    count += 1
                    curr_idx += phrase_len
                else:
                    break

            # If the phrase repeats more times than our natural stutter threshold
            if count > loop_threshold and count > best_match_count:
                best_match_len = phrase_len
                best_match_count = count

        if best_match_len > 0:
            # We found an ASR loop! Keep only one instance of the phrase.
            result.extend(tokens[i : i+best_match_len])
            # Skip the pointer past all the hallucinated duplicate blocks
            i += best_match_len * best_match_count 
        else:
            # No loop found, keep the current token and move forward
            result.append(tokens[i])
            i += 1

    return " ".join(result)

from rapidfuzz.distance import Levenshtein
import string
import unidecode

def normalize_token(t):
    return unidecode.unidecode(t.lower().strip(string.punctuation))

def is_similar_phrase(a, b, phrase_len):
    a_str = " ".join(a)
    b_str = " ".join(b)

    if phrase_len <= 3:
        return Levenshtein.distance(a_str, b_str) <= 1
    else:
        return a == b  # strict for longer phrases

def remove_asr_loops_v2(text, max_phrase_len=10, loop_threshold=2):
    tokens = text.split()
    if not tokens:
        return text

    norm_tokens = [normalize_token(t) for t in tokens]

    i = 0
    result = []

    while i < len(tokens):
        best_len = 0
        best_count = 0

        for phrase_len in range(max_phrase_len, 0, -1):
            if i + phrase_len * 2 > len(tokens):
                continue

            base = norm_tokens[i:i+phrase_len]

            count = 1
            j = i + phrase_len

            while j + phrase_len <= len(tokens):
                candidate = norm_tokens[j:j+phrase_len]

                if is_similar_phrase(base, candidate, phrase_len):
                    count += 1
                    j += phrase_len
                else:
                    break

            if count > loop_threshold and count * phrase_len >= 4:
                if count > best_count:
                    best_len = phrase_len
                    best_count = count

        if best_len > 0:
            result.extend(tokens[i:i+best_len])
            i += best_len * best_count
        else:
            result.append(tokens[i])
            i += 1

    return " ".join(result)

def remove_asr_hallucination_loops_fast(text, max_ngram_size=10, loop_threshold=2):
    """
    Fast contiguous n-gram deduplicator.
    Identifies and collapses repeating blocks of words (ASR loops) 
    while preserving natural human disfluencies and stop words.
    """
    tokens = text.split()
    if not tokens:
        return text
        
    n_tokens = len(tokens)
    
    # Pre-compute a cleaned version of tokens for fast, punctuation-agnostic comparison.
    # List comprehensions are highly optimized in Python.
    norm_tokens = [t.strip(string.punctuation).lower() for t in tokens]
    
    result_indices = []
    i = 0
    
    while i < n_tokens:
        best_n = 0
        best_count = 0
        
        # Check largest possible n-grams first to catch long hallucination loops
        max_possible_n = min(max_ngram_size, (n_tokens - i) // 2)
        
        for n in range(max_possible_n, 0, -1):
            pattern = norm_tokens[i : i + n]
            count = 1
            curr_idx = i + n
            
            # Fast sequence comparison (Python does this at C-speed)
            while curr_idx + n <= n_tokens and norm_tokens[curr_idx : curr_idx + n] == pattern:
                count += 1
                curr_idx += n
                
            # If the block repeats beyond our natural stutter threshold
            if count > loop_threshold and count > best_count:
                best_n = n
                best_count = count
                
        if best_n > 0:
            # We found an ASR loop! Keep only the first instance of the phrase.
            result_indices.extend(range(i, i + best_n))
            # Skip the pointer past all the hallucinated duplicate blocks
            i += best_n * best_count 
        else:
            # No loop found, keep the current token
            result_indices.append(i)
            i += 1
            
    # Reconstruct the string using the original tokens to preserve formatting
    return " ".join([tokens[idx] for idx in result_indices])

def save_results(raw_result_lines, save_dir):
    df = pd.DataFrame(raw_result_lines)
    eval_csv_paths = glob(os.path.join(save_dir, "eval_results.*.csv"))
    current_result_n = 0
    for p in eval_csv_paths:
        p_n = int(p.split(".")[-2])
        if p_n >= current_result_n:
            current_result_n = p_n+1

    raw_results_basename = os.path.join(save_dir, f"raw_results.{current_result_n}")
    eval_basename = os.path.join(save_dir, f"eval_results.{current_result_n}")
    #save csv and excel
    df.to_csv(f"{raw_results_basename}.csv", index=False)
    df.to_excel(f"{raw_results_basename}.xlsx", index=False)
    
    eval_lines = []

    col_pairs = [
        ('wer_raw', 'original', 'transcribed'),
        ('wer_basic_norm', 'original_norm', 'transcribed_norm'),
        ('wer_special', 'original_special', 'transcribed_special'),
        ('wer_punct', 'original_punct', 'transcribed_punct'),
        ('wer_repeats', 'original_repeats', 'transcribed_repeats'),
    ]

    if not "chunking_speedup" in df.columns:
        df["chunking_speedup"] = 1.0
    else:
        #fill NaN with 1.0
        df["chunking_speedup"] = df["chunking_speedup"].fillna(1.0)

    for model_id, model_lines in df.groupby('model_id'):
        print(f"Evaluating {model_id}...")
        
        eval_line = {
            "model_id": model_id,
            "n_samples": len(model_lines)
        }
        for new_col, original_col, transcribed_col in col_pairs:
            if original_col in model_lines.columns and transcribed_col in model_lines.columns:
                transcribed_list = model_lines[transcribed_col].tolist()
                originals_list = model_lines[original_col].tolist()

                wer_value = eval_transcriptions(transcribed_list, originals_list)
                print(f"\t{new_col}: {wer_value}")
                eval_line[new_col] = wer_value
        eval_line['total_audio_seconds'] = model_lines['audio_len'].sum()
        eval_line['total_seconds_pipeline'] = model_lines['processing_time'].sum()
        eval_line['total_seconds_chunks'] = (model_lines['processing_time'] / model_lines['chunking_speedup']).sum()
        eval_line['speed_pipeline'] = eval_line['total_audio_seconds'] / eval_line['total_seconds_pipeline']
        eval_line['speed_chunks'] = eval_line['total_audio_seconds'] / eval_line['total_seconds_chunks']
        #In a real scenario, the audio would arrive already chunked.
        #Because of this, we need to estimate what would be the total processing time if the audio was already in small chunks.
        #To do this, we divide the processing time by the speedup factor.
        eval_lines.append(eval_line)
    
    eval_df = pd.DataFrame(eval_lines)
    #sort df by wer_norm
    eval_df = eval_df.sort_values(by='wer_raw')
    eval_df.to_csv(f"{eval_basename}.csv", index=False)
    eval_df.to_excel(f"{eval_basename}.xlsx", index=False)