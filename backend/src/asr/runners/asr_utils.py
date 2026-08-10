import os
import re
import logging
import string

if os.path.exists("/logs/main.log"):
    logging.basicConfig(filename="/logs/main.log", level=logging.INFO)
logger = logging.getLogger(__name__)

env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(".env", "r").read().split("\n")
    if "=" in rawline
}
for k, v in env_vals.items():
    os.environ[k] = v
print(env_vals)


def remove_duplicates_regex(text: str, max_ngram: int = 5) -> str:
    """
    Remove repetições consecutivas de n-gramas (1..max_ngram).
    Ex.: "fez a de fez a de fez a de" -> "fez a de"
    Mantém separadores (pontuação/espacos) na medida do possível.
    """

    if not text or text.strip() == "":
        return text

    # Tokeniza em palavras (\w+) e não-palavras (separadores)
    tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
    word_indices = [
        i for i, tok in enumerate(tokens) if re.match(r"\w+", tok, flags=re.UNICODE)
    ]
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
            seq = tuple(lower_words[i : i + n])
            j = i + n
            # conta quantas vezes a seq se repete consecutivamente
            while j + n <= L and tuple(lower_words[j : j + n]) == seq:
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
        if re.match(r"\w+", tok, flags=re.UNICODE):
            if keep_word[widx]:
                out.append(tok)
            # se palavra removida, não append; mantemos separadores seguintes normalmente
            widx += 1
        else:
            out.append(tok)

    result = "".join(out)
    # Normaliza espaços extras introduzidos pela remoção
    result = re.sub(r"\s{2,}", " ", result)
    # Remove espaço antes de pontuação (opcional, melhora saída)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    return result.strip()


"""def remove_duplicates_regex_simple(seq):
    my_output = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', seq, flags=re.IGNORECASE)
    return my_output"""


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
    text = re.sub(r"[\(\)\[\]\{\}]", "", text)

    # 2. Replace ellipses and hyphens with spaces
    # ASR models usually ignore hyphens ("vira-lata" -> "vira lata", "MG-230" -> "mg 230")
    # Ellipses in the reference text usually indicate pauses, which ASR interprets as space.
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"-", " ", text)

    # 3. Strip quotation marks and apostrophes
    # These rarely impact spoken-word semantics in this context.
    text = re.sub(r'[\'\"”"“‘`]', "", text)

    # 4. Collapse repeated terminal punctuation
    # Converts panicked typing like "Ajudar!!!" into "Ajudar!"
    text = re.sub(r"\!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r",+", ",", text)

    # 5. Fix spacing around punctuation
    # Prevents WER from penalizing "ajuda ?" vs "ajuda?"
    text = re.sub(r"\s+([.,?!;:>])", r"\1", text)

    # 6. Clean up any resulting double spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_duplicates_regex_simple(text, max_ngram_size=10, loop_threshold=2):
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
            while (
                curr_idx + n <= n_tokens
                and norm_tokens[curr_idx : curr_idx + n] == pattern
            ):
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
