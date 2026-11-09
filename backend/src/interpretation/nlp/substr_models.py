import re
import unicodedata
import joblib
import numpy as np
import scipy.sparse as sp
import Stemmer  # pip install PyStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# 1. Stopwords expandidas para domínio de atendimento de emergência (190/192/193)
PT_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era",
    "essa", "essas", "esse", "esses", "esta", "estas", "este", "estes", "eu",
    "foi", "for", "foram", "ha", "isso", "isto", "ja", "la", "lhe", "lhes",
    "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito",
    "na", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos",
    "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos",
    "por", "qual", "quando", "que", "quem", "sao", "se", "seja", "seu", "seus",
    "so", "sua", "suas", "tambem", "te", "tem", "tendo", "tenho", "ter", "teu",
    "tu", "tua", "um", "uma", "umas", "uns", "voce", "voces", "vos",
    # Stopwords de diálogo / transcrição de emergência (evita falsos matches de alto IDF)
    "nao", "sem", "sim", "ok", "alo", "bom", "dia", "tarde", "noite", "favor",
    "moco", "moca", "senhor", "senhora", "operador", "solicitante", "roteiro",
    "descricao", "breve", "relata", "relato", "informa", "informacoes", "detalhes",
    "chamada", "ligacao", "atendimento", "equipe", "viatura", "ajuda", "caminho",
    "proximo", "proxima", "perto", "frente", "lado", "numero", "bloco", "lote",
    "quadra", "bairro", "centro", "cidade", "endereco", "referencia", "ponto",
    "area", "onde", "aqui", "ali", "algo", "alguem", "ninguem", "outro", "outra",
    "outros", "outras", "parece", "estou", "esta", "estao", "ficar", "fica",
    "fique", "pode", "posso", "consegue", "consigo", "sabe", "sei", "ver", "vendo"
}

# 2. Radicais ultra-genéricos em chamadas que NÃO devem ter alto peso como unigramas soltos
GENERIC_STEM_WEIGHTS = {
    "pesso": 0.12,     # Evita inflar "Pessoa {X}" quando a chamada só diz "pessoas correndo"
    "rua": 0.10,       # Evita inflar "Pessoa em Situação de Rua" por causa de endereços
    "local": 0.15,     # Evita inflar "Perícia em Local de..." e "Estacionar em Local..."
    "localiz": 0.20,   # Evita inflar "Localização de..." por "Localizado na rua X"
    "cas": 0.25,       # "em casa" vs "Casa de Pesca"
    "predi": 0.30,
    "hom": 0.20,
    "mulh": 0.20,
    "vizinh": 0.20,
    "morador": 0.20,   # Evita inflar "Abordagem a Morador de Rua" quando diz "Morador relata"
    "veicul": 0.30,
    "carr": 0.30,
    "estacion": 0.20,  # Evita inflar 28 multas de "Estacionar em Local Proibido"
    "emergenc": 0.25,
    "acident": 0.35,
    "vitim": 0.25,
    "pres": 0.35,      # Só ganha peso forte se vier acompanhado do contexto (ou bigrama)
}

# 3. Expansão léxica de domínio (conecta termos coloquiais das chamadas às naturezas oficiais)
QUERY_SYNONYM_RULES = [
    # Desambiguação crítica: "fogo" sem "arma" -> "incendio"
    (re.compile(r"\b(fogo|chama|chamas|fumaca|queimando|incendiando)\b"), " incendio "),
    # Desabamento / Colapso estrutural / Subsidência
    (re.compile(r"\b(desabando|desmoronando|rachadura|rachaduras|estalo|estalos|afundamento|afundamentos|tijolos|cedendo)\b"),
     " risco desabamento desmoronamento subsidencias colapsos "),
    # Queda da própria altura
    (re.compile(r"\b(tropecou|escorregou|caiu em casa|caida em casa|caiu no chao|caiu na calcada|caiu no jardim)\b"),
     " queda propria altura "),
    # Mal súbito / Desmaio
    (re.compile(r"\b(desmaiado|desmaiada|desmaiou|desacordado|desacordada|tonto|tonta|zonzo|mal subito)\b"),
     " desmaio tontura mal subito "),
    # Engasgo / OVACE
    (re.compile(r"\b(engasgo|engasgado|engasgada|engasgou|desobstrucao)\b"),
     " obstrucao via aerea corpo estranho ovace "),
    # Convulsão / Epilepsia
    (re.compile(r"\b(convulsao|convulsionando|epiletico|epilepsia|babando)\b"),
     " ataque epiletico convulsao "),
    # Pedido de remédio
    (re.compile(r"\b(remedio|medicamento|insulina)\b"),
     " pedido remedio "),
    # Banhista / Afogamento
    (re.compile(r"\b(banhista|afogando|piscina|caimbra|parte rasa)\b"),
     " atendimento banhista afogamento "),
    # Violência doméstica / Agressão
    (re.compile(r"\b(marido|esposa|companheiro|companheira|namorado|ex-marido)\b"),
     " violencia domestica maria penha "),
    (re.compile(r"\b(batendo|espancando|agredindo|agrediu|chutando|chutou|socos)\b"),
     " agressao fisica lesao corporal rixa briga "),
    # Árvore
    (re.compile(r"\b(galho|galhos|tronco)\b"),
     " queda ameaca arvore vistoria "),
]


def _normalize_str(text: str) -> str:
    text_norm = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text_norm if not unicodedata.combining(c))


def _protect_compounds(text_clean: str) -> str:
    """Protege termos compostos antes da tokenização para evitar falsos positivos unigrama."""
    # "arma de fogo" vira um único token "armadefog", impedindo match com "fogo no lixo/casa"
    text_clean = re.sub(r"\barma\s+de\s+fogo\b", " armadefog ", text_clean)
    text_clean = re.sub(r"\bsituacao\s+de\s+rua\b", " situacaoderua ", text_clean)
    text_clean = re.sub(r"\bmorador\s+de\s+rua\b", " moradorderua ", text_clean)
    return text_clean


def _clean_emergency_query(raw_text: str) -> str:
    """
    Remove falas do operador (que contêm perguntas especulativas) e remove nomes de ruas/endereços
    para que 'na Rua X' não dispare 'Pessoa em Situação de Rua'.
    """
    # Se for transcrição multi-linha com "operador:" e "solicitante:", filtra apenas o solicitante
    lines = raw_text.split("\n")
    if len(lines) > 1 and any("operador:" in l.lower() for l in lines):
        kept_lines = [
            l for l in lines
            if not l.strip().lower().startswith(("operador:", "roteiro: operador:"))
        ]
        if kept_lines:
            raw_text = "\n".join(kept_lines)

    text_clean = _normalize_str(raw_text)

    # Remove cabeçalhos de campos estruturados
    text_clean = re.sub(r"\b(roteiro|solicitante|descricao_breve|outras_observacoes)\s*:", " ", text_clean)

    # Remove referências de endereço ("na rua magnolia", "rua sao luis, 983", "av. b q")
    text_clean = re.sub(
        r"\b(na\s+|em\s+|da\s+)?(rua|avenida|av|rodovia|travessa|alameda|estrada|sqn|sqs|smas|qi)\s+[a-z0-9\s\-\.]{1,30}?(?=[,\.\n\(\)]|\s+numero|\s+n\b|\s+perto|\s+proximo|$)",
        " ",
        text_clean
    )
    # Remove "fala na rua" / "ficam na rua" / "carros na rua" quando não é morador/situação de rua
    text_clean = _protect_compounds(text_clean)
    text_clean = re.sub(r"\b(na|da|pela)\s+rua\b", " via_publica ", text_clean)

    # Aplica regras de expansão de sinônimos apenas se não houver contexto contraditório
    for pattern, expansion in QUERY_SYNONYM_RULES:
        if pattern.search(text_clean):
            text_clean += " " + expansion

    return text_clean


class StemmedWordTokenizer:
    """Tokenizador com proteção de compostos, remoção de stopwords e Stemming (compatível com joblib)."""
    def __init__(self, is_query: bool = False):
        self.is_query = is_query
        self._stemmer = None
        self._token_pattern = re.compile(r"\b[a-z_][a-z0-9_]+\b")

    @property
    def stemmer(self):
        if self._stemmer is None:
            self._stemmer = Stemmer.Stemmer("portuguese")
        return self._stemmer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stemmer"] = None
        return state

    def __call__(self, text: str) -> list[str]:
        if self.is_query:
            text_clean = _clean_emergency_query(text)
        else:
            text_clean = _protect_compounds(_normalize_str(text))

        tokens = [
            t for t in self._token_pattern.findall(text_clean)
            if t not in PT_STOPWORDS and not t.isdigit()
        ]
        return self.stemmer.stemWords(tokens)


class RobustBM25Model:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.35,
        char_weight: float = 0.15,
        saturation_tau: float = 2.5,
        normalize_scores: bool = True
    ):
        self.k1 = k1
        self.b = b
        self.char_weight = char_weight
        self.saturation_tau = saturation_tau
        self.normalize_scores = normalize_scores

        self.word_vec = CountVectorizer(
            tokenizer=StemmedWordTokenizer(is_query=False),
            token_pattern=None,
            ngram_range=(1, 2)
        )
        self.query_tokenizer = StemmedWordTokenizer(is_query=True)

        # Caracteres de 4 a 5 letras (evita colisões ruidosas de sufixos de 3 letras)
        self.char_vec = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(4, 5),
            strip_accents="unicode"
        )

    def _fit_branch(self, X_counts: sp.csr_matrix, feature_names: np.ndarray = None):
        X_counts = X_counts.tocsc()
        n_docs, n_features = X_counts.shape
        df = np.bincount(X_counts.indices, minlength=n_features)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Amortecimento de unigramas genéricos ("pesso", "rua", "local", etc.),
        # preservando 100% do IDF quando fazem parte de um bigrama ("situaca rua", "pesso desaparec")
        if feature_names is not None:
            for idx, feat in enumerate(feature_names):
                if " " not in feat and feat in GENERIC_STEM_WEIGHTS:
                    idf[idx] *= GENERIC_STEM_WEIGHTS[feat]
                elif " " in feat:
                    idf[idx] *= 1.35  # Bônus discriminativo para match de bigrama

        avgdl = float(X_counts.sum(axis=1).mean())
        return idf, avgdl

    def _transform_branch(
        self, X: sp.csr_matrix, idf: np.ndarray, avgdl: float, l2_norm: bool = True
    ) -> sp.csr_matrix:
        X = X.tocsr().astype(np.float32)
        if X.nnz == 0:
            return X
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        rows, cols = X.nonzero()
        tf = X.data

        norm_factor = self.k1 * (1.0 - self.b + self.b * (doc_len[rows] / max(avgdl, 1.0)))
        X.data = (tf * (self.k1 + 1.0)) / (tf + norm_factor) * idf[cols]

        if l2_norm and self.normalize_scores:
            X = normalize(X, norm="l2", axis=1)
        return X

    def fit(self, categories: list[str]):
        self.categories_ = list(categories)

        X_words = self.word_vec.fit_transform(self.categories_)
        word_features = self.word_vec.get_feature_names_out()
        self.word_idf_, self.word_avgdl_ = self._fit_branch(X_words, word_features)
        W_mat = self._transform_branch(X_words, self.word_idf_, self.word_avgdl_, l2_norm=True)

        X_chars = self.char_vec.fit_transform(self.categories_)
        self.char_idf_, self.char_avgdl_ = self._fit_branch(X_chars, None)
        C_mat = self._transform_branch(X_chars, self.char_idf_, self.char_avgdl_, l2_norm=True)

        self.category_matrix_ = sp.hstack([W_mat, C_mat * self.char_weight]).tocsr()
        if self.normalize_scores:
            self.category_matrix_ = normalize(self.category_matrix_, norm="l2", axis=1)
        return self

    def predict_similarity(self, description: str) -> dict[str, float]:
        """Retorna um dicionário {category: similarity} com a similaridade calibrada [0.0 a 1.0]."""
        cleaned_query = _clean_emergency_query(description)

        # Vetorização da query com o tokenizer de query
        q_tokens = self.query_tokenizer(description)
        q_joined = " ".join(q_tokens)
        # Monta contagem sobre vocabulário já ajustado
        q_word_counts = self.word_vec.transform([cleaned_query])
        q_char_counts = self.char_vec.transform([cleaned_query])

        # Na query, NÃO aplicamos L2 isolado antes do produto escalar para evitar que 1 único
        # token genérico ("pessoa") receba norma 1.0
        q_words_raw = self._transform_branch(
            q_word_counts, self.word_idf_, self.word_avgdl_, l2_norm=False
        )
        q_chars_raw = self._transform_branch(
            q_char_counts, self.char_idf_, self.char_avgdl_, l2_norm=False
        )

        q_vec_raw = sp.hstack([q_words_raw, q_chars_raw * self.char_weight]).tocsr()
        if q_vec_raw.nnz == 0:
            return {doc: 0.0 for doc in self.categories_}

        # Produto escalar bruto (mede quanta evidência real de IDF foi encontrada em cada categoria)
        raw_dots = (self.category_matrix_ @ q_vec_raw.T).toarray().ravel()

        if self.normalize_scores:
            # Saturação assintótica suave: impede que matches de 1 palavra genérica cheguem a 0.45+,
            # mas garante que matches de múltiplos termos ou bigramas cheguem perto de 0.60 - 0.85
            sims_final = raw_dots / (raw_dots + self.saturation_tau)
        else:
            sims_final = raw_dots

        return {self.categories_[i]: float(sim) for i, sim in enumerate(sims_final)}

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    def load(self, filepath: str) -> "RobustBM25Model":
        loaded = joblib.load(filepath)
        self.__dict__.update(loaded.__dict__)
        return self