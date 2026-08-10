import json
import sys
import time
import os
from collections import deque
from itertools import islice
from typing import List
from random import shuffle

env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(".env", "r").read().split("\n")
    if "=" in rawline
}
for k, v in env_vals.items():
    os.environ[k] = v


from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class EmergencyEntities(BaseModel):
    rua_ou_logradouro: Optional[List[str]] = Field(
        None, description="Nome de rua, logradouro ou avenida"
    )
    # rua: Optional[List[str]] = Field(None, description="Nome de rua")
    bairro: Optional[List[str]] = Field(None, description="Nome de bairro")
    municipio: Optional[List[str]] = Field(None, description="Nome de município")
    cidade: Optional[List[str]] = Field(None, description="Nome de cidade")
    ponto_de_referencia: Optional[List[str]] = Field(
        None, description="Nome de ponto de referência do endereço"
    )
    nome_do_solicitante: Optional[List[str]] = Field(
        None, description="Nome do solicitante (pessoa que está fazendo o chamado)"
    )
    """pessoa: Optional[List[str]] = Field(
        None, description="Nome de pessoa ou participante"
    )"""
    numero: Optional[List[str]] = Field(None, description="Número do endereço")
    # street_number: Optional[List[str]] = Field(None, description="Street Number")
    # number: Optional[List[str]] = Field(None, description="Número")
    """complemento: Optional[List[str]] = Field(
        None, description="Complemento do endereço (Exemplos: apt 301, casa A, etc)"
    )"""
    endereço_complemento: Optional[List[str]] = Field(
        None,
        description="Número ou código da casa, apartamento ou loja naquele endereço",
    )


class EmergencyClassifications(BaseModel):
    fato_ocorrendo_neste_momento: Literal["Sim", "Não"] = Field(
        "Não", description="Se o fato relatado está ocorrendo neste momento"
    )
    autor_do_fato_no_local: Literal["Sim", "Não"] = Field(
        "Não", description="Se o autor (culpado/acusado) do fato está no local"
    )
    autor_do_fato_armado: Literal["Sim", "Não"] = Field(
        "Não",
        description="Se o autor (culpado/acusado/suspeito) do fato estava armado",
    )
    feridos_com_risco_de_morte: Literal["Sim", "Não"] = Field(
        "Não", description="Se a ocorrência envolve feridos com risco de morte"
    )
    risco_de_tumulto: Literal["Sim", "Não"] = Field(
        "Não", description="Se a ocorrência envolve um risco de tumulto"
    )
    lei_maria_da_penha: Literal["Sim", "Não"] = Field(
        "Não",
        description="Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
    )


class FullSchema(BaseModel):
    entities: EmergencyEntities
    classifications: EmergencyClassifications


schema_key_translations = {
    "fato_ocorrendo_neste_momento": "Fato Ocorrendo Neste Momento ?",
    "autor_do_fato_no_local": "Autor Do Fato No Local ?",
    "autor_do_fato_armado": "Autor Do Fato Armado ?",
    "feridos_com_risco_de_morte": "Feridos Com Risco de Morte ?",
    "risco_de_tumulto": "Risco De Tumulto ?",
    "lei_maria_da_penha": "Lei Maria da Penha ?",
}


def get_ner_schema() -> dict:

    fatoOcorrendoNesteMomento = "Fato Ocorrendo Neste Momento ?"
    autorDoFatoNoLocal = "Autor Do Fato No Local ?"
    autorDoFatoArmado = "Autor Do Fato Armado ?"
    feridosComRiscoDeMorte = "Feridos Com Risco de Morte ?"
    riscoDeTumulto = "Risco De Tumulto ?"
    leiMariaPenha = "Lei Maria da Penha ?"

    clfnames = [
        fatoOcorrendoNesteMomento,
        autorDoFatoNoLocal,
        autorDoFatoArmado,
        feridosComRiscoDeMorte,
        riscoDeTumulto,
        leiMariaPenha,
    ]

    clfs_bool = {
        fatoOcorrendoNesteMomento: {
            "desc": "Se o fato relatado está ocorrendo neste momento",
            "sim": "Sim",
            "nao": "Não",
        },
        autorDoFatoNoLocal: {
            "desc": "Se o autor (culpado/acusado) do fato está no local",
            "sim": "Sim",
            "nao": "Não",
        },
        autorDoFatoArmado: {
            "desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado",
            "sim": "Sim",
            "nao": "Não",
        },
        feridosComRiscoDeMorte: {
            "desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Sim",
            "nao": "Não",
        },
        riscoDeTumulto: {
            "desc": "Se a ocorrência envolve um risco de tumulto",
            "sim": "Sim",
            "nao": "Não",
        },
        leiMariaPenha: {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
            "sim": "Sim",
            "nao": "Não",
        },
    }

    full_schema_dict = {
        "entities": {
            # "rua_ou_logradouro": "Nome de rua, logradouro ou avenida",
            "rua": "Nome de rua",
            "bairro": "Nome de bairro",
            "municipio": "Nome de município",
            "cidade": "Nome de cidade",
            "ponto_de_referencia": "Nome de ponto de referência do endereço",
            "nome_do_solicitante": "Nome do solicitante (pessoa que está fazendo o chamado)",
            # "pessoa": "Nome de pessoa ou participante",
            "numero": "Número do endereço",
            # "street_number": "Street Number",
            # "number": "Número",
            # "complemento": "Complemento do endereço (Exemplos: apt 301, casa A, etc)",
            "endereço_complemento": "Número ou código da casa, apartamento ou loja naquele endereço",
        },
        "boolean": {"natureza_da_ocorrencia": clfs_bool},
    }

    return full_schema_dict


# Load the GLiNER model for Portuguese
GLINER_MODEL_NAME = "knowledgator/gliner-x-large"

ner_labels = {
    "emergency_basic": [
        # "rua_ou_logradouro",
        "rua",
        "bairro",
        "municipio",
        "cidade",
        "ponto_de_referencia",
        "nome_do_solicitante",
        # "pessoa",
        "numero",
        # "street_number",
        # "number",
        # "complemento",
        "endereço_complemento",
    ],
    "emergency_gliclass": get_ner_schema(),
}

ner_labels_pydantic = {
    "emergency_gliclass": FullSchema,
}

sinonimos = {
    "rua": "rua_ou_logradouro",
    "street_number": "numero",
    "number": "numero",
    "cidade": "municipio",
    "endereço_complemento": "complemento",
}

many_values = {"pessoa", "numero", "number"}

values_to_ignore = {
    "nome_do_solicitante": [
        "solicitante",
        "operador",
        "atendente",
        "vitima",
        "testemunha",
        "vizinho",
        "senhor",
    ]
}


def load_gliner_cuda(model_name_str=GLINER_MODEL_NAME, use_cuda=True):

    from gliner import GLiNER
    from gliner.data_processing.tokenizer import (
        TokenSplitterBase,
        SpaCyTokenSplitter,
        WordsSplitter,
    )

    word_splitter_name = "spacy"
    word_splitter = WordsSplitter(splitter_type=word_splitter_name)
    print("Loading GLiNER model...")
    if use_cuda:
        try:
            gliner_large_model = GLiNER.from_pretrained(
                model_name_str, words_splitter=word_splitter
            ).to("cuda")
            print("Loaded GLiNER model")
        except Exception as e:
            print(f"Error loading GLiNER model with CUDA: {e}")
            print("Falling back to CPU...")
            # If CUDA is not available, load the model on CPU
            gliner_large_model = GLiNER.from_pretrained(
                model_name_str, words_splitter=word_splitter
            )
            print("Loaded GLiNER model")
    else:
        print("Loading GLiNER model on CPU...")
        # Load the model on CPU
        gliner_large_model = GLiNER.from_pretrained(
            model_name_str, words_splitter=word_splitter
        )
        print("Loaded GLiNER model on CPU")

    return gliner_large_model, word_splitter


def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))


def sliding_window_over_paragraph(text, n_words=128, sobreposicao=12, fillvalue=""):
    windows = [
        " ".join(x).strip()
        for x in sliding_window(
            text.split(" "),
            size=n_words,
            step=n_words - sobreposicao,
            fillvalue=fillvalue,
        )
    ]
    return windows


def join_entity_predictions(entity_dicts: List[dict]) -> dict:
    combined = None

    for entities in entity_dicts:
        if combined is None:
            combined = entities
        else:
            for ent_key in entities.keys():
                new_values = entities[ent_key]
                previous = []
                if ent_key in combined:
                    previous = combined[ent_key]
                value_points = {}
                for value, points in new_values + previous:
                    if value in value_points:
                        if points > value_points[value]:
                            value_points[value] = points
                    else:
                        value_points[value] = points
                updated_values = [(key, p) for key, p in value_points.items()]
                updated_values.sort(key=lambda xy: xy[1], reverse=True)
                combined[ent_key] = updated_values
    return combined


def extract_labels(
    transcript: str,
    gliner_model,
    word_splitter,
    labels: List[str] = ner_labels["emergency_gliclass"],
    top_to_keep=3,
    not_keep_top=set(),
) -> dict:
    req_start = time.time()
    entities = gliner_model.predict_entities(
        transcript, labels, threshold=0.5, flat_ner=False
    )
    infer_finish = time.time()
    req_duration = infer_finish - req_start

    entities_dict = {label: [] for label in labels if label not in sinonimos}
    for entity in entities:
        # print(entity)
        new_val = entity["text"].strip().replace("\n", " ").replace("\r", "")
        if new_val != "":
            ent_label = entity["label"]
            if ent_label in sinonimos:
                ent_label = sinonimos[ent_label]
            entities_dict[ent_label].append((new_val, round(entity["score"] * 100, 3)))

    for key in entities_dict:
        non_redundant_lower = []
        non_redundant = []
        for val, score in entities_dict[key]:
            lower = val.lower()

            if lower not in non_redundant_lower:
                non_redundant_lower.append(lower)
                non_redundant.append((val, score))
        entities_dict[key] = non_redundant
    # print(f'entities_dict 1: {json.dumps(entities_dict, indent=2)}')
    for label in values_to_ignore.keys():
        if label in entities_dict:
            not_ignore = []
            for value, score in entities_dict[label]:
                if value.lower() not in values_to_ignore[label]:
                    not_ignore.append([value, score])
            entities_dict[label] = not_ignore

    for label in entities_dict.keys():
        entities_dict[label].sort(key=lambda x: x[1], reverse=True)
        if label not in not_keep_top and len(entities_dict[label]) > top_to_keep:
            entities_dict[label] = entities_dict[label][
                :top_to_keep
            ]  # Keep only the top 3 values for each label
    # print(f'entities_dict 2: {json.dumps(entities_dict, indent=2)}')

    # contando comprimento do input e output
    token_generator_input = word_splitter.splitter(transcript)
    n_input_tokens = len(list(token_generator_input))
    output_str = json.dumps(entities_dict, ensure_ascii=False)
    token_generator_output = word_splitter.splitter(output_str)
    n_output_tokens = len(list(token_generator_output))

    post_processing_duration = time.time() - infer_finish

    meta = {
        "processing_time": req_duration,
        "no_gpu_time": post_processing_duration,
        "input_tokens": n_input_tokens,
        "output_tokens": n_output_tokens,
        "model_name": "gliner",
    }

    return entities_dict, meta


def extract_labels_parallel(
    transcripts: str,
    gliner_model,
    word_splitter,
    labels: List[str] = ner_labels["emergency_gliclass"],
    top_to_keep=3,
    not_keep_top=set(),
    batch_size=5,
    max_tokens=128,
) -> dict:
    process_start_time = time.time()
    original_index = []
    transcripts_split = []
    n_input_tokens_start = []
    result_dicts = {}
    # print("Original transcripts:", len(transcripts))
    for index, t in enumerate(transcripts):
        token_generator_input = word_splitter.splitter(t)
        n_input_tokens = len(list(token_generator_input))
        result_dicts[index] = []
        if n_input_tokens <= max_tokens:
            original_index.append(index)
            transcripts_split.append(t)
            n_input_tokens_start.append(n_input_tokens)
        else:
            windows = sliding_window_over_paragraph(
                t, n_words=max_tokens, sobreposicao=5, fillvalue=""
            )
            for w in windows:
                original_index.append(index)
                transcripts_split.append(w)
                token_generator_input = word_splitter.splitter(w)
                n_input_tokens = len(list(token_generator_input))
                n_input_tokens_start.append(n_input_tokens)

    # print("Splited transcripts:", len(transcripts_split))
    # print(original_index)
    req_start = time.time()
    entities_lists = gliner_model.run(
        transcripts_split, labels, threshold=0.55, batch_size=batch_size
    )
    infer_finish = time.time()
    req_duration = infer_finish - req_start
    # print("entities_lists:", len(entities_lists))

    results_final = []
    metas = []
    for entities, n_input_tokens in zip(entities_lists, n_input_tokens_start):
        entities_dict = {label: [] for label in labels if label not in sinonimos}
        for entity in entities:
            # print(entity)
            new_val = entity["text"].strip().replace("\n", " ").replace("\r", "")
            if new_val != "" and len(new_val) <= 300:
                ent_label = entity["label"]
                if ent_label in sinonimos:
                    ent_label = sinonimos[ent_label]
                entities_dict[ent_label].append(
                    (new_val, round(entity["score"] * 100, 3))
                )

        for key in entities_dict:
            non_redundant_lower = []
            non_redundant = []
            for val, score in entities_dict[key]:
                lower = val.lower()

                if lower not in non_redundant_lower:
                    non_redundant_lower.append(lower)
                    non_redundant.append((val, score))
            entities_dict[key] = non_redundant
        # print(f'entities_dict 1: {json.dumps(entities_dict, indent=2)}')
        for label in values_to_ignore.keys():
            if label in entities_dict:
                not_ignore = []
                for value, score in entities_dict[label]:
                    if value.lower() not in values_to_ignore[label]:
                        not_ignore.append([value, score])
                entities_dict[label] = not_ignore

        for label in entities_dict.keys():
            entities_dict[label].sort(key=lambda x: x[1], reverse=True)
            if label not in not_keep_top and len(entities_dict[label]) > top_to_keep:
                entities_dict[label] = entities_dict[label][
                    :top_to_keep
                ]  # Keep only the top 3 values for each label
        # print(f'entities_dict 2: {json.dumps(entities_dict, indent=2)}')

        # contando comprimento do output
        output_str = json.dumps(entities_dict, ensure_ascii=False)
        token_generator_output = word_splitter.splitter(output_str)
        n_output_tokens = len(list(token_generator_output))

        meta = {
            "input_tokens": n_input_tokens,
            "output_tokens": n_output_tokens,
            "model_name": "gliner",
        }

        metas.append(meta)
        results_final.append(entities_dict)

    metas2 = []
    results_final2 = []

    for index, r, m in zip(original_index, results_final, metas):
        result_dicts[index].append((r, m))
    """for index, r, m in zip(original_index, results_final, metas):
        print(f'Transcript {index}:')
        print('\t', r)
        print('\t', m)
    print('Results per index:')
    for index, rs in result_dicts.items():
        print(f'Index {index}: {len(rs)}')"""

    total_duration = time.time() - process_start_time
    post_processing_duration = total_duration - req_duration

    for n in range(len(transcripts)):
        all_results = result_dicts[n]
        w = len(all_results) / len(results_final)
        if len(all_results) == 1:
            r, m = all_results[0]
            m["no_gpu_time"] = post_processing_duration * w
            m["processing_time"] = req_duration * w
            metas2.append(m)
            results_final2.append(r)
        else:
            rs = [r for r, m in all_results]
            ms = [m for r, m in all_results]
            r2 = join_entity_predictions(rs)

            m2 = {
                "no_gpu_time": post_processing_duration * w,
                "processing_time": req_duration * w,
                "input_tokens": sum([x["input_tokens"] for x in ms]),
                "output_tokens": sum([x["output_tokens"] for x in ms]),
                "model_name": ms[0]["model_name"],
            }

            metas2.append(m2)
            results_final2.append(r2)

    return results_final2, metas2


if __name__ == "__main__":
    g_model, word_splitter = load_gliner_cuda(
        "knowledgator/gliner-x-large", use_cuda=False
    )
    print("compiling")
    # m.compile()

    texts = [
        "Meu nome é pitágoras e moro em Natal, tenho 30 anos.",
        "Meu nome é Rafaela e moro no Tocantins, tenho 27 anos.",
        "Meu nome é Robson e moro em Curitiba, tenho 38 anos e sou analista",
    ]
    texts = texts + texts + texts
    shuffle(texts)
    labels = ner_labels["emergency_gliclass"]

    """print('Warmup 1')
    m.run([texts[0]], labels=labels)
    print('Warmup 2')
    m.run([texts[4], texts[5]], labels=labels)
    print('Warmup 3')
    #m.run([texts[1]], labels=labels)
    #m.run([texts[2]], labels=labels)"""

    print("Batch...")
    speed_ups = []
    max_tokens_values = [75, 100, 110, 125, 135, 150, 165, 175, 200, 230]
    batch_sizes = [2, 3]
    for bs in batch_sizes:

        print("Sequential...")
        no_batch_start = time.time()
        results = {}
        for t in texts:
            r1, m1 = extract_labels(t, g_model, word_splitter, labels=labels)
            results[t] = {"entities": r1, "meta": m1}
            # print(m1['processing_time'])
        # print(json.dumps(results, indent=2, ensure_ascii=False))
        no_batch_end = time.time()
        sequential_duration = no_batch_end - no_batch_start
        print("sequencial", sequential_duration)

        for mt in max_tokens_values:
            batch_start = time.time()
            results2 = {}
            r_vec, meta_vec = extract_labels_parallel(
                texts,
                g_model,
                word_splitter,
                labels=labels,
                batch_size=bs,
                max_tokens=mt,
            )
            for t, r, m in zip(texts, r_vec, meta_vec):
                results2[t] = {"entities": r, "meta": m}
                print(m["processing_time"])

            # print(json.dumps(results2, indent=2, ensure_ascii=False))
            batch_end = time.time()
            delta = batch_end - batch_start
            s = sequential_duration - delta
            result = [mt, bs, s]
            print(result)
            speed_ups.append(result)

    print("batch done")
    for max_tokens, bts, speed_diff in speed_ups:
        print(max_tokens, bts, speed_diff)

    del m
    del word_splitter

    import torch

    torch.cuda.empty_cache()
