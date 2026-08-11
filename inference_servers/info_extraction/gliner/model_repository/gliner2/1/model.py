import time
import json
import sys
from typing import List
from collections import deque
from itertools import islice
from typing import List
import numpy as np
import torch
from gliner2 import GLiNER2
from gliner.data_processing.tokenizer import WordsSplitter
import triton_python_backend_utils as pb_utils


def load_gliner_cuda(model_name_str):

    try:
        gliner_large_model = GLiNER2.from_pretrained(model_name_str).to("cuda")
        print("Loaded GLiNER model")
    except Exception as e:
        print(f"Error loading GLiNER model with CUDA: {e}")
        raise (e)

    return gliner_large_model


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


def sliding_window_over_paragraph(text, n_words=160, sobreposicao=12, fillvalue=""):
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


class TritonPythonModel:
    """
    Classe do modelo Python para o Triton.
    """

    def initialize(self, args):
        """
        Carrega o modelo da Hugging Face.
        """
        self.model_id = "fastino/gliner2-multi-v1"

        print(f"Carregando modelo {self.model_id}...")
        self.model = load_gliner_cuda(self.model_id)

        word_splitter_name = "spacy"
        self.word_splitter = WordsSplitter(splitter_type=word_splitter_name)

    def find_labels(self, labels_str, transcript):
        try:
            labels_list = json.loads(labels_str)
        except json.JSONDecodeError as err:
            labels_list = labels_str.split(",")
            labels_list = [
                label.strip() for label in labels_list if label.strip() != ""
            ]

        schema = self.model.create_schema().entities(labels_list, dtype="list")

        req_start = time.time()
        entities = self.model.extract_entities(
            transcript, schema, threshold=0.6, include_confidence=True
        )
        infer_finish = time.time()
        req_duration = infer_finish - req_start

        entities_dict = {}
        entities = entities["entities"]
        for ent_label, entity_values in entities.items():
            for entity in entity_values:
                new_val = entity["text"].strip().replace("\n", " ").replace("\r", "")
                if new_val != "":
                    if not ent_label in entities_dict:
                        entities_dict[ent_label] = []
                    entities_dict[ent_label].append(
                        (new_val, round(entity["confidence"] * 100, 3))
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

        return entities_dict, req_duration

    def classify_by_schema(self, classification_schema_str, transcript):
        print(classification_schema_str, file=sys.stderr)
        try:
            classification_schema = json.loads(classification_schema_str)
            print(classification_schema, file=sys.stderr)
        except json.JSONDecodeError as err:
            raise Exception("Invalid classification schema: " + str(err))

        try:
            req_start = time.time()
            new_entities = self.model.extract_json(
                transcript,
                classification_schema,
                threshold=0.5,
                include_confidence=False,
                include_spans=False,
            )
            infer_finish = time.time()
            req_duration = infer_finish - req_start

            return new_entities, req_duration
        except Exception as err:
            raise Exception(
                "Error classifying transcript: "
                + str(err)
                + "\nWith clf schema:\n"
                + str(classification_schema)
            )

    def execute(self, requests):
        """
        Chamado para cada BATCH de inferência.

        Processa cada request individualmente (mas dentro do batch).
        """
        responses = []

        # O Triton agrupa N requests (definido pelo dynamic_batching).
        # Iteramos por cada request no batch.
        for request in requests:
            try:
                proc_start = time.time()
                # 2. EXTRAIR OS INPUTS DO TRITON
                # Extrai o PROMPT (texto)
                input_text_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                # Acessa o valor do array NumPy e decodifica.
                # Para shape [1, 1], o valor está em [0, 0].
                transcript = input_text_tensor.as_numpy()[0, 0].decode("utf-8")

                # Extrai a LABEL_LIST (string JSON)
                labels_tensor = pb_utils.get_input_tensor_by_name(request, "LABEL_LIST")
                # Acessa o valor do array NumPy e decodifica.
                labels_str = labels_tensor.as_numpy()[0, 0].decode("utf-8")

                # Extrai a CLASSIFICATION_SCHEMA (string JSON)
                classification_schema_str = pb_utils.get_input_tensor_by_name(
                    request, "CLASSIFICATION_SCHEMA"
                )
                # Acessa o valor do array NumPy e decodifica.
                classification_schema_str = (
                    classification_schema_str.as_numpy()[0, 0]
                    .decode("utf-8")
                    .replace('"', '"')
                )

                entities_dict, req_duration1 = self.find_labels(labels_str, transcript)
                new_entities, req_duration2 = self.classify_by_schema(
                    classification_schema_str, transcript
                )

                for k, v in new_entities.items():
                    entities_dict[k] = v

                # Contando comprimento do input e output
                token_generator_input = self.word_splitter.splitter(transcript)
                n_input_tokens = len(list(token_generator_input))
                output_str = json.dumps(entities_dict, ensure_ascii=False)
                token_generator_output = self.word_splitter.splitter(output_str)
                n_output_tokens = len(list(token_generator_output))

                total_duration = time.time() - proc_start
                post_processing_duration = (
                    total_duration - req_duration1 - req_duration2
                )
                meta = {
                    "processing_time": req_duration1 + req_duration2,
                    "no_gpu_time": post_processing_duration,
                    "input_tokens": n_input_tokens,
                    "output_tokens": n_output_tokens,
                    "model_name": "gliner",
                }
                meta_str = json.dumps(meta, ensure_ascii=False)

                # CRIAR OS TENSORES DE SAÍDA
                # Converte as strings de saída de volta para tensores Triton
                output_entities_tensor = pb_utils.Tensor(
                    "ENTITIES_JSON", np.array([output_str], dtype=np.object_)
                )
                output_meta_tensor = pb_utils.Tensor(
                    "META_INFO", np.array([meta_str], dtype=np.object_)
                )

                # CRIAR A RESPOSTA
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_entities_tensor, output_meta_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                # Se algo der errado (ex: JSON mal formatado), retorna um erro
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)

        # 6. RETORNAR A LISTA DE RESPOSTAS
        return responses

    def finalize(self):
        """
        Chamado quando o modelo é descarregado.
        """
        print("Limpando o modelo...")
        self.model = None
        torch.cuda.empty_cache()
        print("Finalizado.")
