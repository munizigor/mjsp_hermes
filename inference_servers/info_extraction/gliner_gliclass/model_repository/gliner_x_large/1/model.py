import time
import json
import sys
from typing import List
from collections import deque
from itertools import islice
from typing import List
import numpy as np
from gliner import GLiNER
from gliner.data_processing.tokenizer import WordsSplitter
from transformers import AutoTokenizer, pipeline
import triton_python_backend_utils as pb_utils
from tqdm import tqdm
import torch

'''MAX_TOKENS = 128
MAX_TOKENS = 384
MAX_TOKENS = 340'''
MAX_TOKENS = 520

MAX_WINDOW_BATCH = 8

answer_to_value = {
    "Certamente Sim": 1.0,
    "Sim": 0.95,
    "Provavelmente Sim": 0.65,
    "Não sei": 0.5,
    "Provavelmente Não": 0.35,
    "Não": 0.05,
    "Certamente Não": 0.0,
}

def load_gliner_cuda(model_name_str):
    word_splitter_name = 'spacy'
    word_splitter = WordsSplitter(splitter_type=word_splitter_name)
    
    try:
        gliner_large_model = GLiNER.from_pretrained(model_name_str).to('cuda')
        print('Loaded GLiNER model')
    except Exception as e:
        print(f"Error loading GLiNER model with CUDA: {e}")
        raise(e)
    
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
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

def sliding_window_over_paragraph(text, n_words=160, sobreposicao=12, fillvalue=''):
    windows = [' '.join(x).strip()
               for x in sliding_window(text.split(' '), 
                    size=n_words, step=n_words-sobreposicao, fillvalue=fillvalue)]
    return windows

def join_entity_predictions(entity_dicts: List[dict]) -> dict:
    combined = None

    for entities in entity_dicts:
        if combined is None:
            combined = entities
        else:
            for ent_key in entities.keys():
                new_values = entities[ent_key]
                if type(new_values) == list:
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
                elif type(new_values) in [int, float]:
                    if ent_key in combined:
                        old_value = combined[ent_key]
                        if old_value != old_value or old_value is None:
                            old_value = -1
                    else:
                        old_value = -1
                    if new_values > old_value:
                        combined[ent_key] = new_values

    return combined

def join_entity_predictions_mean(entity_dicts: List[dict]) -> dict:
    combined = {}
    list_accumulators = {}
    scalar_accumulators = {}

    for entities in entity_dicts:
        for ent_key, new_values in entities.items():
            if type(new_values) == list:
                if ent_key not in list_accumulators:
                    list_accumulators[ent_key] = {}
                for value, points in new_values:
                    if value not in list_accumulators[ent_key]:
                        list_accumulators[ent_key][value] = []
                    list_accumulators[ent_key][value].append(points)
            elif type(new_values) in [int, float]:
                if ent_key not in scalar_accumulators:
                    scalar_accumulators[ent_key] = []
                scalar_accumulators[ent_key].append(new_values)

    for ent_key, values_map in list_accumulators.items():
        updated_values = []
        for value, points_list in values_map.items():
            mean_points = sum(points_list) / len(points_list)
            updated_values.append((value, mean_points))
        updated_values.sort(key=lambda xy: xy[1], reverse=True)
        combined[ent_key] = updated_values

    for ent_key, values_list in scalar_accumulators.items():
        if values_list:
            combined[ent_key] = sum(values_list) / len(values_list)

    return combined

def prepare_warmup():
    text1 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
        Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. """
    text2 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, 
        perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, 
        moro na cidade de São Paulo. Ela não tem risco de morte! Nenhum!"""
    text3 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
    Ela está inconsciente. Foi o marido que bateu nela! Aquele covarde! Meu nome é Maria Oliveira. 
    Meu telefone é 99999-8888, moro na cidade de São Paulo. Ela não tem risco de morte! Nenhum!"""
    text4 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
    Foi o marido que bateu nela! Aquele covarde! Meu nome é Maria Oliveira. 
    Meu telefone é 99999-8888, moro na cidade de São Paulo. Sim, o marido violento espancou ela!."""
    text5 = """Socorro, preciso de uma viatura rápido! Estão atirando em mim! Quando? Agora mesmo! 
        Ele tem uma pistola! Meu nome é Maria Oliveira. Meu telefone é 99999-3888, moro na cidade de Natal. 
        Repito: o Claudio começou a atirar em mim com uma pistola, do nada"""
    text6 = """Quero avisar de um risco de tumulto. Meu nome é Maria Oliveira. Meu telefone é 99999-3888, 
    moro na cidade de Natal. Há uma manifestação em frente ao shopping. Há muitas pessoas e veículos.
    Parece que vai começar uma briga. """
    text7 = """Alô? É da ambulancia? Meu pai levou um tiro na rua. Ele chegou em casa sangrando e disse que foi tiro. 
    Não, ele não sabe quem foi. Aconteceu a mais ou menos 1 hora. Meu nome é Pedro. Meu telefone é 99999-3888, 
    moro na cidade de Natal. Meu pai é o Claudio. ele disse que foi na rua salgado filho, número 123, perto do shopping.
    Mas nós moramos em outro lugar. O endereço da nossa casa? É rua capitão mor golveia, número 450.
    """
    all_texts = [text1, text2, text3, text4, text5, text6, text7]

    fatoOcorrendoNesteMomento = "Fato Ocorrendo Neste Momento ?"
    autorDoFatoNoLocal = "Autor Do Fato No Local ?"
    autorDoFatoArmado = "Autor Do Fato Armado ?"
    feridosComRiscoDeMorte = "Feridos Com Risco de Morte ?"
    riscoDeTumulto = "Risco De Tumulto ?"
    leiMariaPenha = "Lei Maria da Penha ?"
    violenciaDomestica = "Violência Doméstica ?"

    clfnames = [
        fatoOcorrendoNesteMomento,
        autorDoFatoNoLocal,
        autorDoFatoArmado,
        feridosComRiscoDeMorte,
        riscoDeTumulto,
        leiMariaPenha,
        violenciaDomestica
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
        violenciaDomestica: {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
            "sim": "Sim",
            "nao": "Não",
        }
    }

    full_schema_dict = {
        "entities": {
            "rua_ou_logradouro": "Nome de rua, logradouro ou avenida",
            "rua": "Nome de rua",
            "bairro": "Nome de bairro",
            "municipio": "Nome de município",
            "cidade": "Nome de cidade",
            "ponto_de_referencia": "Nome de ponto de referência do endereço",
            "nome_do_solicitante": "Nome do solicitante (pessoa que está fazendo o chamado)",
            "pessoa": "Nome de pessoa ou participante",
            "numero": "Número do endereço",
            "street_number": "Street Number",
            "number": "Número",
            "complemento": "Complemento do endereço (Exemplos: apt 301, casa A, etc)",
            "endereço_complemento": "Número ou código da casa, apartamento ou loja naquele endereço",
        },
        "boolean": {"natureza_da_ocorrencia": clfs_bool},
    }

    return all_texts, clfnames, full_schema_dict

class TritonPythonModel:
    """
    Classe do modelo Python para o Triton.
    """

    DEFAULT_GLINERX = "knowledgator/gliner-x-large"
    DEFAULT_GLINERCLASS = "knowledgator/gliclass-large-v3.0"

    def initialize(self, args):
        """
        Carrega o modelo da Hugging Face.
        """
        if "model_id" in args:
            self.model_id = args["model_id"]
        else:
            self.model_id = TritonPythonModel.DEFAULT_GLINERX

        if "clf_model_id" in args:
            self.clf_model_id = args["clf_model_id"]
        else:
            self.clf_model_id = TritonPythonModel.DEFAULT_GLINERCLASS

        #print(f"Carregando modelo {self.model_id}...")
        self.model, self.word_splitter = load_gliner_cuda(self.model_id)

        print(f"Carregando modelo {self.clf_model_id}...")
        if "gliclass" in self.clf_model_id:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            clf_model = GLiClassModel.from_pretrained(self.clf_model_id)
            clf_tokenizer = AutoTokenizer.from_pretrained(self.clf_model_id)
            self.clf_pipeline = ZeroShotClassificationPipeline(
                clf_model, clf_tokenizer, 
                classification_type='multi-label', device='cuda:0',
                progress_bar=False
            )
            self.is_gliclass = True
        else:
            self.clf_pipeline = pipeline(
                "zero-shot-classification",
                model=self.clf_model_id,
                device='cuda:0',
                multi_label=True,
                trust_remote_code=True
            )
            self.is_gliclass = False
        #self.clf_pipeline.set_progress_bar_config(disable=True)

        self.warmup_model()

    def infer_only(self, transcripts: List[str], classification_schema_str, batch_size=MAX_WINDOW_BATCH):
        classification_schema_dict = json.loads(classification_schema_str)
        transcript_parts = []
        transcript_ids = []

        for idx, transcript in enumerate(transcripts):
            windows = sliding_window_over_paragraph(transcript, n_words=MAX_TOKENS, sobreposicao=5)
            transcript_parts.extend(windows)
            transcript_ids.extend([idx]*len(windows))

        if "entities" in classification_schema_dict:
            entities_schema = classification_schema_dict["entities"]
            labels_list = list(entities_schema.keys()) if type(entities_schema) == dict else entities_schema
            req_start = time.time()
            labels_raw = self.model.inference(transcript_parts, labels_list, 
                threshold=0.5, flat_ner=False, batch_size=batch_size)
            infer_finish = time.time()
            req_duration1 = infer_finish - req_start
        else:
            req_duration1 = 0
            labels_raw = [{} for _ in transcript_parts]
            labels_list = None
        
        clf_results = []

        if "boolean" in classification_schema_dict:
            classification_schema_raw = classification_schema_dict["boolean"]

            #classification_schema_1 = {}
            multilabel_schemas = []
            for key, value in classification_schema_raw.items():
                entity_names = []
                for entity_name, info in value.items():
                    #classification_schema_1[entity_name] = info['answers']
                    entity_names.append(entity_name)
                multilabel_schemas.append(entity_names)
            
            req_duration2 = 0.0
            name_translator = {}

            for ml_categories in multilabel_schemas:
                no_interrogation_labels = [n.replace(' ?', '') for n in ml_categories]
                for n, n_no_inter in zip(ml_categories, no_interrogation_labels):
                    name_translator[n_no_inter] = n
                examples = []
                start_clf = time.time()
                if self.is_gliclass == False:
                    results = self.clf_pipeline(transcript_parts, no_interrogation_labels,
                        multi_label=True, batch_size=batch_size)
                elif len(examples) == 0:
                    results = self.clf_pipeline(transcript_parts, no_interrogation_labels, 
                        threshold=0.001, batch_size=batch_size)
                else:
                    results = self.clf_pipeline(transcript_parts, no_interrogation_labels, 
                        threshold=0.001, rac_examples=examples, batch_size=batch_size)
                finish_clf = time.time()
                clf_t = finish_clf - start_clf
                req_duration2 += clf_t
                clf_results.append(results)
        else:
            req_duration2 = 0
            name_translator = {}
            multilabel_schemas = []
        long_return = (transcript_ids, labels_raw, clf_results, 
            req_duration1, req_duration2, 
            multilabel_schemas, labels_list, name_translator)
        return long_return
    
    def post_process_labels(self, entities, labels_list):
        entities_dict = {label: [] for label in labels_list}
        for entity in entities:
            new_val = entity["text"].strip().replace('\n', ' ').replace('\r', '')
            if new_val != '':
                ent_label = entity["label"]
                entities_dict[ent_label].append((new_val, round(entity['score']*100, 3)))

        for key in entities_dict:
            non_redundant_lower = []
            non_redundant = []
            for val, score in entities_dict[key]:
                lower = val.lower()
                if lower not in non_redundant_lower:
                    non_redundant_lower.append(lower)
                    non_redundant.append((val, score))
            entities_dict[key] = non_redundant

        return entities_dict
    
    def post_process_clf(self, results, ml_categories, name_translator):
        if self.is_gliclass:
            new_entities_raw = []
            #print(name_translator)
            for r in results:
                #print(r)
                for info_score in r:
                    #print(info_score)
                    obj = {'label': name_translator[info_score['label']], 
                        'confidence': info_score['score']}
                    new_entities_raw.append(obj)
        else:
            new_entities_raw = []
            for r in results:
                new_entities = [{'label': name_translator[label], 'confidence': score} 
                    for score, label in zip(r['scores'], r['labels'])]
                new_entities_raw.extend(new_entities)
        
        categories_found = [label_info["label"] for label_info in new_entities_raw]
        not_found = [label for label in ml_categories if label not in categories_found]
        for label in not_found:
            new_entities_raw.append({"label": label, "confidence": 0.0})

        results_final = {}
        for label_info in new_entities_raw:
            label_name = label_info["label"]
            label_conf = label_info["confidence"]
            if label_name not in results_final:
                results_final[label_name] = []
            results_final[label_name].append(label_conf)
        
        results_media = {}
        for label_name, label_confs in results_final.items():
            results_media[label_name] = sum(label_confs) / len(label_confs)
        results_sorted = sorted([(k, results_media[k]) for k,v in results_final.items()], key=lambda x: results_media[x[0]], reverse=True)
        new_entities = {}
        for label, conf in results_sorted:
            new_entities[label] = conf
        
        return new_entities

    def multi_inference(self, transcripts, raw_schema_str):
        infers = self.infer_only(transcripts, raw_schema_str)
        (transcript_ids, labels_raw, clf_results, 
            req_duration1, req_duration2, 
            multilabel_schemas, labels_list, name_translator) = infers
        
        '''print("transcript_ids", len(transcript_ids), transcript_ids)
        print("labels_raw", len(labels_raw), labels_raw)
        print("clf_results", len(clf_results))
        print("clf_results_local", len(clf_results[0]), clf_results[0][0])'''
        labels_processed = [self.post_process_labels(raw_info, labels_list) 
            for raw_info in labels_raw]
        #print("labels_processed", len(labels_processed), labels_processed[0])
        clfs_processed = []

        for clf_schema_n, ml_categories in enumerate(multilabel_schemas):
            clf_results_local = clf_results[clf_schema_n]
            # Process each text window individually by wrapping 'res' in a list
            processed_windows = [
                self.post_process_clf([res], ml_categories, name_translator) 
                for res in clf_results_local
            ]
            clfs_processed.append(processed_windows)
        
        ids_unique = set(transcript_ids)
        #print("ids_unique", len(ids_unique), ids_unique)
        #print("clfs_processed", len(clfs_processed), clfs_processed[0])
        #print("clf_schema_results", len(clfs_processed[0]), clfs_processed[0][0])

        joined_results = []
        for uniq_id in sorted(ids_unique):
            #print("uniq_id", uniq_id)
            transcript_part_idx = [pos for pos, idx in enumerate(transcript_ids) 
                if idx == uniq_id]
            #print("transcript_part_idx", len(transcript_part_idx), transcript_part_idx)
            
            entities_dicts = [{} for _ in range(len(transcript_part_idx))]
            labels_processed_local = [labels_processed[pos] for pos in transcript_part_idx]
            clfs_processed_local = []
            for clf_schema_results in clfs_processed:
                if len(clf_schema_results) > 0:
                    local_clfs = [clf_schema_results[pos] for pos in transcript_part_idx]
                    clfs_processed_local.append(local_clfs)
                else:
                    clfs_processed_local.append({})
            
            for window_id in range(len(labels_processed_local)):
                for label, entities in labels_processed_local[window_id].items():
                    entities_dicts[window_id][label] = entities
                for clf_schema_results in clfs_processed_local:
                    if len(clf_schema_results) > 0:
                        local_clfs = clf_schema_results[window_id]
                        for label, conf in local_clfs.items():
                            entities_dicts[window_id][label] = conf
            
            joined_entities = join_entity_predictions_mean(entities_dicts)
            joined_results.append(joined_entities)
        
        return joined_results, req_duration1, req_duration2
        
    def execute(self, requests):
        """
        Chamado para cada BATCH de inferência.

        Processa cada request individualmente (mas dentro do batch).
        """
        all_start = time.time()

        # O Triton agrupa N requests (definido pelo dynamic_batching).
        # Iteramos por cada request no batch.
        # Salvar dicionário que agrupa transcript de acordo com labels_str

        transcript_dict = {}
        transcript_to_pos = {}
        responses = [None for _ in range(len(requests))]
        print('-----')
        print('Número de requisições: ', len(requests))
        print('-----')
        for pos, request in enumerate(requests):
            try:
                input_text_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                # Acessa o valor do array NumPy e decodifica.
                # Para shape [1, 1], o valor está em [0, 0].
                transcript = input_text_tensor.as_numpy()[0, 0].decode("utf-8")
                # Extrai a LABEL_LIST (string JSON)
                labels_tensor = pb_utils.get_input_tensor_by_name(request, "LABEL_LIST")
                # Acessa o valor do array NumPy e decodifica.
                labels_str = labels_tensor.as_numpy()[0, 0].decode("utf-8").replace('\"', '"')

                if not labels_str in transcript_dict:
                    transcript_dict[labels_str] = []
                transcript_dict[labels_str].append(transcript)
                if transcript in transcript_to_pos:
                    transcript_to_pos[transcript].append(pos)
                else:
                    transcript_to_pos[transcript] = [pos]
            except Exception as e:
                print(f"Erro ao processar request {pos}: {str(e)}")
                print(e)
        
        results_correct_order = [{} for _ in range(len(requests))]
        for labels_str, transcripts in transcript_dict.items():
            try:
                infers, req_duration1, req_duration2 = self.multi_inference(transcripts, 
                    labels_str)
                d1_per_transcript = req_duration1 / len(transcripts)
                d2_per_transcript = req_duration2 / len(transcripts)
                
                for transcript, infer in zip(transcripts, infers):
                    n_input_tokens = len(transcript.split())
                    output_str = json.dumps(infer, ensure_ascii=False)
                    n_output_tokens = len(output_str.split())
                    positions = transcript_to_pos[transcript]
                    for pos in positions:
                        results_correct_order[pos] = {'ENTITIES_JSON_RAW': infer,
                            'META_INFO_RAW': {
                                'processing_time': d1_per_transcript + d2_per_transcript,
                                'no_gpu_time': 0,
                                "input_tokens": n_input_tokens,
                            "output_tokens": n_output_tokens,
                            "model_name": "gliner_gliclass"
                        } 
                    }
                
                
            except Exception as e:
                print(f"Erro ao processar schema '{labels_str}' e transcripts:", file=sys.stderr)
                for t in transcripts:
                    print(t, file=sys.stderr)
                print('Erro', file=sys.stderr)
                print(e, file=sys.stderr)
                for transcript, infer in zip(transcripts, infers):
                    positions = transcript_to_pos[transcript]
                    for pos in positions:
                        error_response = pb_utils.InferenceResponse(
                            output_tensors=[], error=pb_utils.TritonError(str(e))
                        )
                        responses[pos] = error_response

                quit(1)

        #Look for failed requests:
        for pos, result in enumerate(results_correct_order):
            if 'META_INFO_RAW' not in result:
                print(f"Request {pos} failed: {result}", file=sys.stderr)
                quit(1)
                    

        infer_time_sum = 0.0
        for result in results_correct_order:
            if 'META_INFO_RAW' in result:
                infer_time_sum += result['META_INFO_RAW']['processing_time']
        
        all_end = time.time()
        all_time = all_end - all_start

        cpu_time_by_request = (all_time - infer_time_sum) / len(results_correct_order)

        for pos, result in enumerate(results_correct_order):
            if responses[pos] is not None:
                continue

            result['META_INFO_RAW']['no_gpu_time'] = cpu_time_by_request
            
            output_str = json.dumps(result['ENTITIES_JSON_RAW'], ensure_ascii=False)
            meta_str = json.dumps(result['META_INFO_RAW'], ensure_ascii=False)

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
            responses[pos] = inference_response
            
        return responses

    def finalize(self):
        """
        Chamado quando o modelo é descarregado.
        """
        print("Limpando o modelo...")
        self.model = None
        torch.cuda.empty_cache()
        print("Finalizado.")

    def warmup_model(self):
        all_texts, clfnames, full_schema_dict = prepare_warmup()
        labels_str = json.dumps(full_schema_dict)

        print('Warming up model...')
        infers, req_duration1, req_duration2 = self.multi_inference(
            all_texts, labels_str)
        print('Warming up model... Done')
        for infer, text in zip(infers, all_texts):
            print(text)
            print(infer)
            print('---')
