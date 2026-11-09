import json
import os
from time import time
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from model_runners.templates import naturezas_vec

naturezas_dump_path = "/datasets/naturezas_cache_vllm.json"
if not os.path.exists(naturezas_dump_path):
    naturezas_dump_path = "/tmp/naturezas_cache_vllm.json"
paraphrase_multil_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class SBertOutput:
    def __init__(self, embedding):
        self.embedding = embedding


class SBertResult:
    def __init__(self, embedding, prompt_token_ids):
        self.outputs = SBertOutput(embedding)
        self.prompt_token_ids = prompt_token_ids

class SentenceTransformersLLM:
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __init__(self, model_name):
        print('Loading', model_name)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print('Loaded', model_name)

    def embed(self, docs):
        encoded_input = self.tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
        tokens = encoded_input['input_ids']
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = SentenceTransformersLLM.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = [vector.tolist() for vector in sentence_embeddings]
        outputs = []
        for s, emb, t in zip(docs, embeddings, tokens):
            outputs.append(SBertResult(emb, t))
        assert len(embeddings) == len(docs)
        assert len(tokens) == len(docs)
        return outputs


class EmbeddingStore:
    backends = ['cuda', 'cpu']

    @staticmethod
    def _parse_raw_cache(data):
        """Converte dados JSON (formato legado mono-modelo ou novo multi-modelo) em dicionário de cache."""
        cache = {}
        if 'models' in data and isinstance(data['models'], dict):
            raw_models = data['models']
        elif 'model_name' in data and 'docs' in data and 'embs' in data:
            raw_models = {
                data['model_name']: {
                    'docs': data['docs'],
                    'n_tokens': data.get('n_tokens', []),
                    'embs': data['embs']
                }
            }
        else:
            raw_models = {}

        for m_name, m_data in raw_models.items():
            docs = []
            embs = []
            n_tokens = []
            seen_docs = set()
            raw_docs = m_data.get('docs', [])
            raw_embs = m_data.get('embs', [])
            raw_tokens = m_data.get('n_tokens', [0] * len(raw_docs))
            for doc, emb, n_tok in zip(raw_docs, raw_embs, raw_tokens):
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    docs.append(doc)
                    embs.append(np.asarray(emb))
                    n_tokens.append(n_tok)
            cache[m_name] = {
                'docs': docs,
                'embs': embs,
                'n_tokens': n_tokens
            }
        return cache

    def load_store(path, model_name, backend='cuda'):
        assert backend in EmbeddingStore.backends
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(err)
            print(f"Re-starting embedding cache with clean data")
            data = {"model_name": model_name, "models": {model_name: {"docs": [], "embs": [], "n_tokens": []}}}
            json.dump(data, open(path, 'w'), indent=2)

        active_model = model_name
        store = EmbeddingStore(model_name=active_model, backend=backend)
        store.cache = EmbeddingStore._parse_raw_cache(data)
        store._ensure_model_entry(active_model)
        return store

    def __init__(self, model_name: str, backend='cuda'):
        assert backend in EmbeddingStore.backends
        self.backend = backend
        self.model_name = model_name
        self.embedding_llm = None
        self._loaded_model_name = None
        self.cache = {}
        self._ensure_model_entry(self.model_name)

    def _ensure_model_entry(self, model_name: str):
        if model_name not in self.cache:
            self.cache[model_name] = {
                'docs': [],
                'embs': [],
                'n_tokens': []
            }

    def _get_all_known_docs(self):
        known_docs = []
        seen = set()
        for m_data in self.cache.values():
            for doc in m_data['docs']:
                if doc not in seen:
                    seen.add(doc)
                    known_docs.append(doc)
        return known_docs

    @property
    def embs(self):
        self._ensure_model_entry(self.model_name)
        return self.cache[self.model_name]['embs']

    @embs.setter
    def embs(self, value):
        self._ensure_model_entry(self.model_name)
        self.cache[self.model_name]['embs'] = list(value) if isinstance(value, np.ndarray) else value

    @property
    def docs(self):
        self._ensure_model_entry(self.model_name)
        return self.cache[self.model_name]['docs']

    @docs.setter
    def docs(self, value):
        self._ensure_model_entry(self.model_name)
        self.cache[self.model_name]['docs'] = value

    @property
    def n_tokens(self):
        self._ensure_model_entry(self.model_name)
        return self.cache[self.model_name]['n_tokens']

    @n_tokens.setter
    def n_tokens(self, value):
        self._ensure_model_entry(self.model_name)
        self.cache[self.model_name]['n_tokens'] = value

    def set_model(self, model_name):
        self.model_name = model_name
        self._ensure_model_entry(model_name)

    def load_llm(self, model_name=None):
        target_model = model_name or self.model_name
        if self.embedding_llm is not None and self._loaded_model_name == target_model:
            return
        if self.backend == 'cuda':
            from vllm import LLM
            self.embedding_llm = LLM(model=target_model, task="embed")
        elif self.backend == 'cpu':
            self.embedding_llm = SentenceTransformersLLM(target_model)
        self._loaded_model_name = target_model

    def add_documents(self, docs: List[str], model_name=None):
        target_model = model_name or self.model_name
        self._ensure_model_entry(target_model)

        existing_docs = set(self.cache[target_model]['docs'])
        missing_docs = []
        for doc in docs:
            if doc not in existing_docs:
                missing_docs.append(doc)
                existing_docs.add(doc)

        if not missing_docs:
            return

        self.load_llm(target_model)
        outputs = self.embedding_llm.embed(missing_docs)
        embs = [np.array(x.outputs.embedding) for x in outputs]
        n_tokens = [len(x.prompt_token_ids) for x in outputs]

        self.cache[target_model]['embs'] += embs
        self.cache[target_model]['docs'] += missing_docs
        self.cache[target_model]['n_tokens'] += n_tokens

    def search(self, new_docs, model_name=None):
        target_model = model_name or self.model_name
        self._ensure_model_entry(target_model)

        # Garante que documentos já registrados no cache (em outros modelos) tenham embeddings para o modelo atual
        known_docs = self._get_all_known_docs()
        if known_docs:
            self.add_documents(known_docs, model_name=target_model)

        m_start = time()
        self.load_llm(target_model)
        emb_start = time()
        new_outputs = self.embedding_llm.embed(new_docs)
        gpu_time = time() - emb_start
        avg_time_per_doc = gpu_time / len(new_docs)
        new_embs = np.asarray([np.array(x.outputs.embedding) for x in new_outputs])
        new_n_tokens = [len(x.prompt_token_ids) for x in new_outputs]

        model_embs = np.asarray(self.cache[target_model]['embs'])
        model_docs = self.cache[target_model]['docs']

        sims = cosine_similarity(model_embs, new_embs)
        sims_dict = {d: [] for d in new_docs}
        for doc, docs_sims in zip(model_docs, sims):
            for new_doc, score in zip(new_docs, docs_sims):
                sims_dict[new_doc].append([score, doc])

        for d, scores in sims_dict.items():
            scores.sort(reverse=True)

        outputs_dict = {}
        m_end = time()
        total_cpu_time = m_end - m_start
        cpu_time = total_cpu_time - gpu_time
        avg_cpu_time = cpu_time / len(new_docs)
        for doc, n_tokens in zip(new_docs, new_n_tokens):
            outputs_dict[doc] = {
                'sims_dict': sims_dict[doc],
                'meta': {
                    'input_tokens': n_tokens,
                    'processing_time': avg_time_per_doc,
                    'no_gpu_time': avg_cpu_time,
                    'model_name': target_model
                }
            }

        return outputs_dict

    def persist(self, path):
        merged_cache = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    existing_data = json.load(f)
                merged_cache = EmbeddingStore._parse_raw_cache(existing_data)
            except Exception:
                merged_cache = {}

        for m_name, m_data in self.cache.items():
            if not m_data['docs']:
                continue
            if m_name not in merged_cache:
                merged_cache[m_name] = {'docs': [], 'embs': [], 'n_tokens': []}
            seen = set(merged_cache[m_name]['docs'])
            for doc, emb, n_tok in zip(m_data['docs'], m_data['embs'], m_data['n_tokens']):
                if doc not in seen:
                    seen.add(doc)
                    merged_cache[m_name]['docs'].append(doc)
                    merged_cache[m_name]['embs'].append(emb)
                    merged_cache[m_name]['n_tokens'].append(n_tok)

        self.cache = merged_cache
        self._ensure_model_entry(self.model_name)

        models_json = {}
        for m_name, m_data in self.cache.items():
            if not m_data['docs']:
                continue
            models_json[m_name] = {
                'docs': m_data['docs'],
                'n_tokens': m_data['n_tokens'],
                'embs': [list(x) if isinstance(x, np.ndarray) else x for x in m_data['embs']]
            }

        j = {
            'model_name': self.model_name,
            'models': models_json
        }
        with open(path, 'w') as f:
            json.dump(j, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if os.path.exists(naturezas_dump_path):
        embedding_store = EmbeddingStore.load_store(naturezas_dump_path, paraphrase_multil_model_name)
    else:
        embedding_store = EmbeddingStore(paraphrase_multil_model_name)
    embedding_store.add_documents(naturezas_vec)
    embedding_store.persist(naturezas_dump_path)