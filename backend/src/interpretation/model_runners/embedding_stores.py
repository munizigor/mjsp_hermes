from time import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from model_runners.templates import naturezas_vec

naturezas_dump_path = "/datasets/naturezas_cache_vllm.json"
default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class SBertOutput:
    def __init__(self, embedding):
        self.embedding = embedding

class SBertResult:
    def __init__(self, embedding, prompt_token_ids):
        self.outputs = SBertOutput(embedding)
        self.prompt_token_ids = prompt_token_ids

'''Adaptador para carregar um modelo com a biblioteca SentenceTransformers
e criar embeddings de documentos usando CPU.
'''
class SentenceTransformersLLM:

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __init__(self, model_name=default_model):
        """
        Inicializa o modelo SentenceTransformer para criar embeddings.
        
        Args:
            model_name (str): Nome do modelo a ser carregado.
        """
        # Load model from HuggingFace Hub
        print('Loading', model_name)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print('Loaded', model_name)

    def embed(self, docs):
        """
        Gera embeddings para uma lista de documentos.

        Args:
            docs (list of str): Lista de documentos para gerar embeddings.

        Returns:
            list: Lista de objetos com embeddings e token ids.
        """
        #print('Tokenizing')
        encoded_input = self.tokenizer(docs, padding=True, 
                                       truncation=True, return_tensors='pt')
        #print(encoded_input['input_ids'].shape)
        tokens = encoded_input['input_ids']
        # Compute token embeddings
        with torch.no_grad():
            #print('Encoding')
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        #print('Mean pooling')
        sentence_embeddings = SentenceTransformersLLM.mean_pooling(model_output, 
            encoded_input['attention_mask'])
        #print(sentence_embeddings.shape)

        embeddings = [vector.tolist() for vector in sentence_embeddings]
        outputs = []
        #print('Saving')
        for s, emb, t in zip(docs, embeddings, tokens):
            #print(s, len(t))
            outputs.append(SBertResult(emb, t))
        assert len(embeddings) == len(docs)
        assert len(tokens) == len(docs)
        return outputs

class EmbeddingStore:
    backends = ['cuda', 'cpu']

    def load_store(path, backend='cuda'):
        assert backend in EmbeddingStore.backends
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        store = EmbeddingStore(model_name=data['model_name'], backend=backend)
        store.embs = np.asarray(data['embs'])
        store.docs = data['docs']
        store.n_tokens = data['n_tokens']
        return store

    def __init__(self, model_name=default_model, backend='cuda'):
        assert backend in EmbeddingStore.backends
        self.backend  = backend
        self.model_name = model_name
        self.embedding_llm = None
        self.embs = []
        self.docs = []
        self.n_tokens = []
    
    def load_llm(self):
        if self.backend == 'cuda':
            #Usar VLLM
            from vllm import LLM
            self.embedding_llm = LLM(model=self.model_name, task="embed")
        elif self.backend == 'cpu':
            self.embedding_llm = SentenceTransformersLLM(self.model_name)

    def add_documents(self, docs):
        if self.embedding_llm is None:
            self.load_llm()
        outputs = self.embedding_llm.embed(docs)
        embs = [np.array(x.outputs.embedding) for x in outputs]
        n_tokens = [len(x.prompt_token_ids) for x in outputs]
        self.embs += embs
        self.docs += docs
        self.n_tokens += n_tokens

    def search(self, new_docs):
        m_start = time()
        if self.embedding_llm is None:
            self.load_llm()
        emb_start = time()
        new_outputs = self.embedding_llm.embed(new_docs)
        gpu_time = time() - emb_start
        avg_time_per_doc = gpu_time / len(new_docs)
        new_embs = np.asarray([np.array(x.outputs.embedding) for x in new_outputs])
        new_n_tokens = [len(x.prompt_token_ids) for x in new_outputs]
        sims = cosine_similarity(np.asarray(self.embs), new_embs)
        sims_dict = {d: [] for d in new_docs}
        for doc, docs_sims in zip(self.docs, sims):
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
                    'model_name': self.model_name
                }
            }

        return outputs_dict

    def persist(self, path):
        import json
        j = {
            'model_name': self.model_name,
            'docs': self.docs,
            'n_tokens': self.n_tokens,
            'embs': [list(x) for x in self.embs]
        }
        json.dump(j, open(path, 'w'), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    #Iniciar cache de naturezas
    embedding_store = EmbeddingStore()
    embedding_store.add_documents(naturezas_vec)
    embedding_store.persist(naturezas_dump_path)