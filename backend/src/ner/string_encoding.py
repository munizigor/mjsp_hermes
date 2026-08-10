from typing import List
from sentence_transformers import SentenceTransformer
import duckdb
import pickle

#Uses sentence-transformers for encoding sentences and saves the encodings in a cache.
#This is useful for comparing the semantic similarity of sentences, such as addresses.
#At each startup, it loads the model and the previous cache from disk.
#The cache is saved in a duckdb database file (parquet format)
#Encodings are lists of floats. They are separated in the cache table by the model name.

class SentenceEncodingCache():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = duckdb.connect(database='results/sentence_encoding_cache.duckdb', read_only=False)
        self.cache.execute("CREATE TABLE IF NOT EXISTS encodings (text TEXT, encoding BLOB, model TEXT)")
        self.cache.commit()

    def pre_calc(self, texts: List[str]):
        encoded_texts = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Save the encodings in the cache
        for text, encoding in zip(texts, encoded_texts):
            encoding_blob = pickle.dumps(encoding)
            self.cache.execute(
                "INSERT INTO encodings (text, encoding, model) VALUES (?, ?, ?)",
                (text, encoding_blob, self.model_name)
            )
        self.cache.commit()

    def get_encoding(self, text: str) -> List[float]:
        result = self.cache.execute(
            "SELECT encoding FROM encodings WHERE text = ? AND model = ?",
            (text, self.model_name)
        ).fetchone()
        if result is not None:
            encoding = pickle.loads(result[0])
            return encoding.tolist()
        else:
            # If not found, encode and save it
            encoding = self.model.encode(text, convert_to_numpy=True)
            encoding_blob = pickle.dumps(encoding)
            self.cache.execute(
                "INSERT INTO encodings (text, encoding, model) VALUES (?, ?, ?)",
                (text, encoding_blob, self.model_name)
            )
            self.cache.commit()
            return encoding.tolist()

    def clear(self):
        self.cache.execute("DELETE FROM encodings WHERE model = ?", (self.model_name,))
        self.cache.commit()

    def close(self):
        self.cache.close()
        self.model = None