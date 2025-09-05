"""
使用 Sentence-BERT 计算零阶信念与一阶信念输出的语义相似度。

依赖:
    pip install -U sentence-transformers scikit-learn

示例:
    from bert_consine_calculate import BeliefSimilarityCalculator
    calculator = BeliefSimilarityCalculator()
    score = calculator.compare(zero_text, first_text)
"""

from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BeliefSimilarityCalculator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _load_sentences_from_text(text: str) -> List[str]:
        lines = [ln.strip(" -\t") for ln in text.splitlines()]
        return [ln for ln in lines if ln]

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def _average_embedding(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.zeros((1, embeddings.shape[1] if embeddings.ndim == 2 else 384))
        return np.mean(embeddings, axis=0, keepdims=True)

    def compare(self, zero_order_text: str, first_order_text: str) -> float:
        """计算零阶信念与一阶信念之间的余弦相似度(0~1)。"""
        zero_sentences = self._load_sentences_from_text(zero_order_text)
        first_sentences = self._load_sentences_from_text(first_order_text)

        zero_emb = self._embed_sentences(zero_sentences)
        first_emb = self._embed_sentences(first_sentences)

        zero_avg = self._average_embedding(zero_emb)
        first_avg = self._average_embedding(first_emb)

        sim = cosine_similarity(zero_avg, first_avg)[0][0]
        return float((sim + 1) / 2)

# 示例用法
if __name__ == "__main__":
    zero_order_text = """
zero_order_beliefs(Bob)
- target_object_state(burger(100011))
  - location(livingroom(4000))
  - completion(incomplete)
- target_object_state(burger(12321))
  - location(bedroom)
  - completion(Unknown)
- target_object_state(burger(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(banana(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(bread(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(apple(Unknown))
  - location(Unknown)
  - completion(Unknown)
- container_state(Unknown)
  - location(Unknown)
  - contents[Unknown, Unknown, Unknown]
- agent_state(Alice)
  - location(Office(1000))
  - subplan(“go to livingroom(1000) and transport the apple")
  - object_in_hand[Null, Null]
- agent_state(Bob)
  - location(Unknown)
  - subplan(Unknown)
  - object_in_hand[Unknown, Unknown]
- room_state(Office(1000))
  - exploration_state(Part)
- room_state(Livingroom(2000))
  - exploration_state(Unknown)
- room_state(Livingroom(3000))
  - exploration_state(Unknown)
- room_state(Livingroom(4000))
  - exploration_state(Unknown)
- room_state(Kitchen(5000))
  - exploration_state(Unknown)
- room_state(Office(7000))
  - exploration_state(Unknown)
- room_state(Bedroom(8000))
  - exploration_state(Unknown)
- bed_location()
  - location(Unknown)
    """
    first_order_text = """
zero_order_beliefs(Alice)
- target_object_state(burger(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(burger(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(burger(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(banana(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(orange(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(bread(Unknown))
  - location(Unknown)
  - completion(Unknown)
- target_object_state(apple(Unknown))
  - location(Unknown)
  - completion(Unknown)
- container_state(Unknown)
  - location(Unknown)
  - contents[Unknown, Unknown, Unknown]
- agent_state(Alice)
  - location(Office(1000))
  - subplan(Unknown)
  - object_in_hand[Null, Null]
- agent_state(Bob)
  - location(Unknown)
  - subplan(Unknown)
  - object_in_hand[Unknown, Unknown]
- room_state(Office(1000))
  - exploration_state(Part)
- room_state(Livingroom(2000))
  - exploration_state(Unknown)
- room_state(Livingroom(3000))
  - exploration_state(Unknown)
- room_state(Livingroom(4000))
  - exploration_state(Unknown)
- room_state(Kitchen(5000))
  - exploration_state(Unknown)
- room_state(Office(7000))
  - exploration_state(Unknown)
- room_state(Bedroom(8000))
  - exploration_state(Unknown)
- bed_location()
  - location(Unknown)
    """
    calculator = BeliefSimilarityCalculator()
    score = calculator.compare(zero_order_text, first_order_text)
    print(f"Belief similarity: {score:.4f}")
