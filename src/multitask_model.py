import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MultiTaskModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', embedding_dim=384):
        super(MultiTaskModel, self).__init__()
        # Shared Transformer Backbone
        self.encoder = SentenceTransformer(model_name)

        # Task A: Sentence Type Classification (4 classes)
        self.task_a_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # 4 classes
        )

        # Task B: Sentiment Analysis (3 classes)
        self.task_b_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # 3 classes
        )

    def forward(self, sentences):
        # Get shared sentence embeddings
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)

        # Task-specific logits
        task_a_logits = self.task_a_head(embeddings)
        task_b_logits = self.task_b_head(embeddings)

        return task_a_logits, task_b_logits
