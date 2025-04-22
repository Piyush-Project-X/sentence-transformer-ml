import torch
from src.multitask_model import MultiTaskModel


# Simulated input sentences (batch size = 2)
sample_sentences = [
    "Please submit the report by tomorrow.",          # command
    "What are the new benefits of this policy?"       # question
]

# Initialize the model
model = MultiTaskModel()

# Set to eval mode (since no training is involved)
model.eval()

# Run forward pass
with torch.no_grad():
    task_a_logits, task_b_logits = model(sample_sentences)

# Output shapes
print("Task A (Sentence Type) Logits Shape:", task_a_logits.shape)
print("Task B (Sentiment) Logits Shape:", task_b_logits.shape)

# Optional: inspect raw logits
print("\nSample Logits (Task A):", task_a_logits)
print("\nSample Logits (Task B):", task_b_logits)
