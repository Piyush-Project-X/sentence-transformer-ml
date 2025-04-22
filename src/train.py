import torch
import torch.nn as nn
import torch.optim as optim
from src.multitask_model import MultiTaskModel

# ------------------------------
# Simulated (hypothetical) data, 
# because of the less sentences in the dataset, model may not give good results but if we provide proper dataset, model will perform at it's best.
# ------------------------------
sentences = [
    "What time is the meeting tomorrow?",          # question
    "Please finalize the report.",                 # command
    "Piyush is presenting his ML model.",          # informational
    "This is a powerful framework.",               # statement
    "Can we connect next week?",                   # question
    "Make sure to update the chart.",              # command
    "The CPU usage was above 90% yesterday.",      # informational
    "It’s a beautiful day outside."                # statement
]

task_a_labels = torch.tensor([1, 2, 3, 0, 1, 2, 3, 0])  # task A: 0=statement, 1=question, 2=command, 3=informational
task_b_labels = torch.tensor([1, 2, 1, 0, 1, 2, 1, 0])  # task B: 0=positive, 1=neutral, 2=negative


# ------------------------------
# Model Setup 
# ------------------------------
model = MultiTaskModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# ------------------------------
# Training Loop (1 epoch only)
# ------------------------------
model.train()
for epoch in range(1):  # simulate 1 epoch
    # Forward pass
    task_a_logits, task_b_logits = model(sentences)

    # Compute losses
    loss_a = criterion(task_a_logits, task_a_labels)
    loss_b = criterion(task_b_logits, task_b_labels)

    # Combine both losses
    total_loss = loss_a + loss_b

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Metrics (argmax predictions)
    pred_a = torch.argmax(task_a_logits, dim=1)
    pred_b = torch.argmax(task_b_logits, dim=1)

    acc_a = (pred_a == task_a_labels).float().mean()
    acc_b = (pred_b == task_b_labels).float().mean()

    print(f"\nEpoch {epoch+1}")
    print(f"Task A Loss: {loss_a.item():.4f} | Accuracy: {acc_a.item()*100:.2f}%")
    print(f"Task B Loss: {loss_b.item():.4f} | Accuracy: {acc_b.item()*100:.2f}%")
    print(f"Total Loss: {total_loss.item():.4f}")
