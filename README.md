# 🚀 Sentence Transformer + Multi-Task Learning (Fetch ML Apprentice)  

This project implements sentence embeddings using Hugging Face + SentenceTransformers.
I have extended it to multi-task learning with custom heads in later stages as per instructions from Fetch.

---

## 🔍 Overview

This project implements a sentence transformer architecture with a multi-task learning setup. It demonstrates the ability to:
- Use a pre-trained sentence transformer as a shared encoder
- Build and evaluate two NLP tasks in parallel
- Design flexible training logic
- Simulate inference, loss, and accuracy within a multi-task loop

📅 All tasks are handled cleanly using PyTorch, SentenceTransformers, and standard ML engineering principles.

---

## ▶️ Run Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/Piyush-Project-X/sentence-transformer-ml.git
cd sentence-transformer-ml
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Individual Tasks

#### Task 1 - Sentence Embedding:
```bash
python -m src.sentence_transformer
```

#### Task 2 - Multi-Task Model Output Test:
```bash
python -m src.test_multitask_model
```

#### Task 4 - Simulated Training Loop:
```bash
python -m src.train
```

---

## 🧠 Task 1: Sentence Transformer Implementation

- Implemented using: `sentence-transformers/all-MiniLM-L6-v2`
- Converts input sentences to 384-dimensional embeddings
- Verified with a sample input and embedding output


## Architectural Choices Outside the Transformer Backbone
To keep the implementation lightweight and focused on sentence embedding quality, I chose to use the pre-trained all-MiniLM-L6-v2 model from Hugging Face's sentence-transformers library as the core embedding generator.

Outside of the transformer backbone, here are the key design choices made:

- Minimal Preprocessing: The model accepts raw input sentences without additional tokenization or cleaning steps. This ensures the use of the pre-trained tokenizer and encoder optimized for general-purpose usage. 

- Direct Embedding Extraction: Instead of adding any additional layers (e.g., MLPs or custom encoders), I directly used the output embeddings from the transformer as fixed-length sentence representations. This keeps the focus on evaluating the transformer’s native ability to encode semantic meaning.

- Batching Support: I structured the input to support both single and multiple sentences, enabling simple extension to larger batches or downstream tasks like classification.

These choices were made to preserve the pre-trained model's generalization power, reduce unnecessary architectural complexity, and keep the focus on reusability and performance in multi-task settings later.


📂 **File:** `src/sentence_transformer.py`

---

## 🤝 Task 2: Multi-Task Learning Expansion

Two parallel tasks were implemented using a shared backbone:

| Task    | Description                  | Labels                                                  |
|---------|------------------------------|---------------------------------------------------------|
| Task A  | Sentence Type Classification | ["statement", "question", "command", "informational"]   |
| Task B  | Sentiment Analysis           | ["positive", "neutral", "negative"]                     |

👷‍♂️ **Architecture:**
- Shared encoder: `SentenceTransformer`
- Two classifier heads: Task A (4 classes), Task B (3 classes), visualized below:

---
          ┌────────────────────────────┐ 
          │   SentenceTransformer      │ ← shared backbone (embeddings)
          └────────────┬───────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
       Task A Head            Task B Head
    (Sentence Type)        (Sentiment Analysis)
      (4 classes)              (3 classes)


---
         

 The above diagram illustrates a potential architecture where a shared Sentence Transformer model is used to generate sentence embeddings. These embeddings are then fed into separate task-specific heads:

* **Task A Head (Sentence Type):** This head is responsible for classifying sentences into one of 4 predefined classes (e.g., Informational, Question, Command, Exclamatory).
* **Task B Head (Sentiment Analysis):** This head performs sentiment analysis, classifying sentences into one of 3 classes (e.g., Positive, Neutral, Negative).

The shared Sentence Transformer backbone allows the model to learn general-purpose sentence representations that can be beneficial for both downstream tasks, potentially improving efficiency and performance.    


## Changes Made to Support Multi-Task Learning (MTL)
To expand the sentence transformer into a multi-task learning (MTL) model, I introduced task-specific classifier heads on top of a shared transformer backbone. Here's how the architecture was extended:

- Shared Sentence Encoder
The SentenceTransformer model (all-MiniLM-L6-v2) is used as a common encoder for all tasks. It converts each input sentence into a fixed-length embedding (384-dim), capturing semantic meaning in a task-agnostic way.

Task-Specific Heads
After encoding, the output embedding is passed into two separate linear layers (heads), each customized for its respective task:

- Task A Head – Sentence Type Classification
A linear layer that outputs logits for 4 classes: ["statement", "question", "command", "informational"].

- Task B Head – Sentiment Analysis
Another linear layer that outputs logits for 3 classes: ["positive", "neutral", "negative"].

- Independent Outputs and Losses
Each head operates independently, producing its own predictions and loss. This modular setup allows for joint or selective training of either task head without impacting the other.

Simple & Flexible Design
I avoided complex cross-task dependencies to keep the model general-purpose and easier to adapt for future tasks. Each task-specific head is small and trainable, allowing focused fine-tuning without modifying the transformer backbone.

This MTL architecture promotes parameter sharing, improves training efficiency, and provides task isolation, while keeping the model extensible and easy to maintain.




📂 **File:** `src/multitask_model.py`  
📆 **Tested using:** `src/test_multitask_model.py`

---

## 🧪 Task 3: Training Considerations

### 🔹 Scenario 1: If the entire network is frozen
If both the transformer and task heads are frozen, the model becomes read-only — useful only for inference. No learning or adaptation happens.

### 🔹 Scenario 2: If only the transformer backbone is frozen
In this case, I kept the sentence embeddings fixed and trained only the task-specific heads. This is efficient and works well when the pre-trained model already captures strong semantic features.

### 🔹 Scenario 3: If only one task head is frozen
Freezing just one head lets us preserve its performance while fine-tuning the other. This is ideal for updating or improving just one task without disturbing the rest.

---

### 🔄 Transfer Learning Strategy

- **Model Chosen**: `sentence-transformers/all-MiniLM-L6-v2`
- **Frozen Layers**: Initially froze the transformer, trained both task heads
- **Why**: Keeps training fast and prevents overfitting. If deeper adaptation is needed, one can unfreeze the top transformer layers.



---

## 🔁 Task 4: Training Loop Implementation

The training loop simulates how the multi-task model would be trained in a real scenario, using dummy data.

### ✅ Key Design Decisions:

- **Hypothetical Data**:  
  I created sample sentences and corresponding labels to represent input-output pairs without relying on external datasets.

- **Forward Pass**:  
  Each input sentence is encoded using the shared transformer, then passed into both task-specific heads. The logits from each head are used to calculate individual task losses.

- **Loss & Backpropagation**:  
  I computed cross-entropy loss for both Task A and Task B, then summed them for a combined loss. This total loss was used to simulate a backpropagation step.

- **Metrics**:  
  Task-specific accuracy and total loss are printed after each epoch to show per-task performance, even though the training is not on real data.

This structure demonstrates how multi-task training operates, with shared representations and separate objectives, and shows how each component fits into a modular, testable training loop.



📂 **File:** `src/train.py`

### Sample Output:
```
Epoch 1
Task A Loss: 1.38 | Accuracy: 37.50%
Task B Loss: 1.10 | Accuracy: 25.00%
Total Loss: 2.48

Note: The output shown is based on sample inputs. It may change if you add, remove, or modify the input sentences.
```

---

## 🐳 How to Run in Docker  

### Step-by-step Docker Usage:

#### Step 1: Build the Image
```bash
docker build -t sentence-transformer-ml .
```

#### Step 2: Run the Container
```bash
docker run -it --rm sentence-transformer-ml
```

This will automatically run the training loop (`src.train`) inside Docker. 
To run a different task module inside Docker, you can update the last line in the Dockerfile. For example, to run Task 1 instead of Task 4:
```
CMD ["python", "-m", "src.sentence_transformer"]
```

---

## 📁 Project Structure
```bash
sentence-transformer-ml/
├── src/
│   ├── sentence_transformer.py     # Task 1
│   ├── multitask_model.py          # Task 2 architecture
│   ├── test_multitask_model.py     # Task 2 test
│   ├── train.py                    # Task 4 training loop
│   └── __init__.py
├── requirements.txt
├── Dockerfile
└── README.md
```



## ✍️ Author
**Piyush Sonawane**  
[LinkedIn](https://www.linkedin.com/in/piyush-sonawane22) | [GitHub](https://github.com/Piyush-Project-X)  
Melbourne, FL | Email: psonawane2022@my.fit.edu

