# CS5760 – Natural Language Processing  
## Homework 4 – RNNs, LSTMs, Attention & Transformers  

*Student Name:* Duggineni Sesha Rao  
*Course:* CS5760 – NLP  
*Semester:* Fall 2025   
*University:* University of Central Missouri  

---

## Overview
This repository contains my submission for Homework 4 of CS5760.  
The assignment is divided into:

1. *Part I – Short Answer Questions*  
2. *Part II – Programming Tasks* involving RNN-based language modeling, a simplified Transformer encoder, and a manual implementation of scaled dot-product attention.

This README summarizes how each section was completed and explains the reasoning and methodology behind the code found in the notebook.

---

# ================================================
# PART I – Short Answer Summary
# ================================================

All short answers are included in the PDF submission. Here is a high-level summary of how I approached each problem.

---

## 1. RNN Families & Use-Cases
I classified each NLP task according to the most suitable RNN input/output structure:

- *Next-word prediction* → many-to-one (model processes several prior words to predict the next).  
- *Sentiment classification* → many-to-one (entire sentence mapped to one sentiment label).  
- *NER* → many-to-many aligned (each token receives one tag).  
- *Machine translation* → many-to-many unaligned (output length differs from input length).

I also explained how unrolling creates a time-expanded version of the RNN, enabling BPTT with shared weights across timesteps. Weight sharing reduces parameters but can make learning long-range dependencies more difficult.

---

## 2. Vanishing Gradients & Remedies
I described how gradients shrink as they pass through many recurrent steps, making it difficult for standard RNNs to retain information from distant tokens.

Architectural solutions included:
- *LSTMs*, which use a linear memory path and gating.  
- *GRUs*, which simplify the gating mechanism but preserve gradient flow effectively.

As a training method, I discussed *gradient clipping* and how constraining gradient magnitude stabilizes optimization.

---

## 3. LSTM Gates & Cell State
I explained the function of LSTM’s gated components:
- Forget gate decides what to erase.  
- Input gate determines what new information enters memory.  
- Output gate regulates what information is revealed at each timestep.

The cell state provides an almost linear conduit for gradients, preventing decay.  
I also contrasted how LSTMs distinguish between internal memory ("what stays") and external output ("what is shown").

---

## 4. Self-Attention
I defined Q, K, and V according to the slide definitions:

- Query: expresses what a token is searching for  
- Key: expresses what each token offers  
- Value: the information returned once attention scores are calculated  

I wrote the mathematical expression  

Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V

and explained why dividing by √dₖ keeps the softmax distribution stable during training.

---

## 5. Multi-Head Attention & Residual Layers
I explained that multiple heads allow attention to learn different types of relationships simultaneously (syntactic, semantic, positional).  
Residual connections and LayerNorm improve training stability by preserving information and preventing activation drift.

---

## 6. Encoder–Decoder & Masking
I described how the decoder uses a future-mask so it cannot look ahead while generating tokens.  
I also outlined the difference between:
- Encoder self-attention (interactions only among input tokens)  
- Cross-attention (decoder attends to encoder outputs)

During inference, the decoder generates tokens sequentially without teacher forcing.

---

# ================================================
# PART II – PROGRAMMING TASKS (Conceptual Explanation)
# ================================================

Below is a complete explanation of how each programming task was approached.  
No direct code is included here; the notebook contains the full implementation with comments.

---

# Q1 – Character-Level RNN Language Model

### Objective
Build a small character-level model capable of predicting the next character based on previous characters. The task emphasizes sequence modeling, embeddings, teacher forcing, and autoregressive sampling.

### Data Preparation
I began by creating a small toy dataset using repeated patterns such as “hello”, “help”, etc.  
This baseline corpus is then optionally extended with an external 50–200 KB plain-text document to create more variability.

The steps include:
- Lowercasing all characters  
- Constructing a character vocabulary  
- Encoding characters into integer indices  
- Creating sliding-window training samples (input = characters [t ... t+L-1], target = [t+1 ... t+L])

### Model Architecture
I used the following pipeline:

*Embedding layer:* Learns a dense vector for each character.  
*Recurrent layer:* Implemented with an LSTM/GRU to capture sequence information.  
*Linear projection:* Maps hidden states to vocabulary logits.  
*Softmax:* Used implicitly through cross-entropy during training.

Hyperparameters (sequence length, batch size, hidden size) follow assignment constraints.

### Training Procedure
Key elements in my implementation:

- *Teacher forcing:* The true previous character is used as input at each timestep.  
- *Loss:* Cross-entropy over the entire sequence.  
- *Optimization:* Adam optimizer with gradient clipping for stability.  
- *Epochs:* Trained for 5–20 epochs depending on dataset size.

A graph of training (and optional validation) loss is included in the notebook.

### Sampling & Temperature
After training, I built a sampling loop:

- Start with an initial character.  
- Predict next characters using the model output.  
- Apply different temperatures (0.7, 1.0, 1.2) to control randomness.  
- Generate 200–400 characters.

I also wrote a reflection discussing how sequence length, hidden size, and temperature change the behavior of the model.

---

# Q2 – Mini Transformer Encoder

### Objective
Implement a simplified Transformer encoder from scratch using:

- Token embeddings  
- Sinusoidal positional encoding  
- Multi-head self-attention  
- Feed-forward network  
- Residual connections + LayerNorm

### Dataset & Tokenization
I selected 10 simple English sentences.  
The steps include:
- Word-level tokenization  
- Building a vocabulary  
- Padding sentences to uniform length  
- Mapping words to indices  
This produced a tensor suitable for Transformer input.

### Positional Encoding
Following the original Transformer design:
- I created sinusoidal encoding matrices for each position  
- Added these encodings to the token embeddings  
This gives the model access to word order information.

### Multi-Head Attention
My implementation included:
- Linear projections to obtain Q, K, V  
- Reshaping into multiple heads  
- Scaled dot-product attention for each head  
- Concatenation and final projection  

### Feed-Forward Layer
A two-layer feed-forward block with ReLU activation was applied to each token independently.

### Add & Norm
Each sublayer used:
- Residual addition  
- Layer normalization  

This improves gradient flow and stabilizes training.

### Outputs
The notebook displays:
- Encoded token IDs  
- Contextual embeddings from the encoder  
- A full attention heatmap showing which words attend to each other  

---

# Q3 – Scaled Dot-Product Attention

### Objective
Re-implement the core attention computation used in Transformers:


Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V


### Implementation Structure
I designed a standalone function that:

1. Computes raw attention scores (QKᵀ).  
2. Scales them by √dₖ.  
3. Applies softmax to get normalized attention weights.  
4. Multiplies weights with V to produce the final outputs.

### Testing
I created random Q, K, and V matrices and passed them through the function.  
I printed:

- Raw scores  
- Scaled scores  
- Attention weight matrix  
- Output vectors  

### Softmax Stability Check
To illustrate the importance of scaling, I compared:

- softmax(raw_scores)  
- softmax(scaled_scores)

This demonstrates how scaling smooths the distribution and avoids extremely large logits that would destabilize softmax.

---

# How to Run the Notebook
1. Open the .ipynb file in Google Colab or Jupyter Notebook.  
2. Ensure PyTorch and matplotlib are available (Colab includes them).  
3. Run the cells in order for Q1, Q2, and Q3.  
4. Upload any custom text file if you wish to extend the Q1 dataset.

---

# Student Information
*Name:* Duggineni Sesha Rao  
*Course:* CS5760 – Natural Language Processing  
*Semester:* Fall 2025  

This README explains the workflow and design choices behind the code in this repository and fulfills the documentation requirement described in the assignment.
