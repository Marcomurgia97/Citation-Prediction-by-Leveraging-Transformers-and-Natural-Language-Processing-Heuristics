# Citation Prediction using Transformers and Natural Language Processing Heuristics

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://python.org) <!-- Optional: Add badges like this -->

This repository contains the code for automating the citation prediction problem using Transformer models and related experiments for performance evaluation, as described in our paper:

➡️ **[Citation prediction by leveraging transformers and natural language processing heuristics](https://www.sciencedirect.com/science/article/pii/S0306457323003205)** ⬅️

We explore two main approaches: a **Generative** one based on GPT-2 and a **Named Entity Recognition (NER)** one using BERT, both enhanced with Natural Language Processing heuristics.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Pre-trained Models:** Download the models we used and fine-tuned:
    *   **GPT-2 (Fine-tuned on S2ORC):** [Download from Google Drive](https://drive.google.com/file/d/1rwYv-hbjLwicLhi3Os4-TQUu5dOaLIdX/view?usp=drive_link)
    *   **BERT (Fine-tuned on S2ORC):** [Download from Google Drive](https://drive.google.com/file/d/1aXrQ3vTegDDC4TCkI1iykk4LOykg7QLl/view?usp=drive_link)
2.  **Environment:**
    *   **Python 3.9.13** (we recommend using [Anaconda](https://www.anaconda.com/products/distribution))
    *   **OS:** Experiments were conducted on Windows 11.
    *   **GPU:** An NVIDIA GPU supporting CUDA is **highly recommended** for reasonable execution times.

## Installation

Follow these steps to set up your environment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Marcomurgia97/Citation-Prediction-by-Leveraging-Transformers-and-Natural-Language-Processing-Heuristics.git
    ```
2.  **Navigate into the directory:**
    ```bash
    cd Citation-Prediction-by-Leveraging-Transformers-and-Natural-Language-Processing-Heuristics
    ```
3.  **Create and activate a virtual environment** (using Anaconda Prompt or your preferred terminal):
    ```bash
    # Create the environment (e.g., named 'citepred')
    conda create -n citepred python=3.9.13
    # Activate the environment
    conda activate citepred
    ```
    *(Alternatively, using `virtualenv`)*
    ```bash
    # python -m venv venv  # Or: virtualenv venv
    # venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On Linux/macOS
    ```

4.  **Install PyTorch with CUDA support:**
    *(Ensure the CUDA version (`cu117` here) matches your system's CUDA installation)*
    ```bash
    pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
    ```
5.  **Install all required packages:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Download the SpaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Running the Experiments

You are now ready to run the experiments. The results (prediction `.txt` files) will be saved in their respective approach folders (`GenApproach` or `NER`). The placeholder `[CITE]` indicates where the model suggests a citation is needed.

---

### Generative Approach (GPT-2)

1.  **Navigate to the folder:**
    ```bash
    cd GenApproach
    ```
2.  **Run the experiments:**

    *   **WITH heuristics:**
        *   Using the ArXiv pre-trained GPT-2 model:
            ```bash
            python citationPrediction.py "lysandre/arxiv-nlp"
            ```
        *   Using our S2ORC fine-tuned GPT-2 model:
            ```bash
            python citationPrediction.py "/path/where/you/downloaded/gpt2"
            ```
            *(Replace `/path/where/you/downloaded/gpt2` with the actual path)*
    *   **WITHOUT heuristics:**
        *   Using the ArXiv pre-trained GPT-2 model:
            ```bash
            python citationPredictionNoHeur.py "lysandre/arxiv-nlp"
            ```
        *   Using our S2ORC fine-tuned GPT-2 model:
            ```bash
            python citationPredictionNoHeur.py "/path/where/you/downloaded/gpt2"
            ```
            *(Replace `/path/where/you/downloaded/gpt2` with the actual path)*

    *   **Output:** These scripts generate the `prediction.txt` file.

---

### NER Approach (BERT)

1.  **Navigate to the folder:**
    ```bash
    cd NER
    ```
    *(If you were in `GenApproach`, use `cd ../NER`)*
2.  **Run the experiments:**

    *   **NER Prediction (without heuristics):**
        *   Using our S2ORC fine-tuned BERT model:
            ```bash
            python NER_test.py "/path/where/you/downloaded/bert/checkpoint-55500"
            ```
            *(Replace `/path/where/you/downloaded/bert/checkpoint-55500` with the actual path)*
        *   **Output:** This script generates the `predictionNer.txt` file.

    *   **(Optional) Apply heuristics:**
        *   To apply heuristics to the NER results:
            ```bash
            python NERHeur.py "predictionNer.txt"
            ```
        *   **Output:** This script reads `predictionNer.txt` and generates `predictionNerHeur.txt`.

---

## Performance Evaluation

To calculate performance metrics (Precision, Recall, F1-Score):

1.  **Navigate to the folder:**
    ```bash
    cd evaluation
    ```
    *(If you were in `NER`, use `cd ../evaluation`)*
2.  **Run the evaluation script**, specifying the prediction file to evaluate:

    *   **For the Generative Approach:**
        ```bash
        python evaluation.py "../GenApproach/prediction.txt"
        ```
    *   **For the NER Approach:**
        *   Without Heuristics:
            ```bash
            python evaluation.py "../NER/predictionNer.txt"
            ```
        *   With Heuristics:
            ```bash
            python evaluation.py "../NER/predictionNerHeur.txt"
            ```

## Contact and Issues

If you have any questions, suggestions, or encounter problems, please feel free to:

*   Open an **Issue** on this GitHub repository.
*   Contact us via email: `m.murgia98@studenti.unica.it`
