# Citation-Prediction-by-Leveraging-Transformers-and-Natural-Language-Processing-Heuristics

Implementation of the project described in the paper Citation-Prediction-by-Leveraging-Transformers-and-Natural-Language-Processing-Heuristics.

This repo contains code for automating the citation prediction problem using transformers and related experiments for performance evaluation.

## Prerequisites
You can download the models we use and fine-tuned from the following links

gpt2: https://drive.google.com/file/d/1rwYv-hbjLwicLhi3Os4-TQUu5dOaLIdX/view?usp=drive_link

bert: https://drive.google.com/file/d/1aXrQ3vTegDDC4TCkI1iykk4LOykg7QLl/view?usp=drive_link

The experiments were carried out on windows 11, using python 3.9.13 (anaconda prompt)

To replicate the experiments it's recommendable to install anaconda prompt and have an nvidia gpu supporting CUDA

## Installing
Steps are the following:

Clone the repo:
```
git clone https://github.com/Marcomurgia97/Citation-Prediction-by-Leveraging-Transformers-and-Natural-Language-Processing-Heuristics.git
```

move into the repo:
```
cd Citation-Prediction-by-Leveraging-Transformers-and-Natural-Language-Processing-Heuristics
```
Open the anaconda prompt and create the virtual environment:
```
virtualenv venv
```
then:
```
venv\Scripts\activate
```
install pytorch with cuda
```
pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```
install all packages required
```
pip install -r requirements.txt
```
then:
```
python -m spacy download en_core_web_sm
```
## Running the tests
to run the experiments for the generative approach move to the folder GenApproach:
```
cd GenApproach
```
then if you want to use the gpt2 model trained on arxiv dataset and run the experiments with heuristics:
```
python citationPrediction.py "lysandre/arxiv-nlp"
```
then if you want to use the gpt2 model trained by us on s2orc  dataset and run the experiments with heuristics:

```
python citationPrediction.py "path\where\you\downloaded\gpt2"
```
instead if you want to use the gpt2 model trained on arxiv dataset and run the experiments without heuristics:
```
python citationPredictionNoHeur.py "lysandre/arxiv-nlp"
```
instead if you want to use the gpt2 model trained by us on s2orc  dataset and run the experiments without heuristics:

```
python citationPredictionNoHeur.py "path\where\you\downloaded\gpt2"
```
