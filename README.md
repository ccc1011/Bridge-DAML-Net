# DAML-Net

Source code for the manuscript "A domain-adaptive multi-level framework for automated bridge component condition rating from textual inspection reports". This repository provides the implementation of the method described in the manuscript.

## Repository contents

| Item                    | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `pdf_to_txt.py`         | Converts PDF inspection reports to text                  |
| `extract_records.py`    | Extracts and cleans component records                    |
| `split_dataset.py`      | Filters ratings and splits the data by inspection report |
| `domain_pretraining.py` | Runs domain-adaptive pre-training                        |
| `train.py`              | Trains and evaluates the model                           |
| `daml_net.py`           | Defines the DAML-Net model                               |
| `configs/daml_net.yaml` | Records the DAML-Net configuration                       |
| `requirements.txt`      | Lists the Python dependencies                            |

## Data

The raw inspection reports are publicly available from the Vermont Agency of Transportation website:  
[https://vtrans.vermont.gov/docs/bridge-inspections](https://vtrans.vermont.gov/docs/bridge-inspections)

The processed data are not publicly available due to copyright restrictions.

## Usage

Place the PDF reports in `pdf_report/` and the pre-training corpus in `pre_training_document/`. Install the dependencies listed in `requirements.txt`. Then run `pdf_to_txt.py`, `extract_records.py`, `split_dataset.py`, `domain_pretraining.py` and `train.py` in this order. The main settings used in the manuscript are recorded in `configs/daml_net.yaml`. They are defined as default arguments in `domain_pretraining.py` and `train.py`, and at the end of the other scripts.

## Environment

Python 3.10, PyTorch 2.3.1, transformers 4.57.1, numpy 1.24.3, pandas 2.3.3, scikit-learn 1.7.2, pdfplumber 0.11.7, tqdm 4.67.1.