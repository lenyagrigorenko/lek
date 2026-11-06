# Usage:  make all            (whole pipeline -> submissions/submission.csv)
#         make PY=/path/to/python all
PY  ?= python
CFG ?= configs/exp/final.yaml

.PHONY: install test prepare external cv ensemble all report
install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

test:
	$(PY) -m pytest

prepare:                       # standardise SMILES, build data/processed/*.pkl
	$(PY) -m vacv_si prepare -c $(CFG)

external:                      # download ChEMBL, assemble + validate data/external/vacv_external.csv
	$(PY) -m vacv_si collect --sources chembl -c $(CFG)

cv:                            # repeated CV of every model + refits on all data (OOF / test predictions)
	$(PY) -m vacv_si cv -c $(CFG)

ensemble:                      # blend, evaluate on the competition metric, write the submission
	$(PY) -m vacv_si ensemble -c $(CFG)

all:                           # collect (if the csv is missing) -> prepare -> cv -> ensemble -> submissions/submission.csv
	$(PY) -m vacv_si run-all -c $(CFG)

report:
	$(PY) -m vacv_si report -c $(CFG)
