# version1: GCN + AUTO-ENCODER




- Multiply features by Adj before learning
- LOSS = NLL + RECONSTRUTION MSE LOSS

DATASET | ACC | STD
--- | --- | ---
CORA | 81.1  | 0.9
CITESEER | 70.6  | 0.5
PUBMED | 79.1  | 0.3



# version2: GCN + AUTO-ENCODER + lINK PREDICTION

- Multiply features by Adj before learning
- LOSS = NLL + RECONSTRUTION MSE LOSS
- KNN used for link prediction

DATASET | ACC | STD
--- | --- | ---
CORA | 81.4 | 0.5
CITESEER | 71.1 | 0.6
PUBMED|78.9 |  0.4