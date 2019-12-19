# version1: GCN + AUTO-ENCODER




- Multiply features by Adj before learning
- LOSS = NLL + RECONSTRUTION MSE LOSS

## CORA
ACC: 81.1   STD: 0.9

## CITESEER
ACC: 70.6   STD: 0.5

## PUBMED
ACC: 79.1   STD: 0.3



# version2: GCN + AUTO-ENCODER + lINK PREDICTION

- Multiply features by Adj before learning
- LOSS = NLL + RECONSTRUTION MSE LOSS
- KNN used for link prediction

## CORA
ACC: 80.7   STD: 0.6

## CITESEER
ACC: 71.8   STD: 0.7

## PUBMED
ACC: 77.3   STD: 0.4