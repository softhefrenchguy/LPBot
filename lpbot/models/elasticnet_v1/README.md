# ElasticNet v1.0

Build dataset:

```bash
python lpbot/models/elasticnet_v1/cli_build_dataset.py --input-1m data/ETHUSDC_1m.csv --out data/elasticnet_v1_dataset.csv --regime-path data/regime_outputs.csv
```

Train (walk-forward):

```bash
python lpbot/models/elasticnet_v1/cli_train.py --input-1m data/ETHUSDC_1m.csv --regime-path data/regime_outputs.csv --artifacts-dir models/elasticnet-v1.0
```

Evaluate:

```bash
python lpbot/models/elasticnet_v1/cli_evaluate.py --artifacts-dir models/elasticnet-v1.0
```

Infer latest:

```bash
python lpbot/models/elasticnet_v1/cli_infer.py --input-1m data/ETHUSDC_1m.csv --artifacts-dir models/elasticnet-v1.0 --n-latest 5
```
