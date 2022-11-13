# CHEST xRAY CLASSIFICATIO END-TO-END USING MLFLOW
# Mlflow-01
```
mlflow ui
```
```
mlruns folder is mlflow project component in action
reproduce mlruns
```
```
conda env export > conda.yaml
```
```
if some want ot run the experiment
all they need is conda.yaml and main.py
```
```
mlflow run .
```

```
mlflow run . --no-conda
```
```
mlflow run githtttps -f 3 -s 4
```
```
mlflow models serve -m abspath -p 1234 --no-conda
```
```
{'colums':[],'data':[[]]}
```
```
mlflow run . -e stage_01 --no-conda
```
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts -h 127.0.0.1 -p 3456
```



    