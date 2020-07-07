# mlflow_examples
test run mlflow with lightgbm (does not work with windows yet)

Running the code
python train.py --colsample-bytree 0.8 --subsample 0.9
You can try experimenting with different parameter values like:

python train.py --learning-rate 0.06 --colsample-bytree 0.6 --subsample 0.8
Then you can open the MLflow UI to track the experiments and compare your runs via:

mlflow ui
Running the code as a project
mlflow run . -P learning_rate=0.2 -P colsample_bytree=0.8 -P subsample=0.9

