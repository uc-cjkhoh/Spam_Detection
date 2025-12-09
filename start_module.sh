#!/bin/bash

set -e

prefect server start --host 0.0.0.0 > prefect.log 2>&1 &
mlflow server --host 0.0.0.0 > mlflow.log 2>&1 &
python main.py > main.log 2>&1 &

wait