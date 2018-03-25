# Simple linear regression example in python


### Description

It uses basic linear regression to calculate predictions for `test` data, with coefficients
build using `training` data. And also verifies that method with calculating `RMSE` for predictions
(square root of summed square error by size).


### Installation and usage

Install required packages with
```bash
pipenv install
```

Start created virtual environment
```bash
pipenv shell
```

Run app
```bash
python src/main.py ./data/train.csv ./data/test_preview.csv
```