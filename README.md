# Quantum CHSH Correlation Predictor

This project aims to train a machine learning model that can predict the CHSH correlation of a two-qubit quantum circuit, given the values of three parameters: theta, 1-qubit gate error rate, and 2-qubit gate error rate.

The CHSH correlation is a well-known quantity used to test the violation of Bell's inequality and determine if the correlations between two distant systems are classical or quantum.

The model is trained using a dataset generated by simulating a quantum circuit that implements the CHSH game, adding depolarizing noise to the gates, and computing the resulting CHSH correlations. The simulation is done using the Qiskit framework.

## Dependencies

- Python 3.7 or higher
- Qiskit
- Scikit-learn
- Pandas
- Numpy

The dependencies can be installed by running:

```shell
pip install -r requirements.txt
```

## Usage

The project consists of two main scripts: `circuit.py` and `model.py`.

`circuit.py` contains the functions for generating the CHSH circuit and computing the CHSH correlation.

`model.py` contains the functions for generating the dataset, training the machine learning model, and evaluating its performance.

To generate the dataset, run:

```shell
python model.py --generate
```

This will create a CSV file data.csv containing the generated data.

To train the machine learning model, run:

```shell
python model.py --train
```

This will train a linear regression model using the generated data and save it as a pickle file `model.pkl`.

To evaluate the model's performance, the script computes the R-squared score on a held-out test set.

## Directory structure

```
.
├── README.md
├── circuit.py
├── data.csv
├── model.pkl
├── model.py
└── requirements.txt
```

`circuit.py` contains the code for the CHSH circuit simulation and CHSH correlation computation.

`data.csv` contains the generated dataset.

`model.pkl` is the trained machine learning model.

`model.py` contains the code for generating the dataset, training the machine learning model, and evaluating its performance.

`requirements.txt` contains the list of Python dependencies.
