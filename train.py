# -*- coding: utf-8 -*-
import argparse
import csv
import itertools
import json
import logging
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
import mlflow
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from datetime import datetime

np.random.seed(16)

import warnings

warnings.filterwarnings("ignore")

# setup the logging environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

params_svm = [dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]

label2int = {
	"fact": {"low": 0, "mixed": 1, "high": 2},
	        "bias": {"left": 0, 'extreme-left': 0,
                 "center": 1, 'right-center': 1, 'left-center': 1,
                 "right": 2, 'extreme-right': 2},
}

int2label = {
	"fact": {0: "low", 1: "mixed", 2: "high"},
	"bias": {0: "left", 1: "center", 2: "right"},
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def log_params(args):
	params = {
		'dataset': args.dataset,
		'task': args.task,
		'type_training': args.type_training,
		'normalize_features': args.normalize_features,
		'features': ", ".join(args.features)
	}
	mlflow.log_params(params)


def load_json(file_path):
	with open(file_path, 'r') as f:
		data = json.load(f)

	return data


def load_prediction_file(file_path):
	with open(file_path, 'r') as f:
		csv_reader = csv.DictReader(f)
		if args.task == 'fact':
			data = {row['source_url']: [float(row['low']), float(row['mixed']), float(row['high'])]
					for row in csv_reader}
		else:
			data = {row['source_url']: [float(row['left']), float(row['center']), float(row['right'])]
					for row in csv_reader}
	return data


def calculate_metrics(actual, predicted):
	"""
	Calculate performance metrics given the actual and predicted labels.
	Returns the macro-F1 score, the accuracy, the flip error rate and the
	mean absolute error (MAE).
	The flip error rate is the percentage where an instance was predicted
	as the opposite label (i.e., left-vs-right or high-vs-low).
	"""
	# calculate macro-f1
	f1 = f1_score(actual, predicted, average='macro') * 100

	# calculate accuracy
	accuracy = accuracy_score(actual, predicted) * 100

	# calculate the flip error rate
	flip_err = sum([1 for i in range(len(actual)) if abs(actual[i] - predicted[i]) > 1]) / len(actual) * 100

	# calculate mean absolute error (mae)
	mae = sum([abs(actual[i] - predicted[i]) for i in range(len(actual))]) / len(actual)
	mae = mae[0] if not isinstance(mae, float) else mae


	mlflow.log_metric('f1', f1)
	mlflow.log_metric('accuracy', accuracy)
	mlflow.log_metric('flip_error', flip_err)
	mlflow.log_metric('mae', mae)
	return f1, accuracy, flip_err, mae


def train_model(splits: Dict[str, Dict[str, List[str]]],
				features: Dict[str, Dict[str, List[float]]],
			    labels: Dict[str, str]):
	# create placeholders where predictions will be cumulated over the different folds
	all_urls = []
	actual = np.zeros(len(labels), dtype=np.int)
	predicted = np.zeros(len(labels), dtype=np.int)
	probs = np.zeros((len(labels), args.num_labels), dtype=np.float)

	i = 0
	num_folds = len(splits)

	logger.info("Start training...")

	for f in range(num_folds):
		logger.info(f"Fold: {f}")

		# get the training and testing media for the current fold
		urls = {
			"train": splits[str(f)]["train"],
			"test": splits[str(f)]["test"],
		}

		all_urls.extend(splits[str(f)]["test"])

		# initialize the features and labels matrices
		X, y = {}, {}

		# concatenate the different features/labels for the training sources
		X["train"] = np.asmatrix([list(itertools.chain(*[v[url] for _, v in features.items()]))
								 for url in urls["train"]]).astype("float")
		y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)

		# concatenate the different features/labels for the testing sources
		X["test"] = np.asmatrix([list(itertools.chain(*[v[url] for _, v in features.items()]))
								for url in urls["test"]]).astype("float")
		y["test"] = np.array([labels[url] for url in urls["test"]], dtype=np.int)

		if args.normalize_features:
			# normalize the features values
			scaler = MinMaxScaler()
			scaler.fit(X["train"])
			X["train"] = scaler.transform(X["train"])
			X["test"] = scaler.transform(X["test"])

		# fine-tune the model
		clf_cv = GridSearchCV(SVC(), scoring="f1_macro", cv=num_folds, n_jobs=4, param_grid=params_svm)
		clf_cv.fit(X["train"], y["train"])

		# train the final classifier using the best parameters during crossvalidation
		clf = SVC(
			kernel=clf_cv.best_estimator_.kernel,
			gamma=clf_cv.best_estimator_.gamma,
			C=clf_cv.best_estimator_.C,
			probability=True
		)
		clf.fit(X["train"], y["train"])

		# generate predictions
		pred = clf.predict(X["test"])

		# generate probabilites
		prob = clf.predict_proba(X["test"])

		# cumulate the actual and predicted labels, and the probabilities over the different folds.  then, move the index
		actual[i: i + y["test"].shape[0]] = y["test"]
		predicted[i: i + y["test"].shape[0]] = pred
		probs[i: i + y["test"].shape[0], :] = prob
		i += y["test"].shape[0]

	# calculate the performance metrics on the whole set of predictions (5 folds all together)
	results = calculate_metrics(actual, predicted)

	# display the performance metrics
	logger.info(f"Macro-F1: {results[0]}")
	logger.info(f"Accuracy: {results[1]}")
	logger.info(f"Flip Error-rate: {results[2]}")
	logger.info(f"MAE: {results[3]}")

	# map the actual and predicted labels to their categorical format
	predicted = np.array([int2label[args.task][int(l)] for l in predicted])
	actual = np.array([int2label[args.task][int(l)] for l in actual])

	# create a dictionary: the keys are the media, and the values are their actual and predicted labels
	predictions = {all_urls[i]: (actual[i], predicted[i]) for i in range(len(all_urls))}

	# create a dataframe that contains the list of m actual labels, the predictions with probabilities.  then store it in the output directory
	df_out = pd.DataFrame({
		"source_url": all_urls,
		"actual": actual,
		"predicted": predicted,
		int2label[args.task][0]: probs[:, 0],
		int2label[args.task][1]: probs[:, 1],
		int2label[args.task][2]: probs[:, 2],
	})
	columns = ["source_url", "actual", "predicted"] + [int2label[args.task][i] for i in range(args.num_labels)]
	df_out.to_csv(os.path.join(out_dir, "predictions.tsv"), index=False, columns=columns)

	return results


def train_combined_model(corpus_path: str, splits_file: str, feature_files: Dict[str, str]):
	# read the dataset
	df = pd.read_csv(corpus_path, sep="\t")

	# create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
	df['labels'] = df[args.task].apply(lambda x: label2int[args.task][x])
	labels = dict(df[['source_url_normalized', 'labels']].values.tolist())

	# load the evaluation splits
	splits = load_json(splits_file)

	# create the features dictionary: each key corresponds to a feature type, and its value is the pre-computed features dictionary
	loaded_features = {}
	for feature, feature_file in feature_files.items():
		loaded_features[feature] = load_json(feature_file)

	with mlflow.start_run():
		log_params(args)
		results = train_model(splits, loaded_features, labels)

	# write the experiment results in a tabular format
	res = PrettyTable()
	res.field_names = ["Macro-F1", "Accuracy", "Flip error-rate", "MAE"]
	res.add_row(results)

	# write the experiment summary and outcome into a text file and save it to the output directory
	with open(os.path.join(out_dir, "results.txt"), "w") as f:
		f.write(summary.get_string(title="Experiment Summary") + "\n")
		f.write(res.get_string(title="Results"))


def train_ensemble_model(corpus_path: str, splits_file: str, feature_files: Dict[str, str]):
	"""Uses the results from previously trained SVM classifier's probabilities for the three classes
	with different features.

	Example:
	Let say you have two features - A, B. To use them in the ensemble training first you need to
	train SVM classifier with each of the features (separately). The model will save for each of
	the records three probabilities for each of the classes:

		|source_url      | actual | predicted | low | mixed |high |
		| -------------- | ------ | --------- | --- | ----- |---- |
		|allthatsfab.com | mixed  | high      | 0.07| 0.22  | 0.69|

	Args:
		corpus_path (str): [description]
		splits_file (str): [description]
		feature_files (Dict[str, str]): [description]
	"""
	# read the dataset
	df = pd.read_csv(corpus_path, sep="\t")

	# create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
	df['labels'] = df[args.task].apply(lambda x: label2int[args.task][x])
	labels = dict(df[['source_url_normalized', 'labels']].values.tolist())

	# load the evaluation splits
	splits = load_json(splits_file)

	loaded_features = {}
	for feature, feature_file in feature_files.items():
		loaded_features[feature] = load_prediction_file(feature_file)

	with mlflow.start_run():
		log_params(args)
		results = train_model(splits, loaded_features, labels)

	# write the experiment results in a tabular format
	res = PrettyTable()
	res.field_names = ["Macro-F1", "Accuracy", "Flip error-rate", "MAE"]
	res.add_row(results)

	# write the experiment summary and outcome into a text file and save it to the output directory
	with open(os.path.join(out_dir, "results.txt"), "w") as f:
		f.write(summary.get_string(title="Experiment Summary") + "\n")
		f.write(res.get_string(title="Results"))


def parse_arguments():
	parser = argparse.ArgumentParser()

	# Required command-line arguments
	parser.add_argument(
		"-f",
		"--features",
		type=str,
		default="",
		required=True,
		help="the features that will be used in the current experiment (comma-separated)",
	)
	parser.add_argument(
		"-tk",
		"--task",
		type=str,
		default="fact",
		required=True,
		help="the task for which the model is trained: (fact or bias)",
	)

    # Boolean command-line arguments
	parser.add_argument(
        "-cc",
        "--clear_cache",
        action="store_true",
        help="flag to whether the corresponding features file need to be deleted before re-computing",
    )
	parser.add_argument(
		'-nf',
		'--normalize_features',
		action='store_true',
		help='flag whether to normalize input features. In the case of ensemble it\'s better to be false'
	)

    # Other command-line arguments
	parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="acl2020",
        help="the name of the dataset for which we are building the media objects",
    )
	parser.add_argument(
		"-nl",
		"--num_labels",
		type=int,
		default=3,
		help="the number of classes of the given task",
	)

	parser.add_argument(
		"-tt",
		"--type_training",
		type=str,
		default="combine",
		help="Indicates what type of model training to do. Possible values are: 'combine', 'ensemble'",
	)

	return parser.parse_args()


if __name__ == "__main__":

	# parse the command-line arguments
	args = parse_arguments()

	if not args.features:
		raise ValueError("No Features are specified")

	# create the list of features sorted alphabetically
	args.features = sorted([feature.strip() for feature in args.features.split(",")])

	# display the experiment summary in a tabular format
	summary = PrettyTable()
	summary.add_row(["task", args.task])
	summary.add_row(["classification mode", "single classifier"])
	summary.add_row(["type_training", args.type_training])
	summary.add_row(["normalize_features", args.normalize_features])
	summary.add_row(["features", ", ".join(args.features)])
	print(summary)

	corpus_path = os.path.join(PROJECT_DIR, "data", args.dataset, "corpus.tsv")
	splits_file = os.path.join(PROJECT_DIR, "data", args.dataset, f"splits.json")

	if args.type_training == "combine":
		# specify the output directory where the results will be stored
		out_dir = os.path.join(PROJECT_DIR, "data", args.dataset, f"results", f"{args.task}_{','.join(args.features)}")

		# remove the output directory (if it already exists and args.clear_cache was set to TRUE)
		shutil.rmtree(out_dir) if args.clear_cache and os.path.exists(out_dir) else None

		# create the output directory
		os.makedirs(out_dir, exist_ok=True)

		feature_files = {feature: os.path.join(PROJECT_DIR, "data", args.dataset, "features", f"{feature}.json")
						 for feature in args.features}

		train_combined_model(corpus_path, splits_file, feature_files)

	elif args.type_training == "ensemble":
		now = datetime.now()

		# specify the output directory where the results will be stored
		out_dir = os.path.join(PROJECT_DIR, "data", args.dataset, 'results', f"ensemble_{args.task}_{','.join(args.features)}_{now.strftime('%Y%m%d')}")

		# remove the output directory (if it already exists and args.clear_cache was set to TRUE)
		# shutil.rmtree(out_dir) if args.clear_cache and os.path.exists(out_dir) else None

		# create the output directory
		os.makedirs(out_dir, exist_ok=True)

		result_dir = os.path.join(PROJECT_DIR, "data", args.dataset, f"results")
		features = [f'{args.task}_{feature}' for feature in args.features]

		features_files = {}
		for feature in args.features:
			feature_folder = f'{args.task}_{feature}'
			if not feature_folder in os.listdir(result_dir):
				raise ValueError(F"Feature '{feature_folder}' was not generated.Please run the code in 'combine' type_training for generating it.")

			if 'predictions.tsv' not in os.listdir(os.path.join(result_dir, feature_folder)):
				raise ValueError(f"Missing 'predictions.tsv' file for '{feature_folder}' in {result_dir}")

			features_files[feature] = os.path.join(result_dir, feature_folder, 'predictions.tsv')

		train_ensemble_model(corpus_path, splits_file, features_files)
	else:
		raise ValueError(f"Unsupported type_training ('{args.type_training}'). Supported ones are 'combine' or 'ensemble'!")
