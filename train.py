# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import shutil
import logging
import argparse
import itertools
import collections
import pandas as pd
from sklearn.svm import SVC
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


import numpy as np
np.random.seed(16)

import warnings
warnings.filterwarnings("ignore")

# setup the logging environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

params_svm = [dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]

label2int = {
	"fact": {"low": 0, "mixed": 1, "high": 2},
	"bias": {"left": 0, "center": 1, "right": 2},
}

int2label = {
	"fact": {0: "low", 1: "mixed", 2: "high"},
	"bias": {0: "left", 1: "center", 2: "right"},
}

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

	return f1, accuracy, flip_err, mae


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
	
    # Other command-line arguments
	parser.add_argument(
        "-hd",
        "--home_dir",
        type=str,
        default="/Users/baly/Projects/News-Media-Reliability",
        help="the directory that contains the project files"
    )
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

	return parser.parse_args()


if __name__ == "__main__":

	# parse the command-line arguments
	args = parse_arguments()

	if not args.features:
		raise ValueError("No Features are specified")

	# create the list of features sorted alphabetically
	args.features = sorted([feature for feature in args.features.split(",")])

	# specify the output directory where the results will be stored
	out_dir = os.path.join(args.home_dir, "data", args.dataset, f"results", f"{args.task}_{','.join(args.features)}")

	# remove the output directory (if it already exists and args.clear_cache was set to TRUE)
	shutil.rmtree(out_dir) if args.clear_cache and os.path.exists(out_dir) else None

	# create the output directory
	os.makedirs(out_dir, exist_ok=True)

	# display the experiment summary in a tabular format
	summary = PrettyTable()
	summary.add_row(["task", args.task])
	summary.add_row(["classification mode", "single classifier"])
	summary.add_row(["features", ", ".join(args.features)])
	print(summary)

	# read the dataset
	df = pd.read_csv(os.path.join(args.home_dir, "data", args.dataset, "corpus.tsv"), sep="\t")

	# create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
	labels = {df["source_url_normalized"][i]: label2int[args.task][df[args.task][i]] for i in range(df.shape[0])}

	# load the evaluation splits
	splits = json.load(open(os.path.join(args.home_dir, "data", args.dataset, f"splits.json"), "r"))
	num_folds = len(splits)

	# create the features dictionary: each key corresponds to a feature type, and its value is the pre-computed features dictionary
	features = {feature: json.load(open(os.path.join(args.home_dir, "data", args.dataset, "features", f"{feature}.json"), "r")) for feature in args.features}

	# create placeholders where predictions will be cumulated over the different folds
	all_urls = []
	actual = np.zeros(df.shape[0], dtype=np.int)
	predicted = np.zeros(df.shape[0], dtype=np.int)
	probs = np.zeros((df.shape[0], args.num_labels), dtype=np.float)

	i = 0

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
		X["train"] = np.asmatrix([list(itertools.chain(*[features[feat][url] for feat in args.features])) for url in urls["train"]]).astype("float")
		y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)

		# concatenate the different features/labels for the testing sources
		X["test"] = np.asmatrix([list(itertools.chain(*[features[feat][url] for feat in args.features])) for url in urls["test"]]).astype("float")
		y["test"] = np.array([labels[url] for url in urls["test"]], dtype=np.int)

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
	df_out = pd.DataFrame({"source_url": all_urls, "actual": actual, "predicted": predicted, int2label[args.task][0]: probs[:, 0], int2label[args.task][1]: probs[:, 1], int2label[args.task][2]: probs[:, 2],})
	columns = ["source_url", "actual", "predicted"] + [int2label[args.task][i] for i in range(args.num_labels)]
	df_out.to_csv(os.path.join(out_dir, "predictions.tsv"), index=False, columns=columns)

	# write the experiment results in a tabular format
	res = PrettyTable()
	res.field_names = ["Macro-F1", "Accuracy", "Flip error-rate", "MAE"]
	res.add_row(results)

	# write the experiment summary and outcome into a text file and save it to the output directory
	with open(os.path.join(out_dir, "results.txt"), "w") as f:
		f.write(summary.get_string(title="Experiment Summary") + "\n")
		f.write(res.get_string(title="Results"))
