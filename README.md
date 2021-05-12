# Factuality and Bias Prediction of News Media

This repository describes the work that was published in two papers (see citations below) on predicting the factuality and political bias in news media. Each paper proposes a different set of engineered features collected from sources of information related to the target media.

```
@InProceedings{baly:2018:EMNLP2018,
  author      = {Baly, Ramy  and  Karadzhov, Georgi  and  Alexandrov, Dimitar and  Glass, James  and  Nakov, Preslav},
  title       = {Predicting Factuality of Reporting and Bias of News Media Sources},
  booktitle   = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  series      = {EMNLP~'18},
  NOmonth     = {November},
  year        = {2018},
  address     = {Brussels, Belgium},
  NOpublisher = {Association for Computational Linguistics}
}
```

```
@InProceedings{baly:2020:ACL2020,
  author      = {Baly, Ramy and Karadzhov, Georgi and An, Jisun and Kwak, Haewoon and Dinkov, Yoan and Ali, Ahmed and Glass, James and Nakov, Preslav},
  title       = {What Was Written vs. Who Read It: News Media Profiling Using Text Analysis and Social Media Context},
  booktitle   = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  series      = {ACL~'20},
  NOmonth     = {July},
  year        = {2020},
  NOpublisher = {Association for Computational Linguistics}
}
```

## Corpus
The corpus was created by retrieving websites along with their factuality and bias labels from the Media Bias/Fact Check (MBFC) [website](http://mediabiasfactcheck.com/).  Two versions of the corpus ("emnlp18" and "acl2020") can be found at `./data/{version}/corpus.tsv`, and contains the following fields:
* **source_url**: the URL to each website (example: http://www.who.int/en/)
* **source_url_normalized**: a shortened version of the *source_url* (example: who.int-en). These will be used as IDs to split the data into 5 folds of training and testing (in `./data/splits.txt`)
* **ref**: the link to the page in the MBFC website analyzing the corresponding website (example: http://mediabiasfactcheck.com/world-health-organization-who/)
* **fact**: the factuality label of each website (low, mixed, or high)
* **bias**: the bias label of each website (extreme-right, right, center-right, center, center-left, left, extreme-left)

## Features
In addition to the corpus, we provide the different features that we used to obtain the results in our papers. We also include the script that reads these features, train the SVM classifier and writes the performance metrics and output predictions to file.  The features can be found at `./data/{version}/features/`.

1. For the *"emnlp18"* paper, the following features are used:
    - articles_body_glove
    - articles_title_glove
	- has_twitter
	- has_wikipedia
	- twitter_created_at
	- twitter_description
	- twitter_engagement
    - twitter_haslocation
	- twitter_urlmatch
	- twitter_verified
	- url_structure
	- wikipedia_categories
	- wikipedia_content
	- wikipedia_summary
	- wikipedia_toc

2. For the *"acl2020"* paper, the following features are used:
	- articles_body_bert
	- articles_title_bert
	- has_facebook
	- has_twitter
	- has_wikipedia
	- has_youtube
	- twitter_profile
	- twitter_followers
	- wikipedia_content
	- youtube_fulltext
	- youtube_nela
	- youtube_numerical
	- youtube_opensmile
    - youtube_subs

Details about each feature can be found in the cited papers. Each of these features is stored as a JSON file, where each key correspond to a source_url (normalized), and its value is a list of numerical values representing this particular feature.

## Training and Classification
To run the training script, use a command-line that follows the template below.

```
python3 train.py -tk [0] -f [1] -ds [2]
```

where
* [0] is the task at hand: "fact" or "bias" prediction
* [1] is the list of features (from the lists above) that will be used to train the model. features must be comma-separated.
* [2] is the name of the dataset we are running the experiment on ("acl2020" or "emnlp18").

The performance metrics and output predictions will be stored in `./data/{version}/results/{task}_{features}/`

#### Examples:
* Example combine model training
```
python train.py --features "articles_body_base_bert,articles_title_base_bert" \
				--task "fact" --type_training "combine"
```
OUTPUT:
```
+---------------------+---------------------------------------------------+
|       Field 1       |                      Field 2                      |
+---------------------+---------------------------------------------------+
|         task        |                        fact                       |
| classification mode |                 single classifier                 |
|    type_training    |                      combine                      |
|       features      | articles_body_base_bert, articles_title_base_bert |
+---------------------+---------------------------------------------------+
01-14 03:07:20 __main__     INFO     Start training...
01-14 03:07:20 __main__     INFO     Fold: 0
01-14 03:07:54 __main__     INFO     Fold: 1
01-14 03:08:25 __main__     INFO     Fold: 2
01-14 03:08:57 __main__     INFO     Fold: 3
01-14 03:09:28 __main__     INFO     Fold: 4
01-14 03:09:58 __main__     INFO     Macro-F1: 58.820048565901054
01-14 03:09:58 __main__     INFO     Accuracy: 65.07566938300349
01-14 03:09:58 __main__     INFO     Flip Error-rate: 5.70430733410943
01-14 03:09:58 __main__     INFO     MAE: 0.4062863795110594
```

* Example ensemble model training
NOTE to run this command first you need to run previous command with each of the features.
```
python train.py --features "articles_body_base_bert,articles_title_base_bert" \
				--task "fact" --type_training "ensemble"
```
OUTPUT:
```
+---------------------+---------------------------------------------------+
|       Field 1       |                      Field 2                      |
+---------------------+---------------------------------------------------+
|         task        |                        fact                       |
| classification mode |                 single classifier                 |
|    type_training    |                      ensemble                     |
|       features      | articles_body_base_bert, articles_title_base_bert |
+---------------------+---------------------------------------------------+
01-14 03:10:00 __main__     INFO     Start training...
01-14 03:10:00 __main__     INFO     Fold: 0
01-14 03:10:02 __main__     INFO     Fold: 1
01-14 03:10:03 __main__     INFO     Fold: 2
01-14 03:10:05 __main__     INFO     Fold: 3
01-14 03:10:06 __main__     INFO     Fold: 4
01-14 03:10:07 __main__     INFO     Macro-F1: 58.55282387414049
01-14 03:10:07 __main__     INFO     Accuracy: 66.23981373690337
01-14 03:10:07 __main__     INFO     Flip Error-rate: 3.7252619324796274
01-14 03:10:07 __main__     INFO     MAE: 0.37485448195576254
```

#### Evaluating experiments
Each experiment saves logging data with mlflow. To visualize the experiments you need to start MlFlow UI by running command inside the project dir
```
>>> mlflow ui

Expected output:
[2021-05-12 10:36:35 +0300] [106694] [INFO] Starting gunicorn 20.1.0
[2021-05-12 10:36:35 +0300] [106694] [INFO] Listening at: http://127.0.0.1:5000 (106694)
[2021-05-12 10:36:35 +0300] [106694] [INFO] Using worker: sync
```

After successful start you can access all logged experiments at "http://127.0.0.1:5000" (Default MlFlow setting)