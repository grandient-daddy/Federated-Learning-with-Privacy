# Federated-Learning-with-Privacy

## Learning Algorithms:

- Sequential Association Rules
  - https://github.com/rn5l/session-rec/blob/master/algorithms/baselines/sr.py
  - https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AADL7wtch61kTi0sTXllKi1ga/Source-Code/algorithms/baselines?dl=0&preview=sr.py&subfolder_nav_tracking=1
  - extended
    - https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AADL7wtch61kTi0sTXllKi1ga/Source-Code/algorithms/baselines?dl=0&preview=sr_ext.py&subfolder_nav_tracking=1

- Sequential kNN models
  - https://github.com/rn5l/session-rec/tree/master/algorithms/knn
  - https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AABMZJSScwRS4jYjYGZSgCS7a/Source-Code/algorithms/knn?dl=0&subfolder_nav_tracking=1

- Sequential MF
  - https://github.com/rn5l/session-rec/tree/master/algorithms/smf
  - https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AACyOAXZ8C5fgtVodfIi0Wyea/Source-Code/algorithms/smf?dl=0&preview=smf.py&subfolder_nav_tracking=1
  - Translation-based MF
    - https://github.com/YifanZhou95/Translation-based-Recommendation/blob/master/src/TransRec.py

## Datasets
- Frappe (Mobile App Usage)
  - https://www.baltrunas.info/context-aware

- From Huawei:
1.  [MIT dataset](http://realitycommons.media.mit.edu/realitymining1.html) - The information collected includes call logs, Bluetooth devices in proximity, cell tower IDs, application usage, and phone status (such as charging and idle), which comes primarily from the Context application. The study generated data collected by one hundred human subjects over the course of nine months and represent approximately 500,000 hours of data on users' location, communication and device usage behavior. Paper describing dataset: [Reality mining: sensing complex social systems](https://link.springer.com/article/10.1007/s00779-005-0046-3)
2.  [Daily phone usage from kaggle](https://www.kaggle.com/johnwill225/daily-phone-usage?select=train.csv) - only launches and screen on/off
3.  [Google play store dataset](https://www.kaggle.com/gauthamp10/google-playstore-apps) - app category, rating, installs, free or paid
4.  [App Usage Behavior Modeling and Prediction](http://fi.ee.tsinghua.edu.cn/appusage/) - one-week app usage trace, contains timestamp, location, app categories, apps traffic, e.t.c

5. · [LiveLab](http://yecl.org/livelab/traces.html), appusage.sql - data contains name of app, time and date and duration, for which application was running in seconds. Paper describing dataset: [LiveLab: Measuring Wireless Networks and Smartphone Users in the Field](https://dl.acm.org/doi/pdf/10.1145/1925019.1925023)

6. · [LSapp](https://github.com/aliannejadi/LSApp) - traces collected from 292 users with mean length of 15 days; besides timestamps and app names contains event_type parameter, that contains info if app was opened, closed, interacted by user or broken.  


## Notebooks

1. DataPrep: Preparation of data.
2. Simple MF: Realization of simple MF decribed in section 5 of the Stage 1 Report. User can run evaluation of the MF method w/o LDP mechanism, with Laplace mechanism and with Harmony mechanism. To run the experiments one need to run all cells before eveluation section. In the Evaluation section, one can run experiments for Laplace and Harmony mechanisms, with the following parameters: number of factors in latent space, number of training epochs, privacy budget, gain, regularization parameter. The HR@K will be computed and depedence privacy budget/HR@K will be ploted. For comparison, one can compute HR@K  for SVD based method and for random model in corresponding sections.

## (Optional) Development Environment with Docker

In order to develop on different platforms we uses custom docker image for non-priviledge user (based on Nvidia CUDA image).
Image contains pre-built native extention and it is parametrized by user name and user ID in a host system.
The latter is crucial thing in binding host volumes.

```shell
docker build -t fedl --build-arg UID=$(id -u) .
docker run --rm -ti -e TERM=$TERM fedl
```

## Environment
We use `conda` package manager to install required python packages. In order to improve speed and reliability of package version resolution it is advised to use `mamba-forge` ([installation](https://github.com/conda-forge/miniforge#mambaforge)) that works over `conda`. Once `mamba is installed`, run the following command (while in the root of the repository):
```
mamba env create -f environment.yml
```
This will create new environment named `flrec` with all required packages already installed. You can install additional packages by running:
```
mamba install <package name>
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of modern IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `mamba install jupyterlab` or with `mamba install jupyter notebook`

*Note*: If you prefer to use `conda`, just replace `mamba` commands with `conda`, e.g. instead of `mamba install` use `conda install`.

## Tune Hyperparameters and Get Table Results

1. To get LSApp dataset and configure internal directories:
   ```shell
   ./bootstrap.sh
   ```

2. Run **data_prep_lsapp.ipynb** to prepare the dataset for further experiments.

3. Run **baselines.ipynb** to get the baselines results. *Table with the results* can be found './metrics_results/lsapp/baselines.csv'.

4. Run **mf.ipynb** to get the results for Matrix Factorization model. *Table with the results* can be found './metrics_results/lsapp/mf.csv'.

5. Run **seqmf_pp.ipynb** to get the results for Sequence-aware matrix factorization:
  - If you want to get the results from the table, then run the notebook as it is. *Table with the results* can be found './metrics_results/lsapp/seqmf_pp.csv'.
  - If you want to retrain the model you should change the third cell: choose 'optimizer_mode': ["adam", "sgd"], "find_optimal"=True, "save_result"=True
  to find optimal hyperparameters. After that you can continue running the notebook to get the final results. *Table with the results* can be found './metrics_results/lsapp/seqmf_pp.csv'.
  Note that you can find optimal hyperparameters in the folder './metrics_results/lsapp_score_params/{model}_scores_params_{optimizer_mode}.csv'.
6. Run **aggregate_results.ipynb** to get the final latex table with the metrics results. *Table with the final results* can be found './metrics_results/lsapp/overall.csv'.

7. Run **ablation-study.ipynb** to evaluate model in dynamics.
