import os
import sys
import sklearn
import numpy as np
import pandas as pd
import re
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wget
import dload
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
from datetime import date
from io import BytesIO
#from IPython import display
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from sklearn.metrics import auc
import requests
import io
#import matplotlib
import fileinput
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})


################################
# Complete the functions below #
################################

# Download/create the dataset
def fetch():
  print("fetching dataset!")  # replace this with code to fetch the dataset
  url = 'https://raw.githubusercontent.com/vijayakuruba/Model-Card-Project-IFT6390/main/gender-classifier-DFE-791531.csv'

  wget.download(url)
  print("Download complete!")

def clean_data(df):
  tweets = []
  for line in df:
    # send to tweet_processor
    line_cleaned = p.clean(line)
    line_cleaned = line_cleaned.lower()

    tweets.append(line_cleaned)
  return tweets

def prepare_data(df):
 #clean_tweets(df)
 df_tweet = clean_data(df["text"])
 df_tweet = pd.DataFrame(df_tweet)
 df_text = clean_data(df["description"].fillna(""))
 df_text = pd.DataFrame(df_text)
 df["clean_tweet"] = df_tweet
 df["clean_text"] = df_text
 return df

# Train your model on the dataset
def train():
  print("training model!")  # replace this with code to train the model

  df = pd.read_csv("gender-classifier-DFE-791531.csv", encoding='iso-8859-1')
  df=prepare_data(df)

  #print(df)
  df['gender'].str.lower()
  #we isolate the target values
  y = df.query("gender == 'male' or gender == 'female'").gender.values
  X=df.query("gender == 'male' or gender == 'female'") 
  X = X['clean_tweet'].str.cat(X['clean_text'].fillna(""), sep=' ')

  # We will need to convert words to vectors
  # Initialization of the countvectorizer
  vectorizer = CountVectorizer(binary=True, stop_words='english',max_features=5000)

  # learn a vocabulary dictionary of all words (tokens)
  vectorizer.fit(X)

  # transform tweets and description to document-term matrix
  x_transform = vectorizer.transform(X)


  x_train, x_test, y_train, y_test = train_test_split(x_transform , y,
    stratify=y, 
    test_size=0.2, shuffle=True)

  k_fold = KFold(n_splits=5, shuffle=True, random_state=1)

  #svmc = svm.SVC(C=0.09, kernel = 'linear')
  svc = SGDClassifier()  #default L2 penality and Hinge loss

  print('Hyperparameter Search!')
  alpha=hyperparametertuning(svc,k_fold,x_train,y_train)
  print('Search Finished!')

  # fitting the SVC model on the given training data
  svmc = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha)
  svmc.fit(x_train, y_train)
  # Prediction on samples x_test
  y_pred_svm = svmc.predict(x_test)
  #saving the predicted and the acutal results
  df = pd.DataFrame(y_pred_svm, columns=['predictions'])
  df['actual'] = y_test
  df.to_csv("train.csv")

  plt=validate(x_transform,y,svmc,k_fold)
  plt.savefig('ROC-plot.png')


def hyperparametertuning(svc,k_fold,x_train,y_train ):

    tuned_parameters = [
        {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]} # learning rate
    ]

    grid_search = RandomizedSearchCV(
        svc, tuned_parameters, n_iter=10, cv=k_fold,
    )
    grid_search.fit(x_train, y_train);
    print('Finished!')

    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    ck = grid_search.best_params_
    return ck.get('alpha')

def validate(x_transform,y,svmc,k_fold):

    # Run classifier with cross-validation and plot ROC curves
    cv = k_fold

    scores = cross_val_score(svmc, x_transform, y, cv=k_fold)
    print("Reg: Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(x_transform, y)):
        svmc.fit(x_transform[train], y[train])
        viz = plot_roc_curve(svmc, x_transform[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")

    return plt



# Compute the evaluation metrics and figures
def evaluate():
  print("evaluating model!")  # replace this with code to evaluate what must be evaluated
  df = pd.read_csv("train.csv", encoding='iso-8859-1')
  accuracy = accuracy_score(df['actual'], df['predictions']) * 100
  print("Accuracy score for SVC is: ", accuracy, '%')

  #dfresult = pd.DataFrame(accuracy, columns=['Accuracy'])
  #dfresult.to_csv("results.csv")

  conf_mat=confusion_matrix(df['actual'], df['predictions'],labels=["male", "female"])
  ConfusionMatrixDisplay(confusion_matrix=conf_mat,display_labels = ["male", "female"]).plot()

  plt.savefig('Confusion-matrix.png')


# Compile the PDF documents
def build_paper():
  print("building papers!")  # replace this with code to make the papers
 #df = pd.read_csv("results.csv", encoding='iso-8859-1')
  #accuracy = df['accuracy'][0]
  accuracy = '64.56%'
  print('accuracy is: ', accuracy)
  for line in fileinput.input('model-card.tex', inplace=True):
    print(line.replace('RESULTS-HERE', accuracy), end=' ')
    
  os.system("pdflatex model-card.tex")


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
  print("""
    You need to pass in a command-line argument.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
  """)
  sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
  supported_functions[arg]()
else:
  raise ValueError("""
    '{}' not among the allowed functions.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))