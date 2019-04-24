import time
from joblib import dump, load
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Tensor Flow Versions
import tensorflow as tf
# Suppress verbose output from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def convert_answer( answer, max_size, score ) :
    "Convert a string to an array of ascii charactor ordinal values with fuzzy match score appended"
    converted = []
    for i in range(0, max_size) :
        if i < len(answer) :
            # print("{} {} ord: {}".format(i,  answer[i], ord(answer[i])))
            converted.append(float(ord(answer[i])))
        else :
            converted.append(32.0)
    converted.append(score)
    return converted

def revert_answer( value ) :
    "Convert an array of ascii ordinal values back to a string"
    answer_array = value[:-1]
    answer = ''
    for c in answer_array :
        answer += chr(int(c))
    return answer

def generate_model_data(filename, row_count = 0) :
    if row_count == 0 :
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(filename, nrows = row_count)

    answers =  data['answer'].apply(str).apply(str.lower)

    score = data['score']

    y =  data['correct_answer']

    converted_answers = []
    for i, answer in enumerate(answers) :
        converted_answers.append(convert_answer(answer, 24, score[i]))
        if i % 10000 == 0:
            print(i)
    X = np.array(converted_answers)
    print("X shape: {}, y shape: {}".format(X.shape, y.shape))

    return X, y

# Alternative approach to generating a column for each possible value of a field.
# Here each possible value is the possible answers. Each new column will have 0/1 values for the combination of the
# column name-answer_value.
# With a small enough sample set we would generate a new column for each possible value of the answer. So for 100
# rows we could have as many as 100 columns each with 0 or 1..  ex: 'answer_Kenneth Branaugh'
# For all permutations this would grow lineraly with the dataset so is not viable for this dataset.
def generate_dummy_model_data(filename, row_count = 0) :
    if row_count == 0 :
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(filename, nrows = row_count)

    data['answers'] =  data['answer'].apply(str).apply(str.lower)

    df = data.drop('correct_answer',axis=1)
    converted_answers = pd.get_dummies(df)

    y =  data['correct_answer']
    X = np.array(converted_answers)
    print("X shape: {}, y shape: {}".format(X.shape, y.shape))

    return X, y

def map_to_int(filename, data) :
    df = pd.read_csv(filename, header=None)
    actors = df[0].values.tolist()

    converted = []
    for actor in data :
        converted.append(actors.index(actor))
    return np.array(converted)

def generate_tensor_data(filename, row_count = 0) :
    if row_count == 0 :
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(filename, nrows = row_count)

    data.columns = [0,1,2]
    answers =  data[0].apply(str).apply(str.lower)

    score = data[2]

    y =  map_to_int("actors.csv",data[1].values)

    converted_answers = []
    for i, answer in enumerate(answers) :
        converted_answers.append(convert_answer(answer, 24, score[i]))
        if i % 10000 == 0:
            print(i)
    X = np.array(converted_answers)
    print("X shape: {}, y shape: {}".format(X.shape, y.shape))

    return X, y



def test_model(filename) :
    X, y = generate_model_data(filename+'.csv')
    # Test the models predication score. This is best done on smaller datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors = 10)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    print("Model score: {}".format(round(knn.score(X_test, y_test)),3))

def test_tensorflow(filename, iterations) :
    tf.logging.set_verbosity(tf.logging.ERROR)
    X, y = generate_tensor_data(filename+'.csv', 10000)
    # Test the models predication score. This is best done on smaller datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
    # Building a 3-layer DNN with 50 units each.
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
    classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                   hidden_units=[50, 50, 50],
                                                   n_classes=330)
    start = time.time()
    print("Start Tensorflow time (sec): {}".format(start))
    classifier_tf.fit(X_train, y_train, steps=iterations)
    end = time.time()
    print("Elapsed Tensorflow DNNClassifier time (sec): {}".format(round(end - start)))
    # dump(classifier_tf, filename + '_tf.joblib')
    predictions = list(classifier_tf.predict(X_test, as_iterable=True))
    print("Model TensorFlow DNNClassifier score: {}".format(metrics.accuracy_score(y_test, predictions)))

def compare_models(filename, iterations) :
    X, y = generate_model_data(filename+'.csv')
    # Test the models predication score. This is best done on smaller datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors = 10)
    # Fit the classifier to the training data
    start = time.time()
    print("Start KNN time (sec): {}".format(start))
    knn.fit(X_train,y_train)
    end = time.time()
    #score = knn.score(X_test, y_test)
    predictions = list(knn.predict(X_test))
    score = metrics.accuracy_score(y_test, predictions)
    print("Elapsed KNN time (sec): {}".format(round(end - start)))
    print("Model KNN score: {}".format(round(score,3)))
    skc = svm.SVC(max_iter=iterations)
    # Fit the classifier to the training data
    start = time.time()
    print("Start SVM time (sec): {}".format(start))
    skc.fit(X_train,y_train)
    # score = skc.score(X_test, y_test)
    predictions = list(skc.predict(X_test))
    score = metrics.accuracy_score(y_test, predictions)
    end = time.time()
    print("Elapsed SVM time (sec): {}".format(round(end - start)))
    print("Model SVM score: {}".format(round(score,3)))
    # test_tensorflow(filename, iterations*2)

def hyper_tune_knn(filename) :
    X, y = generate_model_data(filename + '.csv', 50000)
    # Test the models predication score. This is best done on smaller datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    n_space = list(range(1,3))
    param_grid = {'n_neighbors': n_space}
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print("KNN Score: {}".format(knn.score(X_test, y_test)))

    # Fit the classifier to the training data
    start = time.time()
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(X_train, y_train)
    end = time.time()
    print("Elapsed Tune time (sec): {}".format(round(end - start)))
    n = knn_cv.best_params_['n_neighbors']
    print("KNN Best n_neighbors: {}".format(n))
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    print("KNN Score with n_neighbors={} : {}".format(n ,knn.score(X_test, y_test)))

def build_model(filename) :
    X, y = generate_model_data(filename+'.csv')
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(X, y)
    dump(knn, filename+'.joblib')
    return knn

def check_answer(filename, answer, correct_answer, first_time = False) :
    if first_time :
        knn = build_model(filename+'.csv')

    knn = load('misspellings.joblib')
    score = fuzz.ratio(answer.lower(), correct_answer.lower()) / 100
    print("Checking: {} matches {} Fuzzy Score: {}".format(answer, correct_answer, score))
    prediction = knn.predict([convert_answer(answer.lower(), 24, score)])
    if prediction[0] == correct_answer :
        result = 'correct!'
    else :
        result = 'incorrect...'
    print("{} is {}".format(answer, result))

def get_answer() :
    answers = pd.read_csv('actors.csv')
    count = answers.shape[0]
    row_id = np.random.randint(0,count)
    return answers.values[row_id][0]

def main(first_time = False) :
    filename = 'misspellings'
    correct_answer = get_answer()
    prompt = correct_answer + " - Enter answer: "
    answer = input(prompt)

    start = time.time()
    check_answer(filename, answer, correct_answer, first_time)
    end = time.time()
    print("Elapsed time (sec): {}".format(round(end - start)))

#main()
compare_models('misspellings',50)
#test_tensorflow('misspellings', 100)
# hyper_tune_knn('misspellings')

X, y = generate_dummy_model_data('misspellings.csv', 100)
print("Dummies data shape: {}".format(X.shape))
print(X)
print(X)


