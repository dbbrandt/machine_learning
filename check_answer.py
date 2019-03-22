import time
from joblib import dump, load
import numpy as np
import scipy as sp
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def convert_answer( answer, max_size, score ) :
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
    answer_array = value[:-1]
    answer = ''
    for c in answer_array :
        answer += chr(int(c))
    return answer

def generate_model_data(filename, reload = False) :
    cached_filename = filename+'.cached'
    if reload == True :
        data = pd.read_csv(filename)

        answers =  data['answer'].apply(str).apply(str.lower)
        score = data['score']

        y =  data['correct_answer']

        converted_answers = []
        for i, answer in enumerate(answers) :
            converted_answers.append(convert_answer(answer, 24, score[i]))

        X = np.array(converted_answers)
        print("Original X shape: {}, y shape: {}".format(X.shape, y.shape))
        cached_data = pd.DataFrame(X)
        cached_data['correct_answer'] = y
        print("Cached data shape: {}".format(cached_data.shape))
        cached_data.to_csv(cached_filename, index=False)
    else :
        data = pd.read_csv(cached_filename)
        print("Cached data shape: {}".format(data.shape))
        y = data['correct_answer']
        X = data.drop('correct_answer',1)
        print("Cached X shape: {}, y shape: {}".format(X.shape, y.shape))

    print(X.shape)
    # print(data.shape)

    return X, y

def build_model(filename, reload) :
    X, y = generate_model_data(filename, reload)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors = 10)

    # Fit the classifier to the training data
    # knn.fit(X_train,y_train)

    # print("Model score: {}".format(round(knn.score(X_test, y_test)),3))

    knn.fit(X, y)
    dump(knn, filename+'.joblib')
    return knn

def check_answer(answer, correct_answer) :
    # knn = build_model('misspellings.csv', reload=False)
    knn = load('misspellings.csv.joblib')
    score = fuzz.ratio(answer, correct_answer.lower()) / 100
    print("Checking: {} matches {} Fuzzy Score: {}".format(answer, correct_answer, score))
    prediction = knn.predict([convert_answer(answer, 24, score)])
    if prediction[0] == correct_answer :
        result = 'correct!'
    else :
        result = 'incorrect...'
    print("{} is {}".format(answer, result))

answer = 'maggie gyllenhal'
correct_answer = 'Maggie Gyllenhaal'
start = time.time()
check_answer(answer, correct_answer)
end = time.time()
print("Elapsed time (sec): {}".format(round(end - start)))