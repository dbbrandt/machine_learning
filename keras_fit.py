import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import optimizers
from numpy.random import seed
from tensorflow import set_random_seed
from xgboost import XGBClassifier
import joblib

def convert_answer( answer, correct_answer, max_size=24) :
    "Convert a string to an array of ascii charactor ordinal values with fuzzy match score appended"
    converted = []
    score = fuzz.ratio(answer, correct_answer)
    for i in range(0, max_size) :
        if i < len(answer) :
            # print("{} {} ord: {}".format(i,  answer[i], ord(answer[i])))
            converted.append(float(ord(answer[i])))
        else :
            converted.append(32.0)
    converted.append(score)
    converted.append(len(answer))
    return converted, score

def generate_model(predictors, target, base_model = Sequential()):
    n_cols = predictors.shape[1]
    print("Predictor columns: {}".format(predictors.columns))
    print('count: {}',format(n_cols))
    binary_target = to_categorical(target)
    model = Sequential()
    model.add(Dense(80, activation='relu', input_shape = (n_cols,)))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(len(binary_target[0]), activation='sigmoid'))
    model.set_weights(base_model.get_weights())

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
#                 optimizer= 'adam',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=3)

    model.fit(predictors, binary_target, callbacks=[early_stopping_monitor], validation_split=0.3, epochs=1, verbose=1)
    return model

def generate_boost_model(predictors, target):
    X_train, X_test, y_train, y_test = train_test_split( predictors, target, test_size=0.33, random_state=53)
    model = XGBClassifier(scale_pos_weight=1,
                          learning_rate=0.1,
                          objective='multi:softmax',
                          subsample=0.8,
                          min_child_weight=6,
                          n_estimators=100,
                          max_depth=4,
                          gamma=1)

    # model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=3, verbose=True)
    return model

def validate(predictors, target, model):
    print("Predictor Columns: {}".format(predictors.columns))
    #pred = model.predict_classes(predictors)
    pred = model.predict(predictors)
    score = metrics.accuracy_score(target, pred)
    return score

def validate_set(model, categories, test_cases):
    count = 0
    total = len(test_cases)
    for test in test_cases:
        name = test[0]
        correct_name = test[1]
        x, score = convert_answer(name, correct_name)
        X = pd.DataFrame([x])
        X = X.rename(columns={25:'length'})
        # result = model.predict_classes(X)
        result = model.predict(X)
        pred = categories.iloc[result[0]][0]
        correct = (correct_name == pred)
        if correct:
            count += 1
        else:
            print(
                'Testing entry {} for {} -- Predicted {} - prediction is {} - score: {}'.format(name,
                                                                                    correct_name, pred, correct,score))

    print('Correct: {} Total: {}'.format(count, total))
    pct = round((count / total) * 100, 2)
    print("Percent Correct: {}%".format(pct))

def test_manual_data(model_filename, type, category_filename, test_cases):
    if type == 'boost':
        filename = model_filename + '.xgb'
        model = joblib.load(filename)
    else:
        filename = model_filename + '.h5'
        model = load_model(filename)
    categories = pd.read_csv(category_filename+'.csv')
    test_data = categories.copy()
    test_data['answer'] = categories['correct_answer']
    validate_set(model, categories, test_data.values)
    validate_set(model, categories, test_cases)

def interactive_test(model_filename, type, category_filename):
    if type == 'boost':
        filename = model_filename + '.xgb'
        model = joblib.load(filename)
    else:
        filename = model_filename + '.h5'
        model = load_model(filename)

    categories = pd.read_csv(category_filename+'.csv')
    for i in range(0,100):
        correct_name = categories.loc[int(np.random.uniform(0, 322, 1))][0]
        print('Enter misspelling of actor: {}'.format(correct_name))
        name = input('Actor''s Name? ')
        x, score = convert_answer(name, correct_name)
        X = pd.DataFrame([x])
        X = X.rename(columns={25:'length'})
        print('score: {}'.format(X[24].values))
        #result = model.predict_classes(X)
        result = model.predict(X)
        print("Predicted {}".format(categories.iloc[result[0]]))

def main(filename, train = False, type = 'standard'):
    # Insure that random seeds stay the same for repeatability
    seed(72) # Python
    set_random_seed(72) # Tensorflow

    # For retraining
    # base_filename = 'misspellings_all_dl_5000_train-70-70-sig-adam-custom'
    # base_model = read_model(base_filename+'.h5')

    df = pd.read_csv(filename+'.csv')
    predictors = df.drop(['target','answer', 'correct_answer', 'score'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split( predictors, df['target'], test_size=0.33, random_state=53)

    if train:
        if type == 'boost':
           model = generate_boost_model(X_train, y_train)
           joblib.dump(model, filename + '.xgb')
           # model.save_model(filename + '.xgb')
        else:
           model = generate_model(X_train, y_train)
           model.save(filename + '.h5')
    else:
        if type == 'boost':
            #model.load_model(filename+'.xgb')
            model = joblib.load(filename + '.xgb')
            print(model)
            # y_pred = model.predict(X_test)
            # predictions = [round(value) for value in y_pred]
            # score = metrics.accuracy_score(y_test, predictions)

        else:
            model = load_model(filename+'.h5')
            print(model.summary())

    score = validate(X_test, y_test, model)
    print("Score: {}".format(score))


# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3

type = 'boost'
# filename = 'data/misspellings_all_medium_dl'
filename = 'data/predict_actor'
# main(filename, True, type)
# main(filename, False, type)

category_filename = 'data/actors'

test_cases = [['Amie Adams', 'Amy Adams'],
              ['Michael Fox', 'Michael J. Fox'],
              ['Minny Driver', 'Minnie Driver'],
              ['BLair Underwood', 'Blair Underwood'],
              ['Ralph Finnes', 'Ralph Fiennes'],
              ['Kate Blanchette', 'Cate Blanchett'],
              ['Joakin Pheonix', 'Joaquin Phoenix'],
              ['Ane Hathaway', 'Anne Hathaway'],
              ['Mickey Rorke', 'Mickey Rourke'],
              ['Collin Farrell', 'Colin Farrell'],
              ['Ben Stiler', 'Ben Stiller'],
              ['Cate Winslet', 'Kate Winslet'],
              ['John Hawks', 'John Hawkes'],
              ['George Cloney', 'George Clooney'],
              ['Cathlene Turner','Kathleen Turner'],
              ['Mathew Broderick', 'Matthew Broderick'],
              ['Mat Damon', 'Matt Damon'],
              ['Jennifer Jason Lee', 'Jennifer Jason Leigh'],
              ['Peter Otolle', "Peter O'toole"],
              ['John C Reily', 'John C. Reilly']
             ]

test_manual_data(filename, type, category_filename, test_cases)

# interactive_test(filename, type, category_filename)
