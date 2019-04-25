import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

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
    return converted, score

def get_target_lookup(category_file):
    categories = pd.read_csv(category_file)
    categories['id'] = categories.index
    categories.set_index('correct_answer', inplace=True)
    return categories

def get_data(filename):
    dfraw = pd.read_csv(filename)

    df = dfraw[0:-100]

    dftest = dfraw[99001:]
    dftest_x = dftest.drop('target', axis=1)
    dftest_y = to_categorical(dftest.target)
    return df, dftest_x, dftest_y

def generate_model(filename):
    df = pd.read_csv(filename)

    predictors = df.drop(['target','answer', 'correct_answer', 'score', 'target_as'], axis=1)
    # target = to_categorical(df.target_as)
    target = df.target_as
    n_cols = predictors.shape[1]
    print("Predictor Columns: {}".format(n_cols))
    # print(df.target.iloc[0])
    # print(target)
    # print(np.argmax(target[0]))
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape = (n_cols,)))
    model.add(Dense(20, activation='relu'))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=3)

    # model.fit(predictors, target, callbacks=[early_stopping_monitor], validation_split=0.3, epochs=20, verbose=1)
    model.fit(predictors, target, callbacks=[early_stopping_monitor], epochs=20, verbose=1)
    # model.fit(predictors, target, validation_split=0.1, epochs=20)
    # print(model.summary())
    model.save(filename+'.h5')
    return model

def read_model(filename):
    model = load_model(filename)
    return model

def validate(filename, correct=False):
    df = pd.read_csv(test_filename)

    incorrect = 0
    for index, row in df.iterrows():
        answer = row.answer
        correct_answer = row.correct_answer
        x, score = convert_answer(answer, correct_answer)
        # print('Converted Answer: {}'.format(x))
        X = pd.DataFrame([x])
        results = model.predict_classes(pd.DataFrame(X))
        # print("Prediction Raw: Shape:{} value: {}".format(results.shape, results))
        # prediction = np.argmax(results, axis=1)
        # prob = results[0][1] * 100
        # pred = prediction[0]
        pred = results[0][0]

        if pred == 0:
            incorrect += 1
        #     if correct:
        #         print("{}. Answer: {} - prob: {} - score: {}".format(pred, answer, round(prob, 2), score))
        # else:
        #     if not correct:
        #         print("{}. Answer: {} - prob: {} - score: {}".format(pred, answer, round(prob, 2), score))
        if index % 1000 == 0:
            print(index)

    total = df.shape[0]
    correct_pct = ((total - incorrect) / total) * 100
    print("Total: {} - Incorrect: {} - pct: {}".format(total, incorrect, correct_pct))
    # print(categories.iloc[prediction[0]])

# filename = 'misspellings_dl'
# filename = 'adam_sandler_other'
# filename = 'adam_sandler_other_30_0.1.h5'
filename = 'adam_sandler_bin_dl'

target_file = 'actors.csv'
# model = read_model(filename+'_20.h5')

model = generate_model(filename+'.csv')

categories = get_target_lookup(target_file)

# x = convert_answer('Adam Sandler','Adam Sandler')
test_filename = 'misspellings_adam_sandler.csv'
validate(test_filename, True)

test_filename = 'misspellings_as_test.csv'
validate(test_filename)




# 65.0,100.0,97.0,109.0,32.0,83.0,97.0,110.0,100.0,108.0,101.0,114.0,32.0,32.0,32.0,32.0,32.0,32.0,32.0,32.0,32.0,32.0,32.0,32.0,100.0,0.0,Adam Sandler,Adam Sandler,1.0,1.0