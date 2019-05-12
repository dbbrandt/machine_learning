import numpy as np
import pandas as pd

def convert_answer( answer, max_size, score ) :
    "Convert a string to an array of ascii charactor ordinal values with fuzzy match score appended"
    converted = []
    for i in range(0, max_size) :
        if i < len(answer) :
            # print("{} {} ord: {}".format(i,  answer[i], ord(answer[i])))
            converted.append(float(ord(answer[i])))
        else :
            converted.append(32.0)
    converted.append(score * 100)
    return converted


def get_target_lookup(category_file):
    categories = pd.read_csv(category_file)
    categories['id'] = categories.index
    categories.set_index('correct_answer', inplace=True)
    return categories

def convert_target(target, category_file):
    print("Converting Target Data")
    categories = get_target_lookup(category_file)

    converted = []
    for i, x in enumerate(target):
        converted.append(categories.loc[x].id)
        if i % 10000 == 0:
            print(i)
    return converted

def generate_model_data(filename, category_file, row_count = 0) :
    if row_count == 0 :
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(filename, nrows = row_count)


    answers =  data['answer']
    correct_answers =  data['correct_answer']
    scores = data['score']
    lengths = data['length']
    target = np.array(convert_target(correct_answers, category_file))

    print("Converting Model Data")
    converted_answers = []
    for i, answer in enumerate(answers) :
        converted_answers.append(convert_answer(answer, 24, scores[i]))
        if i % 10000 == 0:
            print(i)
    predictors = np.array(converted_answers)

    print("Predictors shape: {}, target shape: {}".format(predictors.shape, target.shape))

    df = pd.DataFrame(predictors)
    df['target'] = target
    df['answer'] = answers
    df['correct_answer'] = correct_answers
    df['score'] = scores
    df['length'] = lengths
    # df['target_as'] = 0
    # mask = df['target'] == 0.0
    # df.loc[mask, 'target_as'] = 1

    return df

output_filename = 'data/misspellings_all_100_test'
seed_filename = 'data/actors.csv'
row_count = 0

df  = generate_model_data(output_filename + '.csv', seed_filename, row_count)

print('Count of names: {}'.format(df.groupby('correct_answer')['correct_answer'].count()))
df.to_csv(output_filename + '_dl.csv', index=False)
