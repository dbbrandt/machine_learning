import time
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ]


def misspell(sentence, threshold, minimum_score):
    "Generate a misspelled sentence base on the parameters"
    noisy_sentence = []
    score = 0
    while  score < minimum_score :
        i = 0
        while i < len(sentence):
            random = np.random.uniform(0, 1, 1)
            if random < threshold:
                noisy_sentence.append(sentence[i])
            else:
                new_random = np.random.uniform(0, 1, 1)
                if new_random > 0.67:
                    if i == (len(sentence) - 1):
                        continue
                    else:
                        noisy_sentence.append(sentence[i + 1])
                        noisy_sentence.append(sentence[i])
                        i += 1
                elif new_random < 0.33:
                    random_letter = np.random.choice(letters, 1)[0]
                    noisy_sentence.append(random_letter)
                    noisy_sentence.append(sentence[i])
                else:
                    pass
            i += 1
        result = ''
        for c in noisy_sentence :
            result += c
        score = fuzz.ratio(result, sentence) / 100
        noisy_sentence = []
    score = fuzz.ratio(result, sentence) / 100
    return [result, sentence, score]

def generate_samples(target, loops, threshold, minimum_score) :
    "Generate a number of misspellings of a string base on parameters"
    print("Generating samples for {}".format(target))
    dataset = []
    for i in range(0, loops):
        sample = misspell(target, threshold, minimum_score)
        dataset.append(sample)
        if i % 10000 == 0:
            print(i)
    return dataset

def generate_csv(import_data, target, loops, threshold, minimum_score) :

    np_dataset = np.array(generate_samples(target, loops, threshold, minimum_score))
    final_dataset  = np.unique(np_dataset[:,0])
    print("Generated {} samples for {}".format(final_dataset.shape[0], target))
    # print("Final: {} Candidates: {} Ratios: {}".format(final_dataset.shape[0], total, final_dataset.shape[0]/total))
    # import_data = pd.read_csv(filename)
    # print(import_data)
    export_data = pd.DataFrame(final_dataset)
    export_data.columns = ['answer']
    export_data['correct_answer'] = target
    export_data['score'] = np.vectorize(fuzz.ratio)(export_data['answer'],target) / 100
    final_data = import_data.append(export_data, ignore_index=True, sort=False)
    # final_data.to_csv(filename, index=False)
    return final_data

def main() :
    threshold = 0.85
    minimum_score = 0.90
    loops = 50000
    filename = "misspellings.csv"

    import_data = pd.read_csv('starting_data.csv')
    import_data['score'] = round(import_data['score'],3)
    #import_data.to_csv(filename, index=False)

    targets = np.unique(import_data['correct_answer'].values)
    print(targets)

    print("Starting file generation.")
    row_count = import_data.shape[0]
    for i, target in enumerate(targets) :
        start = time.time()
        import_data = generate_csv(import_data, target, loops, threshold, minimum_score)
        end = time.time()
        print("{}. Elapsed time (sec): {}".format(i, round(end - start)))

    import_data.to_csv(filename, index=False)
    print("Generated file: {} with {} rows".format(filename, import_data.shape[0]))


start = time.time()
main()
end = time.time()
print("Total Elapsed time (sec): {}".format(round(end - start)))
