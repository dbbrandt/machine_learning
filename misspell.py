import time
import numpy as np
import pandas as pd
import random
from fuzzywuzzy import fuzz

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ]

phonemes = [[1,['b', 'bb']],
    [2,['d', 'dd', 'ed']],
    [3,['f','ff','ph','gh','lf','ft']],
    [4,['g','gg','gh','gu','gue']],
    [5,['h','wh']],
    [6,['j','ge','g','dge','di','gg']],
    [7,['k','c','ch','cc','lk','qu'',q(u)','ck','x']],
    [8,['l','ll']],
    [9,['m','mm','mb','mn','lm']],
    [10,['n','nn','kn','gn','pn']],
    [11,['p','pp']],
    [12,['r','rr','wr','rh']],
    [13,['s','ss','c','sc','ps','st','ce','se']],
    [14,['t','tt','th','ed']],
    [15,['v','f','ph','ve']],
    [16,['w','wh','u','o']],
    [17,['z','zz','s','ss','x','ze','se']],
    [18,['s','si','z']],
    [19,['ch','tch','tu','ti','te']],
    [20,['sh','ce','s','ci','si','ch','sci','ti']],
    [21,['ng','n','ngue']],
    [22,['y','i','j']],
    [23,['a','ai','au']],
    [24,['a','ai','eigh','aigh','ay','er','et','ei','au','ea','ey']],
    [25,['e','ea','u','ie','ai','a','eo','ei','ae']],
    [26,['e','ee','ea','y','ey','oe','ie','i','ei','eo','ay']],
    [27,['i','e','o','u','ui','y','ie']],
    [28,['i','y','igh','ie','uy','ye','ai','is','eigh']],
    [29,['a','ho','au','aw','ough']],
    [30,['o','oa','oe','ow','ough','eau','oo','ew']],
    [31,['o','oo','u','ou']],
    [32,['u','o','oo','ou']],
    [33,['o','oo','ew','ue','oe','ough','ui','oew','ou']],
    [34,['oi','oy','uoy']],
    [35,['ow','ou','ough']],
    [36,['a','er','i','ar','our','ur']],
    [37,['air','are','ear','ere','eir','ayer']],
    [38,['ir','er','ur','ear','or','our','yr']],
    [39,['aw','a','or','oor','ore','oar','our','augh','ar','ough','au']],
    [40,['ear','eer','ere','ier']],
    [41,['ure','our']]]

def flatten_3D(array):
    "Returns the phoneme list as a dataframe one per row"
    flat = []
    for val in array:
        for p in val[1]:
            flat.append([val[0],p])
    df = pd.DataFrame(flat)
    df.columns = ['id','value']
    return df

flat_phonemes = flatten_3D(phonemes)

def switch_phoneme(input_value, df=flat_phonemes):
    "Returns a random phoneme from any set that includes the input value"
    swap = input_value
    found = df[df['value'] == input_value]
    if len(found) > 0:
        # Get all the ids (phoneme sets) with the input value
        found_ids = found['id'].values
        # Get all related phonemes
        related_rows = df[df['id'].isin(found_ids)]
        alt_phonemes = related_rows['value'].values
        alt_list = np.unique(alt_phonemes).tolist()
        alt_list.remove(input_value)
        #print("Alternates for {} are {}".format(input_value, alt_phonemes))
        swap = np.random.choice(alt_list,1)[0]
    #print("Phone swap {} for {}:".format(input_value,swap))
    return swap

def misspell(sentence, threshold, minimum_score, duplicate, max_loops, loops, answers):
    "Generate a misspelled sentence base on the parameters"
    # Ranges of transformation where
    flip_letters = 0.75
    random_phoneme = 0.60
    random_add = 0.50
    random_swap = 0.25
    noisy_sentence = []
    score = 0

    #
    random = np.random.uniform(0, 1, 1)
    # Approximately half of the non-duplicates end getting no change based on testing
    #

    if random[0] > duplicate:
        # Only accept misspellings that have a fuzzy match above the minimum score
        while  score < minimum_score :
            i = 0
            while i < len(sentence):
                random = np.random.uniform(0, 1, 1)
                # No change
                if random[0] < threshold:
                    #print("No change")
                    noisy_sentence.append(sentence[i])
                else:
                    # Varous changes based on probability
                    new_random = np.random.uniform(0, 1, 1)
                    if new_random > flip_letters:
                        if i == (len(sentence) - 1):
                            continue
                        else:
                            #print("Flip letters {}".format(sentence[i]))
                            noisy_sentence.append(sentence[i + 1])
                            noisy_sentence.append(sentence[i])
                            i += 1
                    # Swap in phoneme
                    elif new_random > random_phoneme:
                        #print("Swap phoneme {}".format(sentence[i]))
                        noisy_sentence.append(switch_phoneme(sentence[i]))
                    # Add in random leter
                    elif new_random > random_add:
                        #print("Random add letter {}".format(sentence[i]))
                        random_letter = np.random.choice(letters, 1)[0]
                        noisy_sentence.append(random_letter)
                        noisy_sentence.append(sentence[i])
                    # Swap in random letter
                    elif new_random > random_swap:
                        #print("Random swap letter {}".format(sentence[i]))
                        random_letter = np.random.choice(letters, 1)[0]
                        noisy_sentence.append(random_letter)
                    # Remove a letter
                    else:
                        #print("Drop letter {}".format(sentence[i]))
                        pass
                i += 1
            result = ''
            for c in noisy_sentence :
                result += c
            score = fuzz.ratio(result, sentence) / 100
            # Ignore duplicates
            if result in answers:
                score = 0
            #print("Score attempt {} : {}".format(result, score))
            loops += 1
            if loops % 1000 == 0:
                print("{} - {}".format(loops, len(answers)))
            if loops > max_loops:
                break
            noisy_sentence = []
    else:
        result = sentence

    answers.append(result)
    score = fuzz.ratio(result, sentence) / 100
    length = len(result)
    return [result, sentence, score, length], loops, answers

def generate_samples(target, target_count, max_loops, threshold, minimum_score, duplicate) :
    "Generate a number of misspellings of a string base on parameters"
    print("Generating samples for {}".format(target))
    dataset = []
    loops = 0
    answers = []
    while loops < max_loops:
        loops += 1
        sample, loops, answers = misspell(target, threshold, minimum_score, duplicate, max_loops, loops,  answers)
        dataset.append(sample)
        if loops % 1000 == 0:
            print("{} - {}".format(loops,len(dataset)))
        if len(dataset) >= target_count:
            break
    return dataset

def generate_csv(target, target_count, max_loops, threshold, minimum_score, duplicate) :

    np_dataset = np.array(generate_samples(target, target_count, max_loops, threshold, minimum_score, duplicate))
    print("Generated {} samples for {}".format(len(np_dataset), target))
    final_data = pd.DataFrame(np_dataset)
    final_data.columns = ['answer','correct_answer','score','length']
    # final_data['correct_answer'] = target
    # final_data['score'] = np.vectorize(fuzz.ratio)(final_dataset['answer'],target) / 100
    return final_data

def main() :
    seed_data = pd.read_csv(seed_filename)
    targets = seed_data['correct_answer'].values

    print("Starting file generation.")
    row_count = seed_data.shape[0]
    print("Creating data for {} seeds".format(row_count))
    header = True
    for i, target in enumerate(targets):
        start = time.time()
        final_data = generate_csv(target, target_count, max_loops, threshold, minimum_score, duplicate)
        end = time.time()
        if i ==  0:
            header = False
            final_data.to_csv(output_filename, index=False)
        else:
            with open(output_filename, 'a') as f:
                final_data.to_csv(f, index=False, header=False)
        print("Seed #{}. Elapsed time (sec): {}".format(i, round(end - start)))

    export_data = pd.read_csv(output_filename)
    print("Generated file: {} with {} rows".format(output_filename, export_data.shape[0]))
    dups = export_data[export_data['answer'].isin(targets)]
    pct = len(dups) / len(export_data) * 100
    print("Duplicate percentage: {}".format(pct))


start = time.time()
# Traingin Data Creation parmaeters
duplicate = 0.25  # The percent of data that should be duplicates of the original
threshold = 0.85  # The percent of times a letter change should be attempted in a sentence
minimum_score = 0.90 # The minimum fuzz.ratio required to be accepted as a valid training sentence
max_loops = 200000  # The maximum number of loops per target to attempt to reach the target count.
target_count = 100  # THe number of samples to generate per target

output_filename = "data/misspellings_all_100_test.csv"
seed_filename = 'data/actors.csv'
main()
end = time.time()
print("Total Elapsed time (min): {}".format(round(end - start)/60))

