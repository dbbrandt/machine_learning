This project inclues a set of Python scripts to generate machine learning test data.
The goals is product alternative spellings for the correct answer data provided.
With this we can train a model to recognize all variations of spelling as potentially 
correct answer.

This project includes three scripts whcih represent the three stages used to generate 
a keras model to be used for predicting if an answer is correct.

1. misspell.py
2. convert_data_for_ml.py
3. keras_fit.py

Details:
1. Misspellings
    * Provide the name of the seed_filename and the output_filename.
    * The seed file should have a single column labled correct_answer.    
    * The misspell.py generates an ouptput file in the format:
        * [ answer, correct_answer, score ]
    * Eachline in the output file represents a possible user answer that is within a defined fuzz.ratio of 
    the correct_answer
    * The script will generated the desired number of alternate answer per correct answer provided in the seed data 
    file. The seed data file must contain at least one named column, "correct_asnwer".
    * Configration:
        1. duplicates: this variable is the fraction (ex. 0.25) of the alternates that should be indentical to the seed.
        The purpose of this is to model the real world in which a significant number of answers are correct. Without 
        this duplicate factor, correct answer will not get a heavier weighting in the model.
        2. threshold: this can usually be left as is at 0.85. This is the ratio of letters in the answer that could 
        be chnaged. It drives the process of the random changes. Since the result still must be greater than the
        miinimum fuzz.ratio (minimum_score), this primarly help distributes the errors throughou the answer.
        3. minimum_score: The code uses the fuzzywuzzy fuzz.ratio to determine the distance from the answer and the 
        correct answer. This insures that answers are reasonable there are weightings within the misspell function
        that determine the ratio of the types of changes. This includes phoneme changes where users spell phonetically.
        4. max_loops: In order to fine an alternate answer but still be within the minimum_score, it can take several 
        misspelling attempts. This limits the number of total tries used to get to the target_count below. This prevents 
        difficult to misspell words for looping infinitely. For example very short answers.
        5. total_count: This is the desired number of alternatives per target answer.
    * Misspell detail configuration:
        1. Withing the actaull misspelling function there are five types of misspelling errors that are genreated. The 
        ratio of these erros can be configured. The four types and there default values are:
            * flip_letters = 0.75
            * random_phoneme = 0.60
            * random_add = 0.50
            * random_swap = 0.25    
            * drop = 0.0
        2. These values represent the random number distributions for these changes. The change type is made when
        the random number lies above the value for that type and below value for the adjacent type. These could be
        represented as raw percentages that must add to 1.0 then calculate the above ranges.
        
2. Convert the Data for input into the Keras model
    * Provide the output filename of the misspell.py step and a new file with _dl (ex. file_dl.csv) will be created
    which represented the string data for the alternate answer converted to numbers. Also include in the output are
    fields from the original data for later checking and validation.
    * The seed data filename must also be provided to calculate a numberic value to categorize the correct answers.
    This value is taken from the numeric index of the seed file data.  
    * To fit the data, it must be converted into numeric values.
    * This script converts letters into an array of number representing the ascii value of the letter.
    * Currently this assume all answers are no more than 24 characters and space pads the output.
    * TODO: check for the longest input and use that as the array size for the strings.
    
    
3. The Keras fit script (keris_fit.py)
    * This script is the most volitile and up to the user to modify to generate the best model output.
    * The final output of this should be an exported model file that can be used by another program as a predictor.
    * This script also provides several was to check the fit is good beyond the accuracy metric.
    * In real practice you want to insure that correct answers are predicted. This is not always the case for similar 
    or very short answers. This script predicts the seed data to insure that it is 100% predicted.
    * An additional array is provided to put in edge cases that result in undesirable predictions. These can be used
    when tuning the model to maximmise both the correct answer prediction percentate which should be 100% and the edge 
    case prediction which should be as close to 100% as possible.
    * Finally an interactive test is provided which can be commented in to try misspellings and see who well they are
    predicted. This can be used to add to the edge cases for latest automatic testing.
    * This script can be run with fit by calling main(filename, True, type) or without fit which reads in an existing 
    for validation (main(filename, False, type)). Both these can be commented out to run the interactive testing which 
    will also read in the model independently. 
    * Note: if running the fit be sure to set the console to emulate a terminal so that the fit output is more concise.
    * There is some idea of supporting multiple models which can be selected by setting the type variabe. This must be 
    coded into the main() to select the appropriate function to build and test the model. One draw back of this is
    that not all models support the same methods so for example the predict for xgboost and the predict_class used 
    for Keras binary_crossentropy are not the same. When the model is later used for prediction, this requies additonal 
    coding. This causes a problem in this script because prediction is used to do manuall testing. I've commented out
    the predict_class while testing the xgboost models (and vice versa would be required for binary_crossentropy).                  