import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []


    # get all test sequences
    sequences = test_set.get_all_sequences()

    # extract the X and lengths
    for idx, sequence in sequences.items():
        X, lengths = test_set.get_item_Xlengths(idx)

        best = float('-inf')
        guessWord = ''

        # iterate over the trained models
        for word, model in models.items():
            try:
                probability = model.score(X, lengths)
            except:
                probability = float('-inf')

            # record the best option so far
            if (probability > best):
                best = probability
                guessWord = word
            
            # Append to probabilities
            probabilities.append( { word: probability })

        # Append our best guess word
        guesses.append(guessWord)

    return probabilities, guesses
