import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Init the values
        bestBicScore = float("inf")
        bestModel = None

        # For each model, calculate the BIC Score. 
        for idx in range(self.min_n_components + 1, self.max_n_components):
            try:
                selectedModel =  self.base_model(idx)
                logL = selectedModel.score(self.X, self.lengths)
                p = (idx**2) + 2 * selectedModel.n_features * idx
                currentBicScore = -2 * logL + p * np.log(len(self.X))

                # Save as the best model the one with smaller BIC Score
                if (currentBicScore < bestBicScore):
                    bestBicScore = currentBicScore
                    bestModel = selectedModel
            except:
                pass
        
        return bestModel

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Init the values
        bestDicScore = float("-inf")
        bestModel = None

        # For each model, calculate the DIC Score. 
        for idx in range(self.min_n_components + 1, self.max_n_components):
            try:
                selectedModel =  self.base_model(idx)
                logL = selectedModel.score(self.X, self.lengths)
                p = (idx**2) + 2 * selectedModel.n_features * idx


                # Remove the current word
                listWithoutWord = list(filter(lambda word: word != self.this_word, self.hwords))

                # calculate the score
                score = 0
                for word in listWithoutWord:
                    X, lengths = self.hwords[word]
                    score += selectedModel.score(X, lengths)

                # calculate the DIC score
                currentDicScore = logL - (score/ ( len(listWithoutWord) -1))
            
                # Save as the best model the one with bigger DIC Score
                if (currentDicScore > bestDicScore):
                    bestDicScore = currentDicScore
                    bestModel = selectedModel
            except:
                pass

        return bestModel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Init the values
        bestCVScore = float("-inf")

        # Set to 1 to get back a model if no improvement is found.
        bestModel = self.base_model(1) #self.min_n_components??
        scores= list()


        # split method limiting the splits up to 2 for speed.
        try:
            split_method = KFold(min(2,len(self.sequences)))
        except:
            return bestModel

        # For each model, calculate the CV Score. TODO: check min-max ids
        for idx in range(self.min_n_components, self.max_n_components+1):
            try:
                
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                
                    X, lengths = combine_sequences(cv_train_idx, self.sequences)
                    X2, lengths2 = combine_sequences(cv_test_idx, self.sequences)

                    
                    selectedModel = self.base_model(idx)
                    selectedModelF = self.base_model(idx).fit(X, lengths)

                    # Calculate the score (log likelihood)
                    score = selectedModelF.score(X2, lengths2)
                    scores.append(score)
                
            
                # Find the best score
                currentMean = np.mean(scores)

                if (currentMean > bestCVScore):
                    bestCVScore = currentMean
                    bestModel = selectedModel

            except:
                pass
        return bestModel
