import re
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

class PUNCT_TFIFDVectorizer(TfidfVectorizer):
    NOPUNCTUATION = lambda sentence: ''.join([character for character in sentence if character not in punctuation])

    #build_analyzer is called on init
    def build_analyzer(self):
        #build_analyzer() of TfidfVectorizer load a method in memory and return a reference to it
        #That method acts on a sentence and breaks it down into words
        #Further those words would have a chance to get added into the vocabulary

        main_analyzer = TfidfVectorizer.build_analyzer(self)
        def custom_analyzer(sentence):
            sentence = PUNCT_TFIFDVectorizer.NOPUNCTUATION(sentence)
            return main_analyzer(sentence)

        return custom_analyzer

class SPAM_HAM_CLASSIFIER:
    def __init__(self, corpus):
        #load the corpus
        fh = open(corpus, 'r')
        #pre compile a regexp pattern for efficient execution
        pattern = re.compile('(.+)\t(.+)\n')

        self.labels = []
        self.messages = []
        #read the file content line by line
        for x in fh:
            #separate the label and the message
            match_obj = re.search(pattern, x)
            self.labels.append(match_obj.group(1))
            self.messages.append(match_obj.group(2))
        fh.close()

    def create_vocabulary(self):
        #create a vectorizer
        self.punct_tfidf = PUNCT_TFIFDVectorizer(stop_words='english')
        #learn a vocabulary from the messages
        self.punct_tfidf.fit(self.messages)
        print(self.punct_tfidf.get_feature_names())
        #bow
        bow = self.punct_tfidf.transform(self.messages)
        print(bow)


def main():
    try:
        sh_classifier = SPAM_HAM_CLASSIFIER('D:/batches/SE_PBL/SMSSpamCollection')
        sh_classifier.create_vocabulary()
        #... to be continued

    except:
        print('Error')
main()


