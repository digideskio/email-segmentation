from __future__ import print_function
from __future__ import division
import email
import os
from os import listdir
from os.path import join, isdir
import re
import numpy as np
from gensim import corpora, models
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets.base import Bunch
from sklearn.svm import SVC
from sklearn.externals import joblib
from spacy.en import English, LOCAL_DATA_DIR

data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(data_dir=data_dir)


class EmailSegmentation(object):

    def __init__(self, num_topics=None):

        self.categories = ['sig', 'reply', 'other']
        self.num_feats = 2
        if num_topics is None:
            self.num_topics = 20
        else:
            self.num_topics = num_topics


    def remove_tuples(self, tuples_list, get_index=1):
        """
        Removes the tuple structure and returns only 2nd tuple items as a list
        """
        tupleless = [tup[get_index] for tup in tuples_list]
        return tupleless


    def get_punct(self, punctuation, sentence):
        """
        Find punctuation in a sentence
        """
        puncts = []
        for c in punctuation:
            c = re.compile(str("\\" + c))
            if re.search(c, sentence) is not None:
                puncts.append("PUNCT")
        return puncts

    def get_pos(self, sentences):
        """
        Returns the part-of-speech tags as a list for each sentence
        """
        punctuation = """.,?!:;"""
        tags_and_puncts = []
        tags_lists = []
        puncts_lists = [self.get_punct(punctuation, sent) for sent in sentences]

        for sent in sentences:
            tags_lists.append([token.tag_ for token in nlp(sent)])
        for tags, puncts in zip(tags_lists, puncts_lists):
            tags_and_puncts.append(tags+puncts)
        return tags_and_puncts


    def feats(self, sentence):
        """
        Some feature designing, two features being extracted here for
        a given sentence.
        """
        feat_vec = np.zeros(self.num_feats)
        tokens = nlp(sentence) 
        tags = [token.tag_ for token in tokens]
        
        if len(tokens) > 0:
            feat_vec[0] = tags.count('NNP')/len(tags)

        if sentence.startswith('>' or ':' or '>>'):
            feat_vec[1] = 1
        return feat_vec


    def extract_feats(self, sentences):
        '''
        Returns extracted features for a list of sentences
        '''
        feat_vecs = map(self.feats, sentences)
        return feat_vecs


    def LSI_fit(self, data):
        '''
        Fits an LSI model and returns it with associated dictionary
        '''
        texts = [[tag for tag in sent] for sent in self.get_pos(data)]
        dictionary = corpora.Dictionary(texts)
        texts = map(dictionary.doc2bow, texts)
        lsi = models.LsiModel(texts, id2word=dictionary, 
                                                    num_topics=self.num_topics)

        return dictionary, lsi

    
    def LSI_transform(self, data, dictionary, lsi):
        """
        Transforms input data using LSA/LSI, currently using  bag-of-pos-tags
        """

        texts = [[tag for tag in sent] for sent in self.get_pos(data)]
        texts = map(dictionary.doc2bow, texts) 
        lsi_vectors = map(self.remove_tuples, lsi[texts])

        return lsi_vectors


    def vec_augmentation(self, lsi_vecs, feat_vecs, feat_flag=True):
        """
        Augments existing LSI vector by concatenating horizontally with other 
        manually designed feature vector.
        """
        hstacked_vecs = np.zeros((len(lsi_vecs),self.num_topics+self.num_feats))

        for i,v in enumerate(lsi_vecs):
            for j,u in enumerate(v):
                hstacked_vecs[i][j] = u

        if feat_flag:
            for i,_ in enumerate(lsi_vecs):
                hstacked_vecs[i][-self.num_feats:] = feat_vecs[i][0:self.num_feats]

        return hstacked_vecs

    def train_segmenter(self, data, targets, target_names, test=True):
        '''
        Trains a support vector machines classifier and returns the 
        trained model and test report if test flag was on.
        '''
        
        X_train, X_test, y_train, y_test= train_test_split(data, 
                                targets, test_size=0.2, random_state=42)
        svc = SVC(probability=True)
        if test:
            clf = svc.fit(X_train, y_train)
            pred= clf.predict(X_test)
            report = classification_report(y_test, pred,
                                    target_names=target_names)
            return clf, report
        else:
            clf = svc.fit(data, targets)
            return clf

    def save_models(self, dictionary, LSI, clf, dir_path):
        '''
        Saves all trained models in given directory 
        '''
        _ = joblib.dump(dictionary, dir_path+'dict.pkl', compress=9)
        _ = joblib.dump(LSI, dir_path+'lsi.pkl', compress=9)
        _ = joblib.dump(clf, dir_path+'clf.pkl', compress=9)

    def load_models(self, dir_path):
        '''
        Loads all the pre-trained models from given models directory
        '''
        dictionary = joblib.load(dir_path+'dict.pkl')
        LSI = joblib.load(dir_path+'lsi.pkl')
        clf = joblib.load(dir_path+'clf.pkl')

        return dictionary, LSI, clf

    def get_segmented(self, dataset, clf, dictionary, LSI):
        '''
        Predicts segment types for a test samples
        '''
        vecs1 = self.LSI_transform(dataset.data, dictionary, LSI)
        vecs2 = self.extract_feats(dataset.data)
        vecs = self.vec_augmentation(vecs1, vecs2)
        preds = clf.predict(vecs)
        preds_probs = clf.predict_proba(vecs)
        report = classification_report(dataset.target, preds,
                                    target_names=dataset.target_names)
        return [(self.categories[pred],sent) for pred, sent in 
                                    zip(preds,dataset.data)], report

    
    def sample_segmented(self, data, clf, dictionary, LSI):
        '''
        Predicts segment types for a single test sample
        '''
        data = map(lambda dd: unicode(dd, errors='ignore'), data)
        vecs1 = self.LSI_transform(data, dictionary, LSI)
        vecs2 = self.extract_feats(data)
        vecs = self.vec_augmentation(vecs1, vecs2)
        preds = clf.predict(vecs)
        preds_probs = clf.predict_proba(vecs)

        return [(self.categories[pred],sent) for pred, sent in 
                                    zip(preds, data)]



    def load_files(self, container_path):
        """
        Each file is mail in EML format and is labeled using 3 classes.

        ##reply## - reply lines
        ##sig## - signature lines
        other - all other lines are just typical email lines

        Returns
        -------
        data : Bunch
            Dictionary-like object
        """

        categories = ['sig', 'reply', 'other']
        data = []
        target = []
        filenames = []
        prev_data = []
        next_data = []
        files = [f for f in sorted(listdir(container_path)) 
                                    if isdir(join(container_path))]

        for filename in files:
            with open(join(container_path, filename), 'rb') as f:

                msg = f.read()
                metadata = msg.split('\n\n')[0]
                msg = msg.split('\n\n')[1:]
                msg = '\n\n'.join(msg).splitlines()

                for i,l in enumerate(msg):


                    if l.startswith('#sig#'):
                        l = l.replace('#sig#', '')
                        if re.search('[a-zA-Z]', l) != None:
                            l = re.sub('[^a-zA-Z]',' ',l)
                            data.append(l)
                            target.append(categories.index('sig'))
                            filenames.append(filename)
                            prev_data.append('\n'.join(msg[i-1:i]))

                    elif l.startswith('#reply#'):
                        l = l.replace('#reply#', '')
                        data.append(l)
                        target.append(categories.index('reply'))
                        filenames.append(filename)
                        prev_data.append('\n'.join(msg[i-1:i]))

                    else:
                        data.append(l)
                        target.append(categories.index('other'))
                        filenames.append(filename)
                        prev_data.append('\n'.join(msg[i-1:i]))

              

        # Ignoring possible unicode errors          
        data = map(lambda dd: unicode(dd, errors='ignore'), data)
        prev_data = map(lambda dd: unicode(dd, errors='ignore'), prev_data)

        return Bunch(
                     data=data, 
                     target=np.array(target), 
                     filenames=filenames, 
                     target_names=categories, 
                     prev_data=prev_data
                     )



def main():

    load_from_disk = False
    
    if load_from_disk:
        # Loading already trained models and testing 
        segmenter = EmailSegmentation()
        dictionary, LSI, clf = segmenter.load_models('models/')
        tests = segmenter.load_files('datasets/sigPlusReply')
        segs, report = segmenter.get_segmented(tests, clf, dictionary, LSI)
        print(report)

    else:
        # Loading dataset and training from scratch 

        lsi_topics = 20
        segmenter = EmailSegmentation(lsi_topics)

        dataset = segmenter.load_files('datasets/sigPlusReply')
        dictionary, LSI = segmenter.LSI_fit(dataset.data)
        data_lsi_vectors = segmenter.LSI_transform(dataset.data, dictionary, LSI) 
        data_feat_vectors = segmenter.extract_feats(dataset.data)
        data_aug_vectors = segmenter.vec_augmentation(data_lsi_vectors, data_feat_vectors)
        clf, report = segmenter.train_segmenter(data_aug_vectors, dataset.target, 
                                                    dataset.target_names, test=True)
        segmenter.save_models(dictionary, LSI, clf, 'models/')
        print(report)




if __name__ == '__main__':
    main()
