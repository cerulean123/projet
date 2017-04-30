from sklearn.base import BaseEstimator
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest

class Classifier(BaseEstimator):
    def __init__(self):
        et = ExtraTreesClassifier(n_estimators=200)
 	gb = GradientBoostingClassifier(n_estimators=200)
	ab = AdaBoostClassifier()
	voting = VotingClassifier(estimators=[('et', et), ('gb', gb), ('ab', ab)], voting='soft')
	variance = VarianceThreshold()
	normalize = Normalizer()
	select = SelectKBest(k=1837)
	self.clf = Pipeline([("normalize", normalize),("variance", variance),("select", select),("voting", voting)]) 

    def fit(self, X, y):
	self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
