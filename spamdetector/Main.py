
from spamdetector.cleanData.CleanData import CleanData
from spamdetector.algorithme.NaiveBayes import NaiveBayes

if __name__ == '__main__':
    clean_data = CleanData()
    naiveBayes = NaiveBayes(clean_data.getDf(), clean_data.getNames())
    naiveBayes.detailed_result()