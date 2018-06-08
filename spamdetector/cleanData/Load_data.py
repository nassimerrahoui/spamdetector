
class Load_data:
    def __init__(self, cleanData):
        self.cleanData = cleanData

    def data(self):
        return self.cleanData.getSpambase_data().drop('is_spam',1)

    def names(self):
        names = self.cleanData.getNames()
        del names[-1]
        return names

    def target(self):
        return self.cleanData.getSpambase_data()['is_spam']

