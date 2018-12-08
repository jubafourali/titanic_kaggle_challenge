import pandas as pd


class TitanicChallenge:
    data_train_df = []
    data_test_df = []
    combined_data = []

    def get_data(self):
        self.data_train_df = pd.read_csv('data/train.csv', sep=',')
        self.data_test_df = pd.read_csv('data/test.csv', sep=',')

        # Combination of data is useful to run certain operations on both datasets together
        self.combined_data = [self.data_train_df, self.data_test_df]


if __name__ == '__main__':

    titanic = TitanicChallenge()
    titanic.get_data()
