import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TitanicChallenge:
    data_train_df = []
    data_test_df = []
    combined_data = []
    features = []
    categorical_features = []
    ordinal_features = []
    continues_features = []
    discrete_features = []
    numerical_features = []
    mixed_data_type_feature = None

    def get_data(self):
        self.data_train_df = pd.read_csv('data/train.csv', sep=',')
        self.data_test_df = pd.read_csv('data/test.csv', sep=',')

        # Combination of data is useful to run certain operations on both datasets together
        self.combined_data = [self.data_train_df, self.data_test_df]

    def analyse_by_describing_data(self):
        # print available features in datasets
        self.features = self.data_train_df.columns.values
        print(self.features)

        # visualize the data
        print(titanic.data_train_df.head())

        # print categorical features
        self.categorical_features = [ 'Survived', 'Sex', 'Embarked']
        self.ordinal_features = ['Pclass']

        print("Categorical features :", self.categorical_features)
        print("Ordinal features :", self.ordinal_features)

        # print numerical features
        self.continues_features = ['Age', 'Fare']
        self.discrete_features = ['SibSp', 'Parch']
        self.numerical_features = self.discrete_features + self.continues_features

        print("Continues features :", self.continues_features)
        print("Discrete features :", self.discrete_features)

        # mixed data types features
        self.mixed_data_type_feature = dict(
            Ticket='Alphanumeric',
            Cabin='mix of mumeric and alphanumeric'
        )

        print(self.mixed_data_type_feature)

        # types of contained data in each of train and test dataset
        print('train dataset types :')
        self.data_train_df.info()
        print('test dataset types :')
        self.data_test_df.info()

        # describe data of numerical features
        print('description of data by categorical features :')
        print(self.data_train_df[self.categorical_features].describe())

        # describe data of ordinal features
        print('description of data by ordinal features :')
        print(self.data_train_df[self.ordinal_features].describe())

        # describe data of numerical features
        print('description of data by numerical features :')
        print(self.data_train_df[self.numerical_features].describe())
        print(self.data_train_df.describe(include=['O']))

    def analyse_by_pivoting_data(self):

        # Survived by Pclass
        print(self.data_train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(
            by='Survived', ascending=False))

        # Survived by Age
        print(self.data_train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(
            by='Survived', ascending=False))

        # Survived by Sex
        print(self.data_train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(
            by='Survived', ascending=False))

        # Survived by Parch
        print(self.data_train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(
            by='Survived', ascending=False))

    def analyse_by_visualizing_data(self):
        # --------------------- Correlating numerical features -------------------
        # Survived by age
        survived_fg = sns.FacetGrid(self.data_train_df, col='Survived')
        survived_fg.map(plt.hist, 'Age', bins=20, color='r')

        # class by age
        survived_fg = sns.FacetGrid(self.data_train_df, col='Pclass')
        survived_fg.map(plt.hist, 'Age', bins=20, color='r')
        # -----> young passengers are mostly placed in Pclass 3

        # ---------------------- Correlating numerical and ordinal features --------------
        # Survived by Pclass
        survived_fg = sns.FacetGrid(self.data_train_df, col='Survived')
        survived_fg.map(plt.hist, 'Pclass', bins=10, color='b')
        # ----> passenger in class 3 mostly survived

        # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
        grid = sns.FacetGrid(self.data_train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.5, bins=20)
        grid.add_legend()

        # ---------------------- Correlating categorical features --------------------
        # grid = sns.FacetGrid(train_df, col='Embarked')
        grid = sns.FacetGrid(self.data_train_df, row='Embarked', size=2.2, aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
        grid.add_legend()

        # ---------------------- Correlating categorical and numerical features ----------
        # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
        grid = sns.FacetGrid(self.data_train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
        grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
        grid.add_legend()

        plt.show()

    # --------------------------- wrangle data ----------------------------
    def correcting_by_dropping_features(self):
        print("Before", self.data_train_df.shape, self.data_test_df.shape, self.combined_data[0].shape, self.combined_data[1])
        self.data_train_df = self.data_train_df.drop(['Ticket', 'Cabin'], axis=1)
        self.combined_data = [self.data_train_df, self.data_test_df]
        print("After", self.data_train_df.shape, self.data_test_df.shape, self.combined_data[0].shape, self.combined_data[1])

    def correcting_by_creating_new_features(self):
        # will be seen later
        pass

    def converting_categorical_features(self):
        # conversion of features which contain strings to numerical values is required by most model algorithms
        for dataset in self.combined_data:
            dataset['Sex'] = dataset['Sex'].map(
                {
                    'female': 1,
                    'male': 0
                }
            ).astype(int)
        print(self.data_train_df.head())

    def completing_a_numerical_continuous_feature(self):
        guess_ages = np.zeros((2, 3))
        for dataset in self.combined_data:
            for i in range(0, 2):
                for j in range(0, 3):
                    guess_df = dataset[(dataset['Sex'] == i) & \
                                       (dataset['Pclass'] == j + 1)]['Age'].dropna()

                    age_guess = guess_df.median()
                    # Convert random age float to nearest .5 age
                    guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

            for i in range(0, 2):
                for j in range(0, 3):
                    dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                                'Age'] = guess_ages[i, j]

            dataset['Age'] = dataset['Age'].astype(int)

        print(self.data_train_df.head())

        # create age brands and determine correlations with Survived
        # Bin age values into 5 discrete intervals
        self.data_train_df['AgeBrand'] = pd.cut(self.data_train_df['Age'], 5)
        # compute values of each Age mark, value of each mark is the mean of corresponding values
        self.data_train_df[['AgeBrand', 'Survived']].groupby(['AgeBrand'], as_index=False).mean().sort_values(
            by='AgeBrand', ascending=True)

        print(self.data_train_df)

        for dataset in self.combined_data:
            dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[dataset['Age'] > 64, 'Age'] = 4
        self.data_train_df.head()

        # drop AgeBrand feature
        self.data_train_df = self.data_train_df.drop(['AgeBrand'], axis=1)
        print(self.data_train_df.head())

    def creating_new_features_from_existing_ones(self):
        # create family size feature
        for dataset in self.combined_data:
            # "1" at the end represent the person in the row
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        self.data_train_df = self.combined_data[0]
        self.data_test_df = self.combined_data[1]

        familysize_survived = self.data_train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        print(familysize_survived)

        # add "is_alone" feature
        for dataset in self.combined_data:
            dataset['IsAlone'] = 0
            dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

        self.data_train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

        # drop 'Parch', 'SibSp', 'FamilySize' features
        train_df = self.data_train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        test_df = self.data_test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        combine = [train_df, test_df]

        train_df.head()

if __name__ == '__main__':

    titanic = TitanicChallenge()
    titanic.get_data()
    # titanic.analyse_by_describing_data()
    # titanic.analyse_by_pivoting_data()
    # titanic.analyse_by_visualizing_data()
    titanic.correcting_by_dropping_features()
    titanic.converting_categorical_features()
    titanic.completing_a_numerical_continuous_feature()
    titanic.creating_new_features_from_existing_ones()