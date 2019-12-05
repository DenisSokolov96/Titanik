import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree


def Titanic():
    print("func titanic")
    sns.set_style('whitegrid')
    # get titanic & test csv files as a DataFrame
    titanic_df = pd.read_csv("datas/train.csv")

    # preview the data
    titanic_df.head()
    titanic_df.info()
    print("----------------------------")
    titanic_df['title'] = titanic_df['Name'].apply(get_title).apply(title_map)
    title_xt = pd.crosstab(titanic_df['title'], titanic_df['Survived'])
    # title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)

    # title_xt_pct.plot(kind='bar',
    #                  stacked=True,
    #                  title='Рейтинг выживания по названию')
    plt.xlabel('заглавие')
    plt.ylabel('Процент выживаемости')
    # plt.show()
    plt.figure()
    # iris = load_iris()
    # my_tree_one = DecisionTreeClassifier().fit(iris.data, iris.target)
    my_tree_one = DecisionTreeClassifier().fit(title_xt, title_xt)
    plot_tree(my_tree_one, filled=True)  # построить дерево с заполнением
    plt.show()


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'


def title_map(title):
    if title in ['Mr']:
        return 1
    elif title in ['Master']:
        return 3
    elif title in ['Ms', 'Mlle', 'Miss']:
        return 4
    elif title in ['Mme', 'Mrs']:
        return 5
    else:
        return 2


def Titanic2():
    train = pd.read_csv('datas/train.csv')

    # impute number values and missing values
    train["Embarked"] = train["Embarked"].fillna("S")
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Pclass"] = train["Pclass"].fillna(train["Pclass"].median())
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())

    train['title'] = train['Name'].apply(get_title).apply(title_map)
    #title_xt = pd.crosstab(train['title'], train['Survived'])
    title_xt = pd.crosstab(train['title'],train["Age"])
    title_yt = pd.crosstab(train['title'],train["Pclass"])

    plt.figure("Дерево")
    my_tree_one = DecisionTreeClassifier().fit(title_xt, title_yt )
    plot_tree(my_tree_one, filled=True)  # построить дерево с заполнением
    plt.show()


if __name__ == '__main__':
    Titanic2()
