import pandas as pd
import matplotlib.pyplot as plt
from pygments.formatters import img
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz


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


def Titanic():
    train = pd.read_csv('datas/train.csv')

    # impute number values and missing values
    train["Embarked"] = train["Embarked"].fillna("S")
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Pclass"] = train["Pclass"].fillna(train["Pclass"].median())
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())

    train['title'] = train['Name'].apply(get_title).apply(title_map)
    # title_xt = pd.crosstab(train['title'], train['Survived'])
    title_xt = pd.crosstab(train['title'], train["Age"])
    title_yt = pd.crosstab(train['title'], train["Pclass"])

    plt.figure("Дерево")
    my_tree_one = DecisionTreeClassifier( max_depth = 4).fit(title_xt, title_yt)
    plot_tree(my_tree_one, filled=True)  # построить дерево с заполнением
    plt.show()


def Sample():
    df = pd.read_csv('datas/train.csv')
    df.head()
    #df['Age'] = df['Age'].factorize()[0]
    #df.head()
    #формирование матрицы признаков и результирующий столбец
    #x = df.drop('Age', axis=1)
    #y = df['Age']
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tree = DecisionTreeClassifier(max_depth=4)
    df['title'] = df['Name'].apply(get_title).apply(title_map)
    x_train = pd.crosstab(df['title'], df["Age"])
    y_train = pd.crosstab(df['title'], df["Pclass"])
    tree.fit(x_train, y_train)
    plot_tree(tree, filled=True)  # построить дерево с заполнением
    plt.show()
    tree.score(x_train, y_train)
    #export_graphviz(tree, out_file='datas/pic.dot', feature_names = x.columns, filled=True)
    # ! dot -Tpng datas/pic.dot -o datas/pic.png
    # <img src='datas/pic.png'>


if __name__ == '__main__':
    Titanic()
