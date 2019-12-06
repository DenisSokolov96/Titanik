import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import eli5
from sklearn import tree


def Titanic():
    X = pd.read_csv('datas/train.csv')
    # указываем зависимую перменную
    y = X['Survived']
    X.head()
    # смотрим, как распределены выжившие в зависимости от пола
    X[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    # удаляем из входов зависимую перменную и незначимые  признаки
    X.drop(['Survived', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
    X.head()
    X.info()
    # в поле Cabin много пропусков, удалим и его
    X.drop(['Cabin'], axis=1, inplace=True)
    X['Embarked'].describe()
    # дозаполняем пропуски
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Embarked'].fillna('S', inplace=True)
    X.info()
    # кодируем поле Embarked методом дамми-кодирования
    X = pd.concat([X, pd.get_dummies(X['Embarked'], prefix="Embarked")], axis=1)
    # удаляем старое поле Embarked
    X.drop(['Embarked'], axis=1, inplace=True)
    # кодируем поле обычным способом (0 и 1)
    X['Sex'] = pd.factorize(X['Sex'])[0]
    X.info()
    # делим выборку на обучающую и тестовую
    X_train = X[:-200]
    X_test = X[-200:]
    y_train = y[:-200]
    y_test = y[-200:]

    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21)
    clf.fit(X_train, y_train)
    clf.score(X_train, y_train)
    clf.score(X_test, y_test)

    rfc = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=21)
    rfc.fit(X_train, y_train)
    rfc.score(X_test, y_test)

    eli5.explain_weights_sklearn(clf, feature_names=X_train.columns.values)
    plot_tree(clf, filled=True)  # построить дерево с заполнением
    plt.show()

    export_graphviz(clf, out_file='datas/pic.dot')

    #export_graphviz(clf, out_file='datas/pic.dot', feature_names=X_train.columns, filled=True)


if __name__ == '__main__':
    Titanic()
