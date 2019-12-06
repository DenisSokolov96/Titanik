import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import eli5
from sklearn import tree
import seaborn as sns


def Titanic():
    trd = pd.read_csv('datas/train.csv')
    tsd = pd.read_csv('datas/test.csv')
    td = pd.concat([trd, tsd], ignore_index=True, sort=False)
    td.isnull().sum()
    sns.heatmap(td.isnull(), cbar=False).set_title("Missing values heatmap")
    plt.show()


    print(td.nunique())
    sns.heatmap(td.Survived(),cbar=False).set_title("123")
    #td.Pclass()
   # td.Sex()
   # td.Age()
    plt.show()



if __name__ == '__main__':
    Titanic()
