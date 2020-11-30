# Do not use this type of regression on single-feature data.

from sklearn.tree import DecisionTreeRegressor

from helper import get_data, split_data, visualize

name = 'Decision tree'

if __name__ == '__main__':
    x, y = get_data()
    x_train, x_test, y_train, y_test = split_data(x, y)

    regression = DecisionTreeRegressor()
    regression.fit(x_train, y_train)

    y_predicted = regression.predict(x_test)

    visualize(y_test, y_predicted, name)
