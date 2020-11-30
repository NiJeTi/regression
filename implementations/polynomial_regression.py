from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from helper import get_data, split_data, visualize

name = 'Polynomial'

if __name__ == '__main__':
    x, y = get_data()
    x_train, x_test, y_train, y_test = split_data(x, y)

    pf = PolynomialFeatures()
    x_train_polynomial = pf.fit_transform(x_train)
    x_test_polynomial = pf.fit_transform(x_test)

    regression = LinearRegression()
    regression.fit(x_train_polynomial, y_train)

    y_predicted = regression.predict(x_test_polynomial)

    visualize(y_test, y_predicted, name)
