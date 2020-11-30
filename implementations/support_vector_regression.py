from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from helper import get_data, split_data, visualize

name = 'Support vector'

if __name__ == '__main__':
    x, y = get_data()
    y = y.reshape(len(y), 1)
    x_train, x_test, y_train, y_test = split_data(x, y)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    regression = SVR(kernel='rbf')
    regression.fit(x_train, y_train)

    y_predicted = regression.predict(x_test)
    y_predicted = y_scaler.inverse_transform(y_predicted)
    y_test = y_scaler.inverse_transform(y_test)

    visualize(y_test, y_predicted, name)
