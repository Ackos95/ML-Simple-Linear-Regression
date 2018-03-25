#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math
import pandas


def mean(values):
    """ Calculates mean value for an array """

    return sum(values) / float(len(values))


def variance(values, mean_value):
    """ Calculates variance for an array """

    return sum([(x - mean_value) ** 2 for x in values])


def covariance(values_one, mean_one, values_two, mean_two):
    """ Calculates covariance between two arrays (and their mean values) """

    return sum([(values_one[i] - mean_one) * (values_two[i] - mean_two) for i in range(len(values_one))])


def calculate_coefficients(values_one, values_two):
    """ Calculates coefficients based on array values """

    mean_one, mean_two = mean(values_one), mean(values_two)

    k = covariance(values_one, mean_one, values_two, mean_two) / variance(values_one, mean_one)
    n = mean_two - k * mean_one

    return k, n


def linear_regression(train_data_array, test_data_array):
    """
        Calculate linear regression

        It creates predictions for test data array, based on coefficients calculated using train data array.

        :param train_data_array: {Array} containing train data values ('x' and 'y' keys are mandatory in each element)
        :param test_data_array: {Array} containing test data values ('x' key is mandatory in each element)

        :return: {Array} prediction values (for test data)
    """

    k, n = calculate_coefficients(train_data_array['x'], train_data_array['y'])

    return [k * x + n for x in test_data_array['x']]


def rmse_metric(actual, predicted):
    """
        Calculate RMSE metric

        It uses square root of summed squared error by size to calculate RMSE metric

        :param: actual {Array} - list of actual values
        :param: predicted {Array} - list of predicted values

        :return: {Float} value of the RMSE metric
    """

    return math.sqrt(sum([(predicted[i] - actual[i]) ** 2 for i in range(len(actual))]) / float(len(actual)))


def load_data(train_path, test_path, x_key, y_key):
    """
        Loads data as '.csv' files, and transforms keys into `x` and `y`

        It loads files using `pandas` library (reading `.csv`) and modifies datasets with custom
        `x` and `y` keys (based on `x_key` and `y_key` parameters)

        :param train_path: {String} path to the training .csv file
        :param test_path: {String} path to the test .csv file
        :param x_key: {String} key for x values in data set(s)
        :param y_key: {String} key for y values in data set(s)

        :return: {Array} [ train_file_data, test_file_data ] (Modified)
    """

    try:
        train_file_data, test_file_data = pandas.read_csv(train_path), pandas.read_csv(test_path)

        train_file_data['x'], train_file_data['y'] = train_file_data[x_key], train_file_data[y_key]
        test_file_data['x'], test_file_data['y'] = test_file_data[x_key], test_file_data[y_key]

        return train_file_data, test_file_data
    except Exception:
        raise Exception('Bad file paths provided...')


def parse_sys_args():
    """
        Helper function to parse command line arguments

        It expects two required cmd args (path to training data .csv file, and path to test data .csv file)
        and two optional (x_key - key under which are stored 'x' values, and y_key - same as x key).
        If optional command line arguments are not passed, default values are passed back.

        :return: {Array} [ train_csv_path, test_csv_path, x_key, y_key ]
        :raise: {Exception} if required arguments are not passed
    """

    try:
        train_path, test_path = sys.argv[1], sys.argv[2]

        x_key = sys.argv[3] if len(sys.argv) > 3 else 'size'
        y_key = sys.argv[4] if len(sys.argv) > 4 else 'weight'

        return train_path, test_path, x_key, y_key
    except Exception:
        raise Exception('Bad command line arguments provided.\n'
                        + 'Usage:\n\tpython src/main.py train_file_path test_file_path [x_key y_key]')


if __name__ == '__main__':
    try:
        train_data, test_data = load_data(*parse_sys_args())
        rmse = rmse_metric(test_data['y'], linear_regression(train_data, test_data))

        print(rmse)
    except Exception as e:
        print(e)
