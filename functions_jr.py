import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt


def set_presentation_plot_style():
    plt.style.use('ggplot')
    font = {'family': 'sans-serif', 'size': 24}
    figure = {'figsize': [16, 9], 'dpi': 300}
    plt.rc('font', **font)
    plt.rc('figure', **figure)



# from https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


def daterange(start_date, end_date):
    """ Generator to create a loop over dates by day
    from: https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
    :param start_date: datetime object
    :param end_date: datetime object
    :return: loop over date
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


if __name__ == '__main__':
    r2 = polyfit(np.linspace(0, 100), np.arange(50, 100), 1)
    print(r2)
