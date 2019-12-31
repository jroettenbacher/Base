import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_presentation_plot_style():
    plt.style.use('ggplot')
    font = {'family': 'sans-serif', 'style': 'normal', 'size': 16}
    plt.rc('font', **font)



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


if __name__ == '__main__':
    r2 = polyfit(np.linspace(0, 100), np.arange(50, 100), 1)
    print(r2)
