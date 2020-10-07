# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#import numpy as np
import adboost
import numpy as np
import data
def test():
    mat = np.arange(9).reshape((3,3))
    print(mat)
    rangeMax = np.max(mat[:,1])
    print(rangeMax)
    mat2 = np.arange(9).reshape((3,3))
    print(mat2==mat)


def main():
    D = np.mat(np.ones((5, 1)) / 5)
    datMat,classLabels = data.loadSImpleData()
    #bestStump, minError, bestClassEst = adboost.bulidStump(datMat,classLabels,D)
    #print(bestStump, minError, bestClassEst)
    classifierArray = adboost.adaBoostTrainDS(datMat,classLabels,9)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    #test()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
