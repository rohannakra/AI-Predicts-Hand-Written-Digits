# AI-Predicts-Hand-Written-Digits

Steps Taken:
* data analysis
    * check if the data is linear or non-linear through applying a ```Perceptron``` model
    * check for zeros or NaN values in the data
        * if one or the other are true, use dimensionality reduction.
* create model
    * use ```Pipeline``` interface in conjunction with ```cross_validate```
* view the results
    * reform data for ```plt.imshow()```
<img src='img.png'>
