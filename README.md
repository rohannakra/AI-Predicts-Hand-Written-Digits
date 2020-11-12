# AI-Predicts-Hand-Written-Digits

Steps Taken:
* data analysis
    * check if the data is linear or non-linear through ```TSNE``` dimensionality reduction technique
    * check for zeros in the data
    * check for NaN values in the data
* create model
    * use ```Pipeline``` interface in conjunction with ```cross_validate```
* view the results
    * reform data for ```plt.imshow()```
<img src='img.png'>
