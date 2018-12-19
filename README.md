# Skin Segmentation Logistic Regression

This repo contains a Jupyter Notebook that outlines the steps to running a logistic regression. The dataset used is called Skin Segmentation. 

The blog post that accompanies this repo can be found [here](https://medium.com/@AudreyLorberfeld/logistic-regression-for-facial-recognition-ab051acf6e4).

The data in Skin Segmentation were collected "by randomly sampling B,G,R values from face images of various age groups (young, middle, and old), race groups (white, black, and asian), and genders obtained from FERET database and PAL database." The dataset has 245,057 rows and 4 columns (B, G, R, and a binary column indicated if the image was classified as containing skin or not containing skin). The latter column made this dataset ripe for logistic regression.

- The site I got the dataset from: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

- The URL of the actual data: http://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt

- The color FERET database: https://www.nist.gov/itl/iad/image-group/color-feret-database

- 1 article that uses the data (behind a paywall): https://ieeexplore.ieee.org/abstract/document/5409447 

- Another article that uses data (behind a paywall): https://link.springer.com/chapter/10.1007/978-3-642-10520-3_69

To run the code in the Jupyter Notebook on your own, you will need to have Python 3, scikit-learn, Pandas, NumPy, matplotlib, Seaborn, Itertools, and imblearn installed/imported on your machine/in your virtual environment.

Requirements.txt coming soon!
