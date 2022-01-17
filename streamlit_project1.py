#insert library
import pandas as pd
import numpy as np

import streamlit as st
import pickle

import scipy
from scipy.stats import chi2_contingency, chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, tree

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import datetime

#import function library
import project1_func

#Source code

#Load dataset
df = pd.read_csv('avocado.csv', index_col = 0)

#convert date and create month feature
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = pd.DatetimeIndex(df['Date']).month

#create season feature
df['season'] = df['month'].apply(lambda x: project1_func.season(x))

#rename avocados type columns
df =df.rename(columns = {
    '4046':'Small_hass',
    '4225':'Large_hass',
    '4770':'Xlarge_hass'
})

#data for regression model
X = pd.read_csv('X.csv', index_col = 0)
y = pd.read_csv('y.csv', index_col = 0)
#Modeling & Evaluating
#Train test plit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)
#Extratree Regressor
etr = ExtraTreesRegressor(max_depth= 60, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 150)
etr.fit(X_train, y_train)
#predict y value from X_test
y_train_pred = etr.predict(X_train)
y_test_pred = etr.predict(X_test)
#Model evaluation
#score original data (R^2)
R_ori_multi = etr.score(X, y)
#score train
R_train_multi = etr.score(X_train, y_train)
#score test
R_test_multi = etr.score(X_test, y_test)

#mse original
mse_ori_multi = mean_squared_error(y, etr.predict(X))
#mse train
mse_train_multi = mean_squared_error(y_train, y_train_pred)
#mse test
mse_test_multi = mean_squared_error(y_test, y_test_pred)

#mae original
mae_ori_multi = mean_absolute_error(y, etr.predict(X))
#mae train
mae_train_multi = mean_absolute_error(y_train, y_train_pred)
#mae test
mae_test_multi = mean_absolute_error(y_test, y_test_pred)


#GUI
#main page
st.title("Data Science Project")
st.write("## Hass Avocado Price Predition")

#Show result
menu = ["EDA", "Regression", "Time Series California", 'Time Series Houston']
choice = st.sidebar.selectbox('Menu', menu)

#eda
if choice == 'EDA':
    st.subheader('EDA')
    st.write('### Dataframe head')
    st.dataframe(df.head(5))
    st.write('### Dataframe Tail')
    st.dataframe(df.tail(5))
    st.write('#### Continous & Continous')
    st.write('#### Pairplot')
    st.image("pairplot.png")
    st.write("#### Correlation Plot")
    st.image("corrplot.png")
    st.image("boxplot.png")
    st.write('All continuous have upper outliers')
    st.image("barplot.png")
    st.write("""This plot showed the top Total Volume region of all year. Top 5 region have a large amount of avocado consume: california, Greatlakes, Northeast, Southcentral, Southeast.""")
    st.image("avocadotype.png")
    st.write("The TotalUS's main consumption is conventional avocado (> 300 mil per year), organic product are not much (lower than 65 mil per year) due to its price and it does not fit with low to middle income region.")
    st.image("avgprice.png")
    st.write("This plot showed clearly the difference between avocado type price. Organic avocado seem to be high standard deviation because it stretched around mean [1.2, 3.25], also it have outliers. While conventional avocados is thinner then organic and its price located from 0.5 to 2.0")
    st.image("factorplot1.png")
    st.write("""- As I said, organic product have a variety prices from 1.2 to 2.4, there are some region will be potential due to its price such as:
    - Houston: the price 2015 quite high and droped down really low in 2016 then rised up from 2017, and this still the bottom Agv price region.
    - Same case as Houston, we have SouthCentral, DallasFtWorth.
    - The common ground of regions is the Organic Avage price droped down badly in 2016 and rised up in 2017. This case occurs clearly for those regions not so well known/sparsely populated. Or evenif that region is well known, may be consumer does not like avocados due to consumer behavior of that region.""")
    st.image("factorplot2.png")
    st.write("""- Conventional case is difference from Organic, because it have a large amount of consumption (> 300 mil per year) so the trend will be differ.
    - Raised from 2015 to the lasted year (2017), the conventional seem to be popurality in all region. And the interesting point abot this case is most of region's Average price are nearly the same, and some big region on top of Organic Average price will have the Conventional Average price high also.
    - The region we have to focus in this case (conventional) is Houston, DallasFTWorth and SouthCentral, same case as Organic""")
elif choice == "Regression":
    st.subheader("Extra-Tree Regression Model")
    st.write("### R-Squared Score")
    st.write("R-Square Original", round(R_ori_multi,4))
    st.write("R-Square Train", round(R_train_multi,4))
    st.write("R-Square Test", round(R_test_multi,4))
    st.write("### MSE Score")
    st.write("MSE Score Original", round(mse_ori_multi,4))
    st.write("MSE Score Train", round(mse_train_multi,4))
    st.write("MSE Score Test", round(mse_test_multi,4))
    st.write("### MAE Score")
    st.write("MAE Score Original", round(mae_ori_multi,4))
    st.write("MAE Score Train", round(mae_train_multi,4))
    st.write("MAE Score Test", round(mae_test_multi,4))
    st.write("### Extra-Tree model give a quite good result, but we need to improve more")
elif choice == 'Time Series California':
    st.write("## California Organic Avocado")
    st.image("trend1.png")
    st.write("""- There is a seasonal trend from 2015 to 2018:
- The trend of AveragePrice will raise from Sep and droped down in Oct and continued through years.
- In the opposite, Total Volume trend will raised from the begining of Spring until droped down in May.""")
    st.write("### ARIMA Model")
    st.image("arima1.png")
    st.write("""- In the next 40 weeks, there is a good result showed that Clifornia will be one of the main market of the organization:
- The seasonal trend will be continued, but this year (2018) will be harder than past year. Specifically, it will begin from December.
- The price will be higher than any past year => Revenue will decreased.
- Notice: the down trend of 2019 will be happended really fast, price will be higher than this year prediction (2018) and the slope will like year 2016. I suggest we will be careful if we still target cali region in 2019 because the plot shoed that althought price droped down, but still higher than the past.""")
    st.write("### Holtwinter Model")
    st.image("holtwinter1.png")
    st.write("""- The price still happened same case with Arima, but the price of prediction seem lower than Arima prediction and it will be continuing its down trend.
- The Holwinter mse is the same as Arima, but give the totaly difference prediction result. I will make a prediction with fbprophet model for advance.""")
    st.write("FbProphet Model")
    st.image("fbprophet1.png")
    st.write("""- Trend will increased from 2018 to 2019
- Same scenario as Arima model, the trend rised up from summer or fall and droped down really fast at the beginning of winter. This time people does not buy avocados until price low in Jan - May.
- But in this model, we can see activity are really high at the weekend (weekly plot) because customer usually buy stuff for the whole week.""")
    st.write("### Result Summarize - Organic")
    st.image("result_summarize1.png")
    st.write("""- The Fbprophet and Holtwinter will be best fit with the data, in my case prefer Holtwinter for flunctuation tactic and FBProphet for trend tactics.
- The FbProphet seem to be safe, but it value prediction is nearly to Holtwinter and it good for showing the trend (in this case the trend will increased).
- In the other hand, we can use Holtwinter for backup plan if the functuation comed up.
- Those 3 model are good, it showed us the price will continuing low in 2018 and 2019, but may be it higher a little bit than 2017 and it still accepable.
So that California willl still our main market in the next year for Organic product.""")
    st.write("## California Conventional")
    st.image("trend2.png")
    st.write("""- There is a seasonal trend from 2015 to 2018:
- The trend of AveragePrice will raise from Sep and droped down in Oct and continued through years. In the opposite, Total Volume trend will raised from the begining of Spring until droped down in May.
- We can see the revenue of conventional avocado increased through years""")
    st.write("### ARIMA Model")
    st.image("arima2.png")
    st.write("""- In the next 40 weeks, there is a good result showed that California will be one of the main market of the organization:
- The seasonal trend will be continued, but this year (2018) will be chance for us for making profit.
- The 2018 conventional price will be lower than last year => Revenue will increased.
- Notice: the down trend of 2019 will be happended really fast, price seem the same with this year prediction (2018) and the slope will like year 2016.""")
    st.write("### Holtwinter Model")
    st.image("holtwinter2.png")
    st.write("""- The price still happened same case with Arima, but the price of prediction seem lower than Arima prediction a lot and it will be continuing its down trend.
- The Holwinter mse is the same as Arima, but give the totaly difference prediction result. I will make a prediction with fbprophet model for advance.""")
    st.write("### FbProphet Model")
    st.image("fbprophet2.png")
    st.write("""- Prophet trend seem quite safe, showed that it can not react with fluctuation just like arima and Holtwinter. Besides, the prediction result showed that this and next year price will increased.
- The Cali conventional market will continued as one of the biggest market of us.""")
    st.write("###Result Summarize - Conventional")
    st.image("result_summarize2.png")
    st.write("""- The Arima and Holtwinter will be best fit with the data, in my case prefer Holtwinter for flunctuation tactic and Arima for trend tactics.
- The Holtwinter seem to be good, but it value prediction is quite low to Arima's functuation and it good for showing a chance in making profit (price drop down in this year) => volume increased..
- In the other hand, we can use Arima for backup plan if the functuation comed up.
- Those 2 model are good, it showed us the price will continuing growth in 2018 and 2019. In this case, Fbprophet does not convince much.
- So that California willl still our main market in the next year for conventional product.""")
elif choice == 'Time Series Houston':
    st.write('## Houston Organic Avocado')
    st.image("trend3.png")
    st.write("""- There is a seasonal trend from 2015 to 2018:
- The trend of AveragePrice will droped down from Spring until May Summer and continued through years. In the opposite, Total Volume trend will raised from the begining of Spring until droped down in May. Also with revenue.""")
    st.write("### ARIMA Model")
    st.image("arima3.png")
    st.write("""- In the next 40 weeks, the Arima model showed that price this year will be lower than 2017.
- But because the mse quite high, so that this result not convinced much.""")
    st.write("### Holtwinter Model")
    st.image("holtwinter3.png")
    st.write("""- The price still happened same case with Arima.
- The Holwinter mse is the lower than Arima, but give the same prediction result. I will make a prediction with fbprophet model for advance.""")
    st.write("### FbProphet Model")
    st.image("fbprophet3.png")
    st.write("""- Prophet trend seem quite safe, showed that it can react with fluctuation just like Arima and Holtwinter.
- The Houston organic market will be continuing its trend acording to prophet model.""")
    st.write("### Result Sumarize - Organic")
    st.image("result_summarize3.png")
    st.write("""- The Arima and Holtwinter will be best fit with the data, when those 2 model gave the same result. We can make profit with organic avocado. But must have backup plan if the price happened like prophet model showed.
- Those 2 model are good, it showed us the price will continuing low in 2018 and 2019, I still doubt about those 3 result.
- So that houston will be our new market in this year for Organic product.""")
    st.write('## Houston Conventional Avocado')
    st.image("trend4.png")
    st.write("""- There is a seasonal trend from 2015 to 2018:
- The trend of AveragePrice will droped down from Spring until March.""")
    st.write("### ARIMA Model")
    st.image("arima4.png")
    st.write("""- In the next 40 weeks, there is a good result showed that Houston will be one of the new market of the organization:
    - The seasonal trend will be continued, but this year (2018) will be harder than past year. Specifically, the price as Arima model gave higher than last year, but it still accepable.
    - The price will be high, but not high than the big market like california, and this year price have decreased lower than last year so the profit will be increased.""")
    st.write("### Holtwinter Model")
    st.image("holtwinter4.png")
    st.write("""- The price of Holtwinter is different than Arima, when the prediction price this year is lower than last year.
- The Holwinter mse is lower than Arima (not much), but give the totaly difference prediction result. The prediction is different than the actual test, when it can not stick with the large margin area.
- I will make a prediction with fbprophet model for advance""")
    st.write("### FbProphet Model")
    st.image("fbprophet4.png")
    st.write("""- Prophet trend seem quite safe, showed that it can not react with fluctuation just like arima and Holtwinter.
- But the prediction price area nearly the same as last year (increased a little bit).
- The houston conventional market will be the new market of the organization.""")
    st.write("### Result Summarize - Conventional")
    st.image("result_summarize4.png")
    st.write("""- The Fbprophet and Holtwinter will be best fit with the data, in my case prefer Holtwinter for flunctuation tactic and FBProphet for trend tactics.
    - The FbProphet seem to be safe, and its value prediction is nearly to Holtwinter and it good for showing the trend (in this case the trend will increased sightly).
    - Besides, the Holtwinter give us a chance for a possitive price this year, and with the volume trend prediction increased at the begining of this file, the profit gainned from this region is really good. Besides, we must have backup plan if the functuation comed up.
- Those 2 model are good, it showed us the price will continuing low in 2018 and 2019, and it will be a chance for us.
- So that houston willl be our new market in this and next year for both conventional & organic product.""")
    


