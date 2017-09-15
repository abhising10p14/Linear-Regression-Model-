import graphlab #for the graphlab module to access and edit the data accordingly
# importing the matlotlib ,pyplot for plotting purpose
import matplotlib.pyplot as plt
#graphlab.get_dependencies()
#import IPython # for the ipython notebook
#Load some house sales data
sales = graphlab.SFrame('home_data.gl')


'''Its is seen that the by default the target is set to the brawser
To change the target to the IPython notebook do :'''
#graphlab.canvas.set_target('ipynb')
#Plotting the data
sales.show(view="Scatter Plot",x ="sqft_living",y="price")
#Scatter Plot is used to give  plotted view of the data of the two 
#corresponding columns.
#sales.show(view="Heat Map",x ="sqft_living",y="price")
#Shows a heat map format of the data 

#Creating a simple regression model of the sqft_living attribute
'''But before that we SPLIT our data into training and tes data'''
train_data,test_data = sales.random_split(0.8,seed=0) #Splits the data into two SFrames corresponding to the fraction provided and seed is provided to 
#to ensure that the spliting always occurs in the same way.



## Build a regression model :
#target : is the column we want to predict
#train_data is the data to input 
#features : the features you want to include 
# As we are not defining which algorithm is to be used for regression
# it will automatically pick up any method , in my case it was Newton method

sqft_model = graphlab.linear_regression.create(train_data,target = 'price',features=['sqft_living'])

'''Linear regression:
--------------------------------------------------------
Number of examples          : 16544
Number of features          : 1
Number of unpacked features : 1
Number of coefficients    : 2
Starting Newton Method
--------------------------------------------------------
+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
| Iteration | Passes   | Elapsed Time | Training-max_error | Validation-max_error | Training-rmse | Validation-rmse |
+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
| 1         | 2        | 1.026463     | 4364844.145642     | 2908872.159685       | 261326.370435 | 293068.561079   |
+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
SUCCESS: Optimal solution found.
'''

#Now evaluate this simple model
print test_data['price'].mean()
# prints the average price of a house 

print sqft_model.evaluate(test_data)


'''Now we are going to plot two curves :
1. The actual curve of test_data['price'] vs test_data['sqft_living'] using '.'
2. The predicted curve of test_data['price'] vs test_data['sqft_living'] as predicted by our model using '-'

'''
# %matplotlib inline
plt.plot(test_data['sqft_living'],test_data['price'],'.',test_data['sqft_living'],sqft_model.predict(test_data),'-')
plt.show()
sqft_model.get('coefficients') #to get the coeeficients of the model
'''Columns:
	name	str
	index	str
	value	float
	stderr	float

Rows: 2

Data:
+-------------+-------+----------------+---------------+
|     name    | index |     value      |     stderr    |
+-------------+-------+----------------+---------------+
| (intercept) |  None | -44893.0489126 | 5010.26674444 |
| sqft_living |  None | 280.501983674  | 2.20185134753 |
+-------------+-------+----------------+---------------+
[2 rows x 4 columns]
'''
#Now we will explore some other features for enhancing the regression model

my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
sales[my_features].show()

sales.show(view= 'BoxWhisker Plot',x='zipcode')

#using another model with more number of features
my_features_model = graphlab.linear_regression.create(train_data,target = 'price',features=my_features)
# RMSE erro = 249232.181221  
'''Linear regression:
--------------------------------------------------------
Number of examples          : 16498
Number of features          : 6
Number of unpacked features : 6
Number of coefficients    : 115
Starting Newton Method
--------------------------------------------------------
+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
| Iteration | Passes   | Elapsed Time | Training-max_error | Validation-max_error | Training-rmse | Validation-rmse |
+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
| 1         | 2        | 0.061205     | 3754187.230628     | 2414662.986243       | 181145.948296 | 205578.683553   |
+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
SUCCESS: Optimal solution found.
'''

#Comparing the errors of the both the models i.e sqft_model and my_features_model

print sqft_model.evaluate(test_data)
#{'max_error': 4099353.761645632, 'rmse': 255327.6550319461}

print my_features_model.evaluate(test_data)
#{'max_error': 3453994.430281713, 'rmse': 179402.3416700109}


#RSS error is reduced in second case 

#Now lets compare the predicted price for two of the houses
house1 = sales[sales['id'] =='5309101200']
print house1

'''+------------+---------------------------+--------+----------+-----------+-------------+
|     id     |            date           | price  | bedrooms | bathrooms | sqft_living |
+------------+---------------------------+--------+----------+-----------+-------------+
| 5309101200 | 2014-06-05 00:00:00+00:00 | 620000 |    4     |    2.25   |     2400    |
+------------+---------------------------+--------+----------+-----------+-------------+
+----------+--------+------------+------+-----------+-------+------------+---------------+
| sqft_lot | floors | waterfront | view | condition | grade | sqft_above | sqft_basement |
+----------+--------+------------+------+-----------+-------+------------+---------------+
|   5350   |  1.5   |     0      |  0   |     4     |   7   |    1460    |      940      |
+----------+--------+------------+------+-----------+-------+------------+---------------+
+----------+--------------+---------+-------------+---------------+---------------+-----+
| yr_built | yr_renovated | zipcode |     lat     |      long     | sqft_living15 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
|   1929   |      0       |  98117  | 47.67632376 | -122.37010126 |     1250.0    | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
[? rows x 21 columns]
'''


#importing image of the house and print the picture along with it 
# I am just trying to show how you can outpput an image using python module
# named PIL  and the image which I have used is of Amityville . I love horror movies :)

from PIL import Image 
im = Image.open("amityville.jpg")
im.show()
#actual price of the house1
print house1['price']
sqft_model.predict(house1)
my_features_model.predict(house1)

plt.plot(test_data['sqft_living'],test_data['price'],'.',test_data['sqft_living'],my_features_model.predict(test_data),'-')
plt.show()

data_of_98039zip = sales[sales['zipcode']=='98039']
 ans = data_of_98039zip['price'].mean() #gives the mean of the houses with this zipcode

 #ans = 2160606.5999999996

#to select those houses who have sqft_living feature in particular range (2000,4000)

# Actualy they are binary_filters , which are used to filter the data of a SFrame

sf = sales[(sales['sqft_living']<2000) &(sales['sqft_living']<=4000)]

#To get the number of rows in SFrame use SFrame.num_rows also see num_columns()

x = sf.num_rows()
print x
#x = 11609


# Adding more number of features call it as Advance features and then creating another model 
advanced_features = 
[
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

Advance_model = graphlab.linear_regression.create(train_data,target = 'price',features=advanced_features)
# Validation rmse = 145570.064601 
Advance_model.evaluate(test_data)
#{'max_error': 3552381.8466896815, 'rmse': 153713.72834208477}

## RMSE = (RSS/N)^0.5

