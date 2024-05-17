# Predciting order returns on Amazon
Amazon is one of the most rewarding platforms for sellers worldwide. But one of the downsides of selling on Amazon is that sellers have to comply with easy return policy which can sometimes be detrimental the seller. Knowing beforehand if the product would be returned or not would be very helpful for the sellers.

## Outline
1. Introduction
2. Dataset information
3. EDA
4. Predicting order cancellation using MLlib in Spark
5. Conclusion
6. References

### Introduction
BL(Boss Leathers) is a small leather products business which has recently started selling its products on Amazon. At present, the seller has several SKU’s on Amazon. Over the past few months, it has incurred some loss due to return orders. Now, Boss Leather seeks help to predict the likelihood of a new order being rejected. This would help them to take necessary actions and subsequently reduce the loss.
Cancellation of orders can be predicted by running logistic regression model on the orders dataset. This dataset has been collected from Kaggle. The dataset belongs to BL(Boss Leathers) which is based in  India. The dataset contains just 171 rows and 12 columns. All the rows and columns contain useful data that is required to create a model that predicts the final status of the order. This model could be used on larger datasets with more information and can also be developed upon as the data increases. 

![image](https://user-images.githubusercontent.com/90857782/166130158-a28425e5-e03b-4a9f-8f7e-f314d07a4d5c.png)

### Dataset information
The Final datset is created by tweaking the original dataset by claeaning and encoding it. 

Just by looking at the data we can clearly see that the columns ship_city, ship_state and cod are primary in determining the final output. We can also see that all apart from one order that returned to the seller was cash on delivery. Which means that if the order is paid through cash on delivery, there is a higher chance of the order returning to the seller. Data like these could be used by the sellers to filter their business model in such a way that there are a smaller number of returns. This would eventually increase the profits as there are less returns.
The columns are: 
1.	order_no
2.	order_date
3.	buyer
4.	ship_city
5.	ship_state
6.	sku
7.	description
8.	quantity
9.	item_total
10.	shipping_fee
11.	cod
12.	order_status
13.	order_status_encoded
 
![image](https://user-images.githubusercontent.com/90857782/166176411-6a874e1d-c1c6-4947-bf73-f0f4d3da3c1b.png)

‘order_no’ column has all unique values which is useful while managing relational databases. It contains the 17-digit unique order number given by Amazon. This can be used as the primary column while linking the data together in querying. The ‘order_date’ column contains the date on which the order was placed. 


I have used this column to extract the day, month and year of the order placed into separate columns using pandas. This is helpful while analyzing the data.  

![image](https://user-images.githubusercontent.com/90857782/166130173-f8845669-39c0-4ead-a3da-c54f73cb86db.png)
 
The column ‘buyer’ contains the name of the buyer. In the dataset I used, which contains a small set of data from a seller on amazon, there is very low chance that a customer has ordered more than once from this seller. But, if the data would have been larger, we could find such instances. That data can be used to see if the order cancellations are coming from the same person. 

The columns ‘ship_city’ and ‘ship_state’ contains the shipped city and state of the order, respectively. If big enough, this data can be used to plot a heat map of the order placed to know the relative popularity of the product all over the country. 

‘sku’ column contains the unique identifier of the product sold which denotes the SKU of which the product belongs to. This with the help of ‘description’ column can be used to create a clustering algorithm.

The ‘item_total’ and ‘shipping_fee’ columns contain the total price of the product and shipping fee respectively. It is in INR(Indian Rupees) as the dataset is sourced from a seller in India. This column can be used to get the insights of how the business is performing financially.

‘cod’ column contains the information about the payment of the order. There are two modes of payments in Amazon India. They are Online payment and cash on delivery. 

The ‘order_status’ is the feature column which we will be predicting. There are two unique variables in this column, which are “Delivered to the buyer” and “Returned to the seller”.

I have created a new column using pandas which encoded the last column into 0’s and 1’s. This is done in python using pandas though Jupyter Notebook. The name of this column is ‘order_status_encoded’.

### Data Analysis
Analyzing the “order_status” column, we can see that out of 171 total entities, around 160 are “Delivered to the buyer" and around 10 are "returned to the seller". This plot is made using countplot from matplotlib library. “value_counts()” function is used to get the count of unique entities in the column. 

![image](https://user-images.githubusercontent.com/90857782/166130325-c6d94366-233a-4fbd-93a0-07edbd174824.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130329-11c609d1-8dab-42a9-a4ec-2dcbbc96bb48.png)

Looking at the order activity during the hours of the day, we can clearly see that the activity increasing until around 3pm and stabilizing after. This can be correlated with the active hours in which most of the people are awake. The ordering activity falls during nighttime as most them would be sleeping. As the number of orders increases, the sales increase and so as the income. At around 4pm, the total income from the sales would be around 11,500 Rupees.

![image](https://user-images.githubusercontent.com/90857782/166130340-6a067c29-2e35-43bc-ac44-c492ef817c09.png)

Code:

![image](https://user-images.githubusercontent.com/90857782/166130358-ec7ec9d9-195f-4871-a3aa-b732949b4b86.png)

Looking at the order activity over the days of the month, we can see that the activity is all over the place. This can be attributed to the small data size. There is information about just over 170 which is not enough to get the clear idea of the orders. The highest number of orders were placed on 4th and 16th and the least on 3rd. 

![image](https://user-images.githubusercontent.com/90857782/166130363-f4ab91a4-5be9-4880-9dc7-96aee1fd3665.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130366-0575824c-f4e3-4c50-80e7-e4d0466ba655.png)

Retaining customers is one of the best ways to develop a business. Looking at the customers who ordered more than once, we can see that ‘Mitra’ has ordered 6 times from the seller. 

![image](https://user-images.githubusercontent.com/90857782/166130371-190442c2-311b-4182-9948-bd9832c75473.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130373-24d075dc-01cc-4efa-81ba-b452e04f03d2.png)

The most common words used in description can be found out using wordcloud. Here is a depiction of the same.
![image](https://user-images.githubusercontent.com/90857782/166130418-e1936a58-5279-4286-9639-65f65f50a8eb.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130423-0c9469c3-acf3-409e-b184-348c2c27b6ee.png)

The top 20 states and cities with respect to the number of orders:
![image](https://user-images.githubusercontent.com/90857782/166130427-fe04630f-ef3d-4c4f-acd4-0c3381975c1c.png)
![image](https://user-images.githubusercontent.com/90857782/166130436-7c612452-f7ec-4bba-b76f-f5f30c5bd7e1.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130440-a4cbb957-9139-4b79-9176-d82808b36be9.png)

Here are the number of cancelled orders with respect to shipped cities and states.

![image](https://user-images.githubusercontent.com/90857782/166130467-7fe5d2b6-2cba-4b4f-bcc4-ae2e4d5127a6.png)

![image](https://user-images.githubusercontent.com/90857782/166130453-e65ca54a-c41a-4a83-9cf0-6baa19025476.png)

Code:

![image](https://user-images.githubusercontent.com/90857782/166130473-41078a57-737c-4d62-b149-8ff458939698.png)

The number of orders with respect to the SKU of the product

![image](https://user-images.githubusercontent.com/90857782/166130480-6b4bae8b-c47f-4a05-a815-4e00563c0545.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130481-8c7682ac-313f-43b6-8a6c-b0615d2d3047.png)

The number of cancelled orders with respect to the SKU’s. There are 2 major SKU’s which has the most cancellations. This data can be useful while inspecting the products.

![image](https://user-images.githubusercontent.com/90857782/166130490-fe0caf4e-5871-4b05-8c42-7a0cf3393098.png)

Cancellations with respect mode of delivery. As we can see from the plot, cancellations with “cash on delivery” are almost half as much as the ones with online payment. 

![image](https://user-images.githubusercontent.com/90857782/166130508-ef84080e-0aa7-4f12-8036-650313334fc4.png)

Code:
![image](https://user-images.githubusercontent.com/90857782/166130510-d6b78acd-b44a-454d-8227-4473b92c6292.png)


### Predicting order cancellation using MLlib in Spark
The prediction of the status of the order is done using the MLlib library in pyspark. I encoded the ‘order_staus’ column with 1’s and 0’s instead of the string data it has. This is done so that the regression is performed easily. This is done on Jupyter notebook using python. It is simply done by creating a pipeline that consists of string indexing one hot encoding of numerical data, creating a vector column using vector assembler and logistic regression. The data frame is then fitted into the pipeline so that all the above steps are performed on the dataset. The data frame is then transformed with the new data. 
The new data frame has a row ‘prediction’ which contains 1’s and 0’s where:
1 – Delivered to buyer
2 – Returned to the seller

![image](https://user-images.githubusercontent.com/90857782/166130527-7a6ca7d5-a677-41f0-abaa-75632bd25fe4.png)

### Conlusion:
Selling on Amazon is very rewarding for the sellers but has its caveats. One of them is that agreeing to Amazon’s easy return policy. Though it is still upto the seller to accept or reject a return,  doing otherwise would mean that the seller would be least preferred one which leads to poor sales. In this case, sellers can use the ML model to predict the probability of an order being cancelled. This can help them to forecast income and also help them to identify shortcomings in their model and rectify them. 

#### Dataset
https://www.kaggle.com/datasets/pranalibose/amazon-seller-order-status-prediction

### References

https://matplotlib.org/

https://spark.apache.org/mllib/





