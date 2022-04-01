# Prediction of order cancellation in Amazon

## Summary
<p>Being a seller on amazon is rewarding but it also has its downsides. One of them being honoring the easy return policy of Amazon. The price of a product is formulated based on various factors like shipping rates, cost price of the product, number of items potentially being returned. Knowing beforehand if the product would be returned or not would be very helpful for the seller on Amazon. If the returns are higher in a certain area, the seller could stop selling that product in the area or accumulate a smaller number of items in that area’s Amazon warehouse.</p>

<p>The dataset is from a seller on Amazon India. Over the past few months, the seller has incurred losses due to higher number of orders being returned. Analyzing the dataset to predict the likelihood of the order being cancelled or returned can help the company reduce the losses and also formulate a business plan based on the demand for the products throughout the country.</p>

## Outline
<p>The dataset contains just 171 rows and 12 columns. All the rows and columns contain useful data that is required to create a model that predicts the final status of the order. This model could be used on larger datasets with more information and can also be developed upon as the data increases.</p>

<p>Just by looking at the data we can clearly see that the columns ship_city, ship_state and cod are primary in determining the final output. We can also see that all apart from one order that returned to the seller was cash on delivery. Which means that if the order is paid through cash on delivery, there is a higher chance of the order returning to the seller.</p>

<p>Data like these could be used by the sellers to filter their business model in such a way that there are a smaller number of returns. This would eventually increase the profits as there are less returns.</p>

<p>I would try and find more datasets like these to train the model and also analyze the trends of orders placed on amazon.</p>

## Sources
Dataset - [Amazon Seller - Order Status Prediction](https://www.kaggle.com/datasets/pranalibose/amazon-seller-order-status-prediction)
