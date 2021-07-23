import pandas as pd
import numpy as np


def predict_price(new_listing_value,feature_column):
	temp_df = train_df
	temp_df['distance'] = np.abs(nyc_listings[feature_column] - new_listing_value)
	temp_df = temp_df.sort_values('distance')
	knn_5 = temp_df.price.iloc[:5]
	predicted_price = knn_5.mean()
	return(predicted_price)


nyc_listings = pd.read_csv('listings_NYC.csv')
print (nyc_listings.shape)

our_acc_value = 3
first_living_space_value = nyc_listings.loc[0,"accommodates"]
print (first_living_space_value)


first_distance = np.abs(first_living_space_value - our_acc_value)
print (first_distance)
#print (nyc_listings.head())

nyc_listings['distance'] = np.abs(nyc_listings.accommodates - our_acc_value)
print (nyc_listings.distance.value_counts().sort_index())

nyc_listings = nyc_listings.sample(frac=1,random_state=0)
nyc_listings = nyc_listings.sort_values('distance')
print (nyc_listings.price.head())

nyc_listings['price'] = nyc_listings.price.str.replace("\$|,",'').astype(float)
mean_price = nyc_listings.price.iloc[:5].mean()
print (mean_price)

#33237,11080

nyc_listings.drop('distance',axis=1)

train_df = nyc_listings.copy().iloc[:33237]
test_df = nyc_listings.copy().iloc[33237:]


for feature in ['accommodates','bedrooms','bathrooms','number_of_reviews']:
    test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column=feature)
    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
    mse = test_df['squared_error'].mean()
    rmse = mse ** (1/2)
    print("RMSE for the {} column: {}".format(feature,rmse))


