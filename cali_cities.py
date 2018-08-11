import pandas as pd
import numpy as np
print(pd.__version__)


city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
# add new Series to existing DataFrame
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['isGreaterThan50sqmi'] = cities['Area square miles'] > 50
cities['isSpanishColonized'] = cities['City name'].apply(lambda name: name.startswith('San'))

print(cities)

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

print(california_housing_dataframe.head()) # print first few rows of DataFrame

california_housing_dataframe.hist('housing_median_age')

# print(cities['City name'][1])
# print(population / 1000)
# print(np.log(population))
# print(population.apply(lambda val: val > 1000000))

# randomly shuffle DataFrame
print(cities.index)
# cities.reindex(np.random.permutation(cities.index))
cities.reindex([0, 4, 2])
print(cities.index)
