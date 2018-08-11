import pandas as pd
print(pd.__version__)


city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

city_populations = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(city_populations)

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

print(california_housing_dataframe.head()) # print first few rows of DataFrame

california_housing_dataframe.hist('housing_median_age')