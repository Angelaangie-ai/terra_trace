import pandas as pd

# Load the provided CSV files
ndvi_first_part_2020_april = pd.read_csv('/content/drive/MyDrive/NDVI_Fifth_Part_2022_November.csv')

# Pivot the data to the desired format lat|lon|date1|date2, with ndvi for each date
pivoted_data = ndvi_first_part_2020_april.pivot_table(index=['latitude', 'longitude'], columns='date', values='ndvi').reset_index()

# Flatten the MultiIndex columns
pivoted_data.columns = ['lat', 'lon'] + [f'{date}' for date in pivoted_data.columns[2:]]

# Save the transformed dataframe to a new CSV file
pivoted_data.to_csv('/content/Transformed_NDVI_Fifth_Part_2022_November.csv', index=False)

# Display the transformed data
print(pivoted_data.head())

import pandas as pd

# Load the provided CSV files
ndvi_first_part_2020_april = pd.read_csv('/content/drive/MyDrive/NDVI_Fifth_Part_2022_October.csv')

# Pivot the data to the desired format lat|lon|date1|date2, with ndvi for each date
pivoted_data = ndvi_first_part_2020_april.pivot_table(index=['latitude', 'longitude'], columns='date', values='ndvi').reset_index()

# Flatten the MultiIndex columns
pivoted_data.columns = ['lat', 'lon'] + [f'{date}' for date in pivoted_data.columns[2:]]

# Save the transformed dataframe to a new CSV file
pivoted_data.to_csv('/content/Transformed_NDVI_Fifth_Part_2022_October.csv', index=False)

# Display the transformed data
print(pivoted_data.head())

import pandas as pd

# Load the provided CSV files
ndvi_first_part_2020_april = pd.read_csv('/content/drive/MyDrive/NDVI_Eight_Part_2020_May.csv')

# Pivot the data to the desired format lat|lon|date1|date2, with ndvi for each date
pivoted_data = ndvi_first_part_2020_april.pivot_table(index=['latitude', 'longitude'], columns='date', values='ndvi').reset_index()

# Flatten the MultiIndex columns
pivoted_data.columns = ['lat', 'lon'] + [f'{date}' for date in pivoted_data.columns[2:]]

# Save the transformed dataframe to a new CSV file
pivoted_data.to_csv('/content/Transformed_NDVI_Eight_Part_2020_May                                                                                                                                                                                                                                                .csv', index=False)

# Display the transformed data
print(pivoted_data.head())

import pandas as pd

# Load the provided CSV files
ndvi_first_part_2020_april = pd.read_csv('/content/drive/MyDrive/NDVI_Seventh_Part_2020_December.csv')

# Pivot the data to the desired format lat|lon|date1|date2, with ndvi for each date
pivoted_data = ndvi_first_part_2020_april.pivot_table(index=['latitude', 'longitude'], columns='date', values='ndvi').reset_index()

# Flatten the MultiIndex columns
pivoted_data.columns = ['lat', 'lon'] + [f'{date}' for date in pivoted_data.columns[2:]]

# Save the transformed dataframe to a new CSV file
pivoted_data.to_csv('/content/Transformed_NDVI_Seventh_Part_2020_December.csv', index=False)

# Display the transformed data
print(pivoted_data.head())

from google.colab import drive
drive.mount('/content/drive')

!pip install ace_tools
