import pandas as pd # for data tables
import os
import shutil

data = pd.read_csv("~/covid_research/metadata.csv")

refereshed_data = data[['finding', 'filename']]

refereshed_data['finding'].mask(refereshed_data['finding'] == 'COVID-19', 'positive', inplace=True)

# print(refereshed_data['finding'].unique())


# for data_images in refereshed_data.values:
#     if data_images[0] == 'positive':
#         try:
#             os.rename("images/"+data_images[1],"dataset/positive/"+data_images[1])
#         except:
#             pass
#     else:
#         try:
#             os.rename("images/"+data_images[1],"dataset/negative/"+data_images[1])
#         except:
#             pass
