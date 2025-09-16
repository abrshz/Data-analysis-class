#!/usr/bin/env python
# coding: utf-8

# In[ ]:


numeric_columns = ['Age', 'Average_Purchase_Amount']

fig, axes = plt.subplots(1, 2, figsize = (12, 6))

for index, column in enumerate(numeric_columns):
    sns.kdeplot(final_data[column], ax = axes[index])
    axes[index].set_title(f'KDE plot of {column}')
    
plt.tight_layout()
plt.show()


# In[ ]:


from scipy.stats import shapiro

shapiro_results = {}

for column in numeric_columns:
    stat, p_value = shapiro(final_data[column])
    shapiro_results[column] = round(p_value, 3)
    
shapiro_results


# In[1]:


import numpy as np

def sqrt_transformation(data, column_name):
    data[f'{column_name}_sqrt'] = np.sqrt(data[column_name])
    stat, p_value = shapiro(data[f'{column_name}_sqrt'])
    kdeplot = sns.kdeplot(data[f'{column_name}_sqrt'])
    
    print(kdeplot)
    print('P value: ', p_value)


# In[ ]:


sqrt_transformation(final_data, 'Age')


# In[ ]:


def log_transformation(data, column_name):
    data[f'{column_name}_log'] = np.log(data[column_name])
    stat, p_value = shapiro(data[f'{column_name}_log'])
    kdeplot = sns.kdeplot(data[f'{column_name}_log'])
    
    print(kdeplot)
    print('P_value: ', p_value)


# In[ ]:


log_transformation(final_data, 'Age')


# In[ ]:


from scipy.stats import boxcox

def boxcox_transformation(data, column_name):
    transformed_data, _ = boxcox(data[column_name])
    data[f'{column_name}_boxcox'] = transformed_data
    stat, p_value = shapiro(data[f'{column_name}_boxcox'])
    kdeplot = sns.kdeplot(data[f'{column_name}_boxcox'])
    
    print(kdeplot)
    print('P value: ', p_value)


# In[ ]:


boxcox_transformation(final_data, 'Age')


# In[ ]:


from scipy.stats import yeojohnson

def yeojohnson_transformation(data, column_name):
    transformed_data, _ = yeojohnson(data[column_name])
    data[f'{column_name}_yeojohnson'] = transformed_data
    stat, p_value = shapiro(data[f'{column_name}_yeojohnson'])
    kdeplot = sns.kdeplot(data[f'{column_name}_yeojohnson'])
    
    print(kdeplot)
    print('P value: ', p_value)


# In[ ]:


yeojohnson_transformation(final_data, 'Age')


# In[ ]:


preprocessed_data = final_data.drop(['Age', 'Age_log', 'Age_boxcox', 'Age_yeojohnson'], axis = 1)
preprocessed_data.head()

