#!/usr/bin/env python
# coding: utf-8

# # Project: Energy Production, Consumption and CO<sub>2</sub> Emission Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# 
# 
# In this project, I'll be analysing data associated with total energy produced and consumed in different countries over about two decades, and the Consumption CO2 per capita for the people in the countries. 
# 
# **Project Aim**
# 
# The main aim of this project it to explore trends on energy production, consumption and CO<sub>2</sub> emissions within two decades from around the world. The research questions include:
# 
# * Which countries are the top and least energy producers?
# * Which countries consume the most and least energy?
# * Which countries are the highest and lowest CO<sub>2</sub> emitters?
# 
# > **3 datasets where used:**: 
# * **Energy Production dataset** - `energy_production_total.csv`. Description: Energy production refers to forms of primary energy--petroleum (crude oil, natural gas liquids, and oil from nonconventional sources), natural gas, solid fuels (coal, lignite, and other derived fuels), and combustible renewables and waste--and primary electricity, all converted into tonnes of oil equivalents. 
#      > 
#      * Unit of measurement: Tonnes of oil equivalent (toe)
#      > 
#      * Source: [World Bank, 2010](https://data.worldbank.org/indicator/EG.EGY.PROD.KT.OE)  
#      > 
# * **Energy Consumption dataset** - `'energy_use_per_person.csv`. Description: Energy use refers to use of primary energy before transformation to other end-use fuels, which is equal to indigenous production plus imports and stock changes, minus exports and fuels supplied to ships and aircraft engaged in international transport.
#      > 
#      * Unit of measurement: Kg of oil equivalent per capita
#      > 
#      * Source: [World Bank, 2015](https://data.worldbank.org/indicator/EG.USE.PCAP.KG.OE)
#      > 
# * **Consumption CO<sub>2</sub> per capita dataset** - `consumption_emissions_tonnes_per_person.csv`. Description: Per capita carbon dioxide emissions from the fossil fuel consumption, cement production and gas flaring, minus export, plus import during the given year.
#      > 
#      * Unit of measurement: Metric tons of CO2 per person
#      > 
#      * Source: [Gapminder](https://github.com/open-numbers/ddf--gapminder--co2_emission)
# 

# In[1]:


# Setting up import statements for all of the packages that will be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > In this section, I load the datasets, and explore them for cleanliness, and then trim and clean the datasets for analysis based on the observations made. For each dataset, I cleaned completely before moving to the next dataset.
# 
# ### General Properties

# In[2]:


#Loading energy production dataset
df_energy_prod = pd.read_csv('energy_production_total.csv')
print(df_energy_prod.shape)
df_energy_prod.head(2)


# In[3]:


#Loading energy consumption dataset
df_energy_usepp = pd.read_csv('energy_use_per_person.csv')
print(df_energy_usepp.shape)
df_energy_usepp.head(2)


# In[4]:


#Loading CO2 dataset
df_co2_consump = pd.read_csv('consumption_emissions_tonnes_per_person.csv')
print(df_co2_consump.shape)
df_co2_consump.head(3)


# In[5]:


# Exploring energy production dataset
df_energy_prod.describe()


# In[6]:


df_energy_prod.info()


# In[7]:


# Exploring energy consumption dataset
df_energy_usepp.describe()


# In[8]:


df_energy_usepp.info()


# In[9]:


# Exploring CO2 dataset
df_co2_consump.describe()


# In[10]:


df_co2_consump.info()


# #### Observations 1
# * In both energy produced and consumed datasets, 25-26 countries have enteries from 1960-1970. This is a really small number, compared to the 135 countries. 
# * 2010 in energy production dataset and 2015 in energy consumption dataset both have 34 enteries
# * Also, the CO2 consumption dataset has only enteries from 1990, and from 119 countries. We might be dropping the rows in both energy datasets from 1960-1970. 
# * The values are float types, not integers
# * Case type in country column needs to be changed (for uniformity)
# * There is are a lot of missing values in multiple columns in the energy consumption dataset
# 
# First, let us look at what countries are featured in the first decade.

# In[11]:


#exploring the countries with values in the first decade

not_nullprod = df_energy_prod[df_energy_prod["1964"].notnull()]
not_nulluse = df_energy_usepp[df_energy_usepp["1964"].notnull()]

print(not_nullprod['country'])
print(not_nulluse['country'])


# In[12]:


not_nullprod.tail()


# #### Observations 2
# * The 25-26 countries that have enteries from 1960-1970 in both energy produced and consumed datasets are similar.
# * Also, i can see that some values have K, decimal, and M. These figures for Energy Produced are measured in Tonnes of oil equivalent per capita. That is to say, that `17.6k` from Italy in 1968 is `17.6k tonnes of oil` and `1.34M` from USA in 1968 is `1.34 million tonnes of oil`.
# * Case type in country column needs to be changed (for uniformity)
# 
# To do:
# * Drop columns 1960-1970
# * Change case type for the country column in all three datasets
# * Explore 2010 column and 2015 column in the energy produced and energy consumed datasets respectively
# * Change the values in the coulumns to integers, without k, M or decimal. `k` is interpreted as thousands while `M` is interpreted as million

# ### Cleaning Energy Production Dataset

# In[13]:


#dropping columns that we don't need in df_energy_prod dataset
df_energy_prod.drop(['1960','1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
       '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989'], axis=1, inplace=True)


# In[14]:


#changing case type for the country column in all three datasets
df_energy_prod['country'] = (df_energy_prod['country']
                             .str.lower()
                             .str.replace("-", "_")
                             .str.replace(" ", "_")
                             .str.replace(",","")
                             .str.replace(".","")
                            )

df_energy_usepp['country'] = (df_energy_usepp['country']
                             .str.lower()
                             .str.replace("-", "_")
                             .str.replace(" ", "_")
                             .str.replace(",","")
                             .str.replace(".","")
                            )
df_co2_consump['country'] = (df_co2_consump['country']
                             .str.lower()
                             .str.replace("-", "_")
                             .str.replace(" ", "_")
                             .str.replace(",","")
                             .str.replace(".","")
                            )


# In[15]:


#verifying dropped columns and case types
df_energy_prod.head()


# In[16]:


#understanding missing values in 2010 
not_nullprod_2010 = df_energy_prod[df_energy_prod['2010'].notnull()]
not_nullprod_2010


# These are the exact countries that had figures for 1960-1970. They probably have better reporting methods. I am going to drop the 2010 column as well.

# In[17]:


#dropping NaN values in 2010 column with 0
df_energy_prod.drop(['2010'], axis=1, inplace=True)


# In[18]:


#confirming the fillna
df_energy_prod.info()


# In[19]:


#removing Nan rows
df_energy_prod.dropna(inplace=True)
df_energy_prod.info()


# Now that the data set have been cleaned of NaN values, the next step is to eliminate the k, M and decimals in the values, and convert to `int` type.
# 
# To do:
# * Make the country column an index column
# * Assign all the columns to a variable
# * Use the replace function to eliminate the k, M and decimals

# In[20]:


#Setting index column
df_energy_prod =df_energy_prod.set_index('country')
df_energy_prod.info()


# In[21]:


cols1 = df_energy_prod.columns
cols1


# In[22]:


#Addressing k, M and the decimal, and ensuring that the values are integers
df_energy_prod = df_energy_prod[cols1].replace({'k': '*1e3', 'M': '*1e6', }, regex=True).applymap(pd.eval).round(2).astype(int)
df_energy_prod.head()


# In[23]:


# Verifying changes
df_energy_prod.info()


# In[24]:


df_energy_prod.describe()


# Min value shows 0 for each column. If a country did not produce energy within the 20 years relevant to this project, then they should not be studied. So, the next step will be to replace all zeros with NaN values.

# In[25]:


np.where(~df_energy_prod.any(axis=1))[0]


# In[26]:


# Resetting index column
df_energy_prod = df_energy_prod.reset_index()


# In[27]:


# Verifying
df_energy_prod.head(1)


# In[28]:


# Identifying the country with the empty column
df_energy_prod.iloc[47,0]


# In[29]:


# Setting index back to country column
df_energy_prod = df_energy_prod.set_index('country')


# In[30]:


# Dropping gibraltar
df_energy_prod.drop(['gibraltar'], inplace=True)


# In[31]:


#checking that there are no zero values
df_energy_prod.describe()


# In[32]:


df_energy_prod.info()


# There are still columns with zero, but now row is completely filled with zero. Thus, I will replace zeros with NaN as I envisage that I will work with means, and I want to ensure accurate mean and account for zero energy production.

# In[33]:


#replacing zeros with NaN to ensure accurate mean, and account for missing values.
df_energy_prod = df_energy_prod.replace(0, np.NaN) 


# In[34]:


df_energy_prod.describe()


# ### Cleaning Energy Consumption Dataset
# 
# > To clean the energy consumption dataset, I repeated an adapted version of the steps used to clean the energy production dataset. The steps are as follows:
# * Drop columns from 1960-1989
# * Explore missing values in 2015
# * Eliminate the k, M and decimals in the values by:
#  * Make the country column an index column
#  * Assign all the columns to a variable
#  * Use the replace function to eliminate the k, M and decimals
# * Convert values to`int` type
# * Adressing incomplete columns

# In[35]:


#overview of the energy use dataset
df_energy_usepp.head(3)


# In[36]:


#dropping columns that we don't need, to allign with the production dataset
df_energy_usepp.drop(['1960','1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', 
                      '1971', '1972', '1973', '1974', '1975', '1976', '1977','1978', '1979', '1980', '1981', 
                      '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989','2010', '2011', '2012', 
                      '2013', '2014', '2015'], axis=1, inplace=True)

df_energy_usepp.head(3)


# In[37]:


#exploring energy use dataset
df_energy_usepp.info()


# #### Observations 3
# Not one of the columns has complete data 
# 
# Filtering out the rows based on this will leave my dataset with roughly the same number of enteries like the energy production dataset. First, I will filter out the NaN rows, and then find out if the countries in cleaned `df_energy_usepp` and `df_energy prod` are the same.

# In[38]:


#removing Nan rows
df_energy_usepp.dropna(inplace=True)
df_energy_usepp.info()


# In[39]:


# Resetting index column
df_energy_prod = df_energy_prod.reset_index()
# Comparing
df_energy_usepp['country'].isin(df_energy_prod['country']).value_counts()


# There are 3 countries in the energy consumption dataset that are not in the energy production set

# In[40]:


extra = df_energy_usepp[df_energy_usepp['country'].isin(df_energy_prod['country']) == False]
extra


# In[41]:


#Setting index column
df_energy_prod = df_energy_prod.set_index('country')
df_energy_usepp = df_energy_usepp.set_index('country')


# Now to address the k, M, and decimals...

# In[42]:


cols2 = df_energy_usepp.columns
cols2


# In[43]:


#Addressing the k, M, and decimals

df_energy_usepp = df_energy_usepp[cols2].replace({'k': '*1e3', 'M': '*1e6', }, regex=True).applymap(pd.eval).round(2).astype(int)

df_energy_usepp.head()


# In[44]:


#Verifying changes
df_energy_usepp.info()


# In[45]:


df_energy_usepp.describe()


# ### Cleaning Consumption CO2 per capita Dataset

# In[46]:


#Exploring CO2 dataset
df_co2_consump.tail()


# In[47]:


#Exploring CO2 dataset
df_co2_consump.info()


# In[48]:


df_co2_consump.describe()


# #### Observations
# * The values have decimals, but they're float type
# * There are no missing values, but there are zero values. It is however alsmost impossible for a country to not emit CO2. It could just be that it wasn't accounted for.
# * There are no values with k, or M
# * This dataset has columns up to 2017
# * The dataset has 119 enteries, which is smaller than either of the previous ones
# 
# To do:
# * Make the country column an index column,
# * Change the values to int type.
# * Make zeros NaN

# In[49]:


# Setting index
df_co2_consump = df_co2_consump.set_index('country')


# In[50]:


# Converting to integers
df_co2_consump = df_co2_consump.round(2).astype(int)
#replacing zeros with NaN to ensure accurate mean, and account for missing values.
df_co2_consump = df_co2_consump.replace(0, np.NaN)


# In[51]:


df_co2_consump.tail()


# In[52]:


df_co2_consump.describe()


# #### Observations
# Converting the values to integers approximated the values. This is important especially for CO2 because consumption CO2 per capita is measured in metric tons of CO2 per person, and 1 unit of this has significant impact on the environment and climate change

# <a id='eda'></a>
# ## Exploratory Data Analysis

# ### Top and Low Energy Producing Countries

# #### Top Energy Producing Countries

# In[53]:


#finding the mean of 2 decades of production
df_energy_prod['country_mean'] = df_energy_prod.mean(axis=1)
df_energy_prod.head(5)


# In[54]:


#Exploring country mean
df_energy_prod['country_mean'].describe()


# In[55]:


df_energy_prod.query('country_mean == "0"')


# In[56]:


#Grouping countries based on their mean energy production
bin_edges = [ 1.000000, 3010.375, 13355.00, 58907.50, 1661000.00 ]


# In[57]:


# Labels for the four levels of production groups
bin_names = ['low producer','medium producer', 'high producer', 'top producer' ] 


# In[58]:


# Creates prod_levels column
df_energy_prod['prod_levels'] = pd.cut(df_energy_prod['country_mean'], bin_edges, labels=bin_names)

# Checks for successful creation of this column
df_energy_prod.head(3)


# In[59]:


#grouping top producers
top_prod = df_energy_prod.query('prod_levels == "top producer"')
top_prod = top_prod.sort_values(by = ['country_mean', 'country'], ascending = [False, True])
top_prod.head(6)


# In[60]:


top_prod.index


# In[61]:


top_prod["country_mean"].plot(kind='pie', title='Average Production of Top Energy Producing Countries', figsize=(10,10))


# In[62]:


#plotting top 5 accross the years
united_states = top_prod.iloc[0, :-2]
united_states.plot(kind='bar', title='Energy Production in United States')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

china = top_prod.iloc[1, :-2]
china.plot(kind='bar', title='Energy Production in China')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

russia = top_prod.iloc[2, :-2]
russia.plot(kind='bar', title='Energy Production in Russia')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

saudi_arabia = top_prod.iloc[3, :-2]
saudi_arabia.plot(kind='bar', title='Energy Production in Saudi Arabia')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

india = top_prod.iloc[4, :-2]
india.plot(kind='bar', title='Energy Production in India')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()


# #### Observation 4
# While countries like the USA and Russia have maintained production levels within the same range over the years, China and India have gradually increased their production over the years.

# #### Low Energy Producing Countries

# In[63]:


#finding the mean of 2 decades of production
low_prod = df_energy_prod.query('prod_levels == "low producer"')
low_prod = low_prod.sort_values(by = ['country_mean', 'country'], ascending = [True, False])
low_prod.head(6)


# In[64]:


low_prod.index


# In[65]:


singapore = low_prod.iloc[0, :-3]
singapore.plot(kind='bar', title='Energy Production in Singapore')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

cyprus = low_prod.iloc[1, :-3]
cyprus.plot(kind='bar', title='Energy Production in Cyprus')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

hong_kong_china = low_prod.iloc[2, :-3]
hong_kong_china.plot(kind='bar', title='Energy Production in Hong Kong, China')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

luxembourg = low_prod.iloc[3, :-3]
luxembourg.plot(kind='bar', title='Energy Production in Luxembourg')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()

moldova = low_prod.iloc[4, :-3]
moldova.plot(kind='bar', title='Energy Production in Moldova')
plt.xlabel('Years')
plt.ylabel('Energy Produced (Tonnes of Oil Equivalent)')
plt.show()


# #### Observation 5
# All the low energy producing countries have actually increased their energy production over the years

# ### Top and Low Energy Consumers

# #### Top Energy Consumers

# In[66]:


#replacing zeros with NaN to ensure accurate mean, and account for missing values.
df_energy_usepp = df_energy_usepp.replace(0, np.NaN)
#finding the mean of 2 decades of energy consumption
df_energy_usepp['country_mean'] = df_energy_usepp.mean(axis=1)
df_energy_usepp.head(5)


# In[67]:


#Exploring country mean
df_energy_usepp['country_mean'].describe()


# In[68]:


#Grouping countries based on their mean energy production
bin_edgess = [ 149.200000, 644.550000, 1501.800000, 3449.500000, 17725.000000 ] # Fill in this list with five values


# In[69]:


# Labels for the four levels of production groups
bin_namess = ['low consumer','medium consumer', 'high consumer', 'top consumer' ] 


# In[70]:


# Creates usepp_levels column
df_energy_usepp['usepp_levels'] = pd.cut(df_energy_usepp['country_mean'], bin_edgess, labels=bin_namess)

# Checks for successful creation of this column
df_energy_usepp.head(3)


# In[71]:


#grouping top producers
top_usepp = df_energy_usepp.query('usepp_levels == "top consumer"')
top_usepp = top_usepp.sort_values(by = ['country_mean', 'country'], ascending = [False, True]) # Sorting
top_usepp.head(6)


# In[72]:


top_usepp.index


# In[73]:


#plotting top 5 accross the years

qatar = top_usepp.iloc[0, :-2]
qatar.plot(kind='bar', title='Energy Consumption in Qatar')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

curaçao = top_usepp.iloc[1, :-2]
curaçao.plot(kind='bar', title='Energy Consumption in Curaçao')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

iceland = top_usepp.iloc[2, :-2]
iceland.plot(kind='bar', title='Energy Consumption in Iceland')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()


bahrain = top_usepp.iloc[3, :-2]
bahrain.plot(kind='bar', title='Energy Consumption in Bahrain')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

united_arab_emirates = top_usepp.iloc[4, :-2]
united_arab_emirates.plot(kind='bar', title='Energy Consumption in United Arab Emirates')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()


# Energy consumption has dropped in UAE and Qatar since the early 2000s, but this may be more about population, since energy consumption is per person per country.

# #### Low Energy Consumers

# In[74]:


#finding the mean of 2 decades of energy consumption
low_usepp = df_energy_usepp.query('usepp_levels == "low consumer"')
low_usepp = low_usepp.sort_values(by = ['country_mean', 'country'], ascending = [True, False])
low_usepp.head()


# In[75]:


low_usepp.index


# In[76]:


medium_usepp = df_energy_usepp.query('usepp_levels == "medium consumer"')
high_usepp = df_energy_usepp.query('usepp_levels == "high consumer"')


# In[77]:


medium_usepp.index


# In[78]:


high_usepp.index


# In[79]:


#plotting lowest 5 accross the years
senegal = low_usepp.iloc[0, :-2]
senegal.plot(kind='bar', title='Energy Consumption in Senegal')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

haiti = low_usepp.iloc[1, :-2]
haiti.plot(kind='bar', title='Energy Consumption in Haiti')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

yemen = low_usepp.iloc[2, :-2]
yemen.plot(kind='bar', title='Energy Consumption in Yemen')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

congo_rep = low_usepp.iloc[3, :-2]
congo_rep.plot(kind='bar', title='Energy Consumption in congo_rep')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()

congo_dem_rep = low_usepp.iloc[4, :-2]
congo_dem_rep.plot(kind='bar', title='Energy Consumption in Congo Dem Rep')
plt.xlabel('Years')
plt.ylabel('Energy Consumed (Kg of oil equivalent per capita)')
plt.show()


# ### Exploring Consumption CO2 per capita

# #### Top Consumption CO2 per capita countries

# In[80]:


#finding the mean of of Consumption CO2 per capita
df_co2_consump = df_co2_consump.replace(0, np.NaN)
df_co2_consump['country_mean'] = df_co2_consump.mean(axis=1)
df_co2_consump.head(5)


# In[81]:


#Exploring country mean
df_co2_consump['country_mean'].describe()


# In[82]:


#Grouping countries based on their mean energy production
bin_edgess = [ 1.00, 1.732143, 5.285714, 10.482143, 33.928571 ] 


# In[83]:


# Labels for the four levels of Consumption CO2 per capita groups
bin_namess = ['low CO2','medium CO2', 'high CO2', 'top CO2' ] 


# In[84]:


# Creates CO2_levels column
df_co2_consump['CO2_levels'] = pd.cut(df_co2_consump['country_mean'], bin_edgess, labels=bin_namess)

# Checks for successful creation of this column
df_co2_consump.head(3)


# In[85]:


#grouping top CO2
top_co2 = df_co2_consump.query('CO2_levels == "top CO2"')
top_co2 = top_co2.sort_values(by = ['country_mean', 'country'], ascending = [False, True])
top_co2.head(5)


# In[86]:


top_co2.index


# In[87]:


#plotting top 5 accross the years
luxembourg = top_co2.iloc[0, :-2]
luxembourg.plot(kind='bar', title='Consumption CO2 per capita in Luxembourg')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

united_arab_emirates = top_co2.iloc[1, :-2]
united_arab_emirates.plot(kind='bar', title='Consumption CO2 per capita in United Arab Emirates')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

singapore = top_co2.iloc[2, :-2]
singapore.plot(kind='bar', title='Consumption CO2 per capita in Singapore')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

kuwait = top_co2.iloc[3, :-2]
kuwait.plot(kind='bar', title='Consumption CO2 per capita in Kuwait')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

united_states = top_co2.iloc[4, :-2]
united_states.plot(kind='bar', title='Consumption CO2 per capita in United States')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()


# #### Low Consumption CO2 per capita countries

# In[88]:


#finding the mean of of Consumption CO2 per capita
low_co2 = df_co2_consump.query('CO2_levels == "low CO2"')
low_co2 = low_co2.sort_values(by = ['country_mean', 'country'], ascending = [True, False])
low_co2.head()


# In[89]:


low_co2.index


# In[90]:


#plotting lowest 5 accross the years
zimbabwe = low_co2.iloc[0, :-2]
zimbabwe.plot(kind='bar', title='Consumption CO2 per capita in Zimbabwe')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

peru = low_co2.iloc[1, :-2]
peru.plot(kind='bar', title='Consumption CO2 per capita in Peru')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()


indonesia = low_co2.iloc[2, :-2]
indonesia.plot(kind='bar', title='Consumption CO2 per capita in Indonesia')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

colombia = low_co2.iloc[3, :-2]
kuwait.plot(kind='bar', title='Consumption CO2 per capita in Colombia')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()

vietnam = low_co2.iloc[4, :-2]
vietnam.plot(kind='bar', title='Consumption CO2 per capita in Vietnam')
plt.xlabel('Years')
plt.ylabel('Consumption CO2 per capita (metric tons of CO2 per person)')
plt.show()


# In[91]:


medium_co2 = df_co2_consump.query('CO2_levels == "medium CO2"')
high_co2 = df_co2_consump.query('CO2_levels == "high CO2"')


# In[92]:


medium_co2.index


# In[93]:


high_co2.index


# <a id='conclusions'></a>
# ## Conclusions
# 
# The main aim of this project was to explore trends on energy production, consumption and CO<sub>2</sub> emissions within two decades (1990-2009) from around the world. To achieve this aim, the following questions were asked:
# 
# * Which countries are the top and least energy producers?
# * Which countries consume the most and least energy?
# * Which countries are the highest and lowest CO<sub>2</sub> emitters? 
# 
# Here are the main findings from the 3 datasets which provided information on energy production, consumption and CO<sub>2</sub> emission from around the world:
# 
# > **Two decades of Energy Production**: In this project it was seen that between 1990 and 2009, the top 5 energy producers were the United States, China, Russia, Saudi Arabia, India and Canada. It was also seen that the lowest producers were Singapore, Cyprus, Hong Kong China, Luxembourg and Moldova.
# 
# > **Two decades of Energy Use Per Person**: Interestingly, none of the top 5 energy producers made it to the top 5 energy consumers as the top 5 consumers of Energy were Qatar, Curaçao, Bahrain, Iceland and United Arab Emirates. Canada and United States on the list of Top Energy Consumers were at position 7 and 8 respectively. This suggests that the top 5 energy producers may not be consuming all the energy that they are producing. Also, it is important to restate that this indicator, Energy Use Per person, is a function of the country's population. A suggestion for further studies would be to place consumption figures at par with population figures. Another interesting insight is that all the top energy producers and consumers are either developed countries or high developing countries. Moving now to low energy consumers, the top 5 were Senegal, Haiti, Yemen, Myanmar and Congo Rep - all of which are low developing countries. 
# 
# > **Two decades of CO<sub>2</sub> Emissions per Capita Per Person**: The top 5 CO<sub>2</sub> emitters are Luxembourg, United Arab Emirates, Singapore, Kuwait, the United States. Canada made the 6th position. To avoid implying causation where the situation may be more linked to corelation, it is important to  highlight that energy production or consumption is not always the cause of CO<sub>2</sub> emission. Thus, in higlighting trends, a key pointis that 2 countries, Singapore and Luxemborg, that were listed among the top 5 lowest energy producers are also among the top 5 CO<sub>2</sub> emitters. Again, recall that this indicator is also a funtion of population and consumption of CO<sub>2</sub>. Also, the United Arab Emirates, a top CO<sub>2</sub> emitter, is among the top 5 energy consumers. Low CO<sub>2</sub> emitters include Zimbabwe, Peru, Indonesia, Colombia, and Vietnam.
