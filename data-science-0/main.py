#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.shape


# In[79]:


black_friday


# In[5]:


black_friday.describe()


# In[6]:


black_friday.info()


# In[20]:


#Testes da questão: 2
#%timeit black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')].shape[0]


# In[21]:


#%timeit sum((black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35'))


# In[121]:


#Testes da questão: 3
black_friday[black_friday["Product_Category_2"].isna() | black_friday["Product_Category_3"].isna()].shape[0]


# In[48]:


#Testes da questão: 4
black_friday.isna().any(axis='columns').value_counts(normalize=True)


# In[118]:


#Testes da questão: 5
black_friday.isna().sum().max()


# In[120]:


#Testes da questão: 6
black_friday['Product_Category_3'].mode().item()


# In[125]:


#Testes da questão: 7
(black_friday['Purchase'] / black_friday['Purchase'].max()).mean()


# In[22]:


#Testes da questão: 8
#%timeit (black_friday['Purchase'].mean() - black_friday['Purchase'].min()) / \
#                 (black_friday['Purchase'].max() - black_friday['Purchase'].min())


# In[24]:


#%timeit ((black_friday['Purchase'] - black_friday['Purchase'].min()) / \
#                 (black_friday['Purchase'].max() - black_friday['Purchase'].min())).mean()


# In[181]:


#Testes da questão: 9
nomal_purchase = (black_friday['Purchase'] - black_friday['Purchase'].mean()) /                 black_friday['Purchase'].std()
nomal_purchase.between(-1, 1).value_counts()[True].item()


# In[209]:


#Testes da questão: 10
aux = black_friday[['Product_Category_2', 'Product_Category_3']].isna()
aux[aux['Product_Category_2'] == True].all()[1]


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape

    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[118]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[(black_friday['Gender'] == 'F') & 
                        (black_friday['Age'] == '26-35')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[78]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[47]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[183]:


def q5():
    # Retorne aqui o resultado da questão 5.
    nan_percent = black_friday.isna().any(axis='columns').value_counts(normalize=True)
    return nan_percent[True].item()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[186]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[185]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].mode().item()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[136]:


def q8():
    # Retorne aqui o resultado da questão 8.
    mean_normal_purchase = (black_friday['Purchase'].mean() - black_friday['Purchase'].min()) /                            (black_friday['Purchase'].max() - black_friday['Purchase'].min())
    return float(mean_normal_purchase)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[182]:


def q9():
    # Retorne aqui o resultado da questão 9.
    normal_purchase = (black_friday['Purchase'] - black_friday['Purchase'].mean()) /                      black_friday['Purchase'].std()
    return normal_purchase.between(-1, 1).value_counts()[True].item()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[211]:


def q10():
    # Retorne aqui o resultado da questão 10.
    aux = black_friday[['Product_Category_2', 'Product_Category_3']].isna()
    return aux[aux['Product_Category_2'] == True].all()[1].item()

