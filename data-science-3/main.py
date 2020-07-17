#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[179]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
fifa.info()


# In[6]:


fifa.describe()


# In[7]:


fifa[fifa.Crossing.isna()].head()


# In[10]:


fifa.dropna(inplace=True)
fifa.shape


# In[74]:


# Análise da questão 1
pca = PCA(n_components=1)
pca.fit(fifa)
pca.explained_variance_ratio_


# In[18]:


# Análise da questão 2
pca_95 = PCA(n_components=0.95)
pca_95.fit(fifa)
np.cumsum(pca_95.explained_variance_ratio_)


# In[177]:


# Análise da questão 3
pca_x = PCA(n_components=2)
pca_x.fit(fifa)
#sns.scatterplot(pca_x.components_[0], pca_x.components_[1]);
#sns.scatterplot([x[0]], [x[1]]);


# In[180]:


# Análise da questão 4
reg = LinearRegression()
y = fifa.Overall
x = fifa.drop(columns='Overall')
rfe = RFE(reg, n_features_to_select=1)
selector_o = rfe.fit(x, y)
ranking = pd.Series(x.columns, index=selector_o.ranking_, name='Overall')
ranking.sort_index().head()


# In[181]:


y_p = fifa.Potential
x_p = fifa.drop(columns='Potential')
selector = rfe.fit(x_p, y_p)
ranking_p = pd.Series(x_p.columns, index=selector.ranking_, name='Potential')
ranking_p.sort_index().head()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[21]:


def q1():
    # Retorne aqui o resultado da questão 1.
    pca = PCA(n_components=1)
    pca.fit(fifa)
    pc1_variance_ratio = pca.explained_variance_ratio_.round(3)
    return float(pc1_variance_ratio)


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[24]:


def q2():
    # Retorne aqui o resultado da questão 2.
    pca_95 = PCA(0.95)
    pca_95.fit(fifa)
    return int(pca_95.n_components_)


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[172]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[173]:


def q3():
    # Retorne aqui o resultado da questão 3.
    pca_x = PCA(n_components=2)
    pca_x.fit(fifa)
    x_coord = pca_x.components_.dot(x).round(3)
    return tuple(x_coord)


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[175]:


def q4():
    # Retorne aqui o resultado da questão 4.
    reg = LinearRegression()
    y = fifa.Overall
    x = fifa.drop(columns='Overall')
    rfe = RFE(reg, n_features_to_select=1)
    selector = rfe.fit(x, y)
    ranking = pd.Series(x.columns, index=selector.ranking_)
    return ranking.sort_index().head().to_list()

