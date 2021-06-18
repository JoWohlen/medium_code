####
# Keep in mind that you need to get the french mortality dataset in the same folder this script is in, to run it properly
# You can find the link to the mortality database in my article 
# https://towardsdatascience.com/functional-principal-component-analysis-and-functional-data-91d21261ab7f
#
#
# make shure to also check out the below linked example from the fda library. It properly explains the process of using fPCA
# https://fda.readthedocs.io/en/latest/auto_examples/plot_fpca.html#sphx-glr-auto-examples-plot-fpca-py
####

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfda
from skfda.exploratory.visualization import plot_fpca_perturbation_graphs
from skfda.preprocessing.dim_reduction.projection import FPCA

sns.set_theme()
sns.set_style("whitegrid", {'axes.grid': False})

df = pd.read_csv("data.txt", delim_whitespace=True)
df = df.replace('.', 0)
df["Male"] = pd.to_numeric(df["Male"], downcast="float")
df = df[df['Male'] > 0]


#bring the data into a proper matrix format
#as we cut off motality of ages greater 100
data_matrix = []
for year in df['Year'].unique():
    rate = df.loc[df['Year'] == year]['Male']._values
    rate = np.log(rate)
    data_matrix.append(rate[0:100])

# create an FDataGrid which we later on need to estimate the 
# functional representation of the data
data_matrix = np.asarray(data_matrix)
data_matrix = data_matrix.astype(float)
fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=range(0, 100))
fd.plot()
plt.xlabel("Age")
plt.ylabel("log Mortality Rate")
plt.title("French Male log Mortality Rate from 1816 - 2018")
plt.show()

#estimate the underlying functional representations
#following up, we will work with the basis representation
basis = skfda.representation.basis.BSpline(n_basis=20)
basis_fd = fd.to_basis(basis)
fig = basis_fd.plot(legend=True)
plt.xlabel("Age")
plt.ylabel("log Mortality Rate")
plt.title("French Male log Mortality Rate from 1816 - 2018")
plt.show()

#fit FPCA and vizualize the first 2 functional principal components
fpca = FPCA(n_components=2)
fpca.fit(basis_fd)
fpca.components_.plot()
plt.legend(labels=["component 1 - 0.94",'component 2 - 0.03'])
plt.title('functional Principal Components')
plt.xlabel("Age")
print(fpca.explained_variance_ratio_)
plt.show()

#get the scores and vizualize them
scores = fpca.fit_transform(basis_fd)
sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=df['Year'].unique(), palette='icefire')
plt.legend()
plt.xlabel("fPC 1 score")
plt.ylabel("fPC 2 score")
plt.title("")
plt.show()
