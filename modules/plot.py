from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_pca(model, data:pd.DataFrame, loading_min:float=0.2, show:bool=True, save:bool=False):
    loadings = pd.DataFrame.from_dict(dict(zip([f'PC{i+1}' for i in range(model.n_components_)], model.components_)), orient='index', columns=model.feature_names_in_).T
    loadings['Length'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
    loadings = loadings[loadings['Length'] >= loading_min]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    colors = ['red' if '0h' in idx else 'blue' if '3h' in idx else 'green' if '9h' in idx else 'orange' for idx in data.index]
    markers = ['^' if 'BS' in idx else 's' if 'SA' in idx else 'o' if 'EC' in idx else 'x' for idx in data.index]
    for x, y, color, marker in zip(data.iloc[:, 0], data.iloc[:, 1], colors, markers):
        ax1.scatter(x, y, c=color, marker=marker, alpha=0.7)
    ax1.set_title('Score Plot')
    ax1.set_xlabel(f'PC1, {model.explained_variance_ratio_[0] * 100:.2f}%')
    ax1.set_ylabel(f'PC2, {model.explained_variance_ratio_[1] * 100:.2f}%')
    xabs_max = abs(max(ax1.get_xlim(), key=abs))
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_xlim(xmin=-xabs_max, xmax=xabs_max)
    ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    for x, y, label in zip(loadings.iloc[:, 0], loadings.iloc[:, 1], loadings.index):
        ax2.arrow(x=0, y=0, dx=x, dy=y, head_width=0.01)
        ax2.text(x, y, label)
    ax2.add_patch(plt.Circle((0, 0), radius=1, edgecolor='r', facecolor='None'))
    ax2.set_title('Loading Plot')
    ax2.set_xlabel(f'PC1, {model.explained_variance_ratio_[0] * 100:.2f}%')
    ax2.set_ylabel(f'PC2, {model.explained_variance_ratio_[1] * 100:.2f}%')

    plt.subplots_adjust(wspace=0.5)
    if show:
        plt.show()

def plot_biplot(model, data:pd.DataFrame, loading_min:float=0.2, title:str=None, show:bool=True, save:bool=False):
    loadings = pd.DataFrame.from_dict(dict(zip([f'PC{i+1}' for i in range(model.n_components_)], model.components_)), orient='index', columns=model.feature_names_in_).T
    loadings['Length'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
    loadings = loadings[loadings['Length'] >= loading_min]
    
    scaled_data = pd.DataFrame(MinMaxScaler((-1, 1)).fit_transform(data), index=data.index, columns=[f'PC{i+1}' for i in range(len(data.columns))])
    scaled_data['Environment'] = ['Ae' if 'Ae' in idx else 'An' for idx in scaled_data.index]
    scaled_data['Medium'] = ['LB' if 'LB' in idx else 'TSA' if 'TSA' in idx else 'TSB' if 'TSB' in idx else 'MSA' for idx in scaled_data.index]
    scaled_data['Species'] = ['BS' if 'BS' in idx else 'EC' if 'EC' in idx else 'SA' if 'SA' in idx else 'Ctrl' for idx in scaled_data.index]
    
    fig, ax = plt.subplots()
    sns.scatterplot(x='PC1', y='PC2', hue='Species', data=scaled_data, alpha=0.7, s=100, legend='full', ax=ax)        
    for x, y, label in zip(loadings.iloc[:, 0], loadings.iloc[:, 1], loadings.index):
        ax.arrow(x=0, y=0, dx=x, dy=y, head_width=0.01)
        ax.text(x, y, label)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f'PC1, {model.explained_variance_ratio_[0] * 100:.2f}%')
    ax.set_ylabel(f'PC2, {model.explained_variance_ratio_[1] * 100:.2f}%')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.rcParams['figure.dpi'] = 300
    if save:
        plt.savefig()
    if show:
        plt.show()

def plot_time(data:pd.DataFrame, title:str=None, show:bool=True, save:bool=False):
    ax = sns.lineplot(data.T)
    if title is not None:
        ax.title(title)
    ax.set_xlabel('Collection Time (h)')
    ax.set_xticks([tick for tick in data.T.index if tick % 3 == 0])
    ax.set_ylabel('Peak Area')
    # ax.set_yscale('log')
    ncol = max(1, int(len(data.index) / 10))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol)
    if show:
        plt.show()