from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, precision_score

from scipy.stats import mannwhitneyu, kruskal, shapiro, kstest
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import pandas as pd
import numpy as np

def shapiro_test(records:pd.DataFrame) -> pd.DataFrame:
    """
    Shapiro-Wilk test for normality (SciPy)
    ---
    Tests the null hypothesis that the data was drawn from a normal distribution.
    Base requirement for parametric tests like Student's T. 
    """
    results = {}
    for compound in records.columns:
        stat, p_value = shapiro(records[compound])
        results[compound] = (stat, p_value)
            
    return pd.DataFrame(results, index=['Statistic', 'p-value']).T

def kruskal_test(records:pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Kruskal-Wallis test by ranks (SciPy) with Dunn's Post-Hoc Test (Scikit-Posthocs)
    ---
    Also known as one-way ANOVA on ranks. Non-parametric method for testing
    whether samples originate from the same distribution.
    
    Dunn's test helps to identify which specific group differs the most, and
    is similar to Tukey's Post-Hoc test for ANOVA. It also performs multiple test
    corrections, and helps to reduce false discoveries. 
    """
    
    labels = []
    grouped_data = []
    grouped_records = records.groupby('Label')
    for label, group in grouped_records:
        labels.append(label)
        grouped_data.append(group.drop('Label', axis=1).values)
    results = kruskal(*grouped_data)
    
    posthoc_results = {}
    no_label = records.drop('Label', axis=1)
    for compound in no_label.columns:
        posthoc = posthoc_dunn(records, group_col='Label', val_col=compound, p_adjust='fdr_bh')
        is_critical = posthoc.values.min() < 0.05
        posthoc_results[compound] = is_critical
    
    results_dict = {}
    for idx, compound in enumerate(no_label.columns):
        results_dict[compound] = (results.statistic[idx], results.pvalue[idx], posthoc_results[compound])
    
    output = pd.DataFrame(results_dict, index=['Statistic', 'p-value', 'Dunn']).T
    output = output[output['p-value'] < 0.05]
    output = output[output['Dunn'] == True]
    output = output.sort_values(by='p-value', ascending=True)
    
    return output

def mwhitney_test(records:pd.DataFrame):
    """
    Mann-Whitney U-test (SciPy) with Bonferonni Correction (SciPy)
    ---
    Non-parametric test to compare two independent groups.
    """
    output = {}
    groups = records['Label'].unique()
    for compound in records.columns:
        if compound == 'Label':
            continue
        
        compound_data = records[compound]
        for group_a, group_b in combinations(groups, 2):
            group_a_data = compound_data[records['Label'] == group_a]
            group_b_data = compound_data[records['Label'] == group_b]
            stat, p_val = mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
            key = f'{group_a}_{group_b}'
            if key not in output:
                output[key] = []
            
            output[key].append((compound, stat, p_val))
        
    output_df = {key: pd.DataFrame(data, columns=['Compound', 'U-statistic', 'p-value']) for key, data in output.items()}
    for df in output_df.values():
        df.set_index('Compound', inplace=True)
        df.sort_values('p-value', ascending=True, inplace=True)
        _, df['Adjusted p-value'], _, _ = multipletests(df['p-value'], method='bonferroni')
    
    return output_df
    
def pca(records:pd.DataFrame, n_components:int=10):
    """
    Principal Component Analysis (Scikit-learn)
    ---
    """
    pca = PCA(n_components=n_components).fit(records)
    pca_data = pd.DataFrame(pca.transform(records), index=records.index, columns=[f'PC{i + 1}' for i in range(n_components)])
    return pca, pca_data

def tsne(records:pd.DataFrame, perplexity):
    """
    t-Distributed Stochastic Neighbour Embedding (Scikit-learn)
    ---
    Does exactly what you would expect. Remember to remove the `Label` column from
    the DataFrame created by `capstone_data.to_df()` for this to work properly.
    """
    tsne_data = pd.DataFrame(TSNE(perplexity=perplexity).fit_transform(records), index=records.index)
    return tsne_data

def svc_rfe(records:pd.DataFrame, n_compounds:int=30):
    """
    Support Vector Classifier Recursive Feature Elimination (Scikit-learn)
    ---
    Based on a paper by I. Guyon, published in 2002. 
    
    Recursively removes 20% of the weakest features every iteration until
    `n_compounds` features are left. Returns a list of ranks for each compound,
    where `1` represents the most significant set of compounds. 
    """
    labels = LabelEncoder().fit_transform(records['Label'])
    no_labels = records.drop('Label', axis=1)
    values = no_labels.values
    estimator = SVC(kernel='linear')
    selector = RFE(estimator, n_features_to_select=n_compounds, step=0.2).fit(values, labels)
    compound_dict = {}
    for idx, compound in enumerate(no_labels.columns):
        compound_dict[compound] = selector.ranking_[idx]
    output = pd.DataFrame.from_dict(compound_dict, orient='index', columns=['Ranking'])
    output = output.sort_values('Ranking', ascending=False)
    return output

def random_forest(records:pd.DataFrame, n_compounds:int=30):
    """
    Feature Ranking by Random Forest (Scikit-learn)
    ---
    A leave-one-out cross validation is applied to this method. It gives the best results,
    but it is computationally expensive, and will take a long time to complete. 
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(records['Label'])
    no_labels = records.drop('Label', axis=1)
    np_data = no_labels.to_numpy()
    
    loo = LeaveOneOut()
    accuracy_scores = []
    for train_idx, test_idx in loo.split(no_labels):
        train_data, test_data = np_data[train_idx], np_data[test_idx]
        train_label, test_label = labels[train_idx], labels[test_idx]
        
        forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
        forest.fit(train_data, train_label)
        
        pred = forest.predict(test_data)
        accuracy = accuracy_score(test_label, pred)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f'Average accuracy: {avg_accuracy}')
    
    compound_dict = {}
    feature_importances = forest.feature_importances_
    for idx, compound in enumerate(no_labels.columns):
        compound_dict[compound] = feature_importances[idx]
    
    output = pd.DataFrame.from_dict(compound_dict, orient='index', columns=['Significance'])
    output = output.sort_values('Significance', ascending=False)
    return output.head(n_compounds)

def permut_rf(records:pd.DataFrame, n_compounds:int=30):
    """
    Feature Ranking by Permutational Random Forest (Scikit-learn)
    ---
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(records['Label'])
    no_labels = records.drop('Label', axis=1)
    np_data = no_labels.to_numpy()
    
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    forest.fit(np_data, labels)
    result = permutation_importance(forest, np_data, labels, n_repeats=10, n_jobs=-1, random_state=42)
    result_dict = {}
    for idx, compound in enumerate(no_labels.columns):
        result_dict[compound] = result.importances_mean[idx]
    output = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Significance'])
    output = output.sort_values('Significance', ascending=False)
    return output.head(n_compounds)

def random_forest_rfe(records:pd.DataFrame, n_compounds:int=30):
    """
    Random Forest with Recursive Feature Elimination (Scikit-learn)
    ---
    Similar in context to SVM-RFE, but uses a random forest as the estimator.
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(records['Label'])
    no_labels = records.drop('Label', axis=1)
    np_data = no_labels.to_numpy()
    
    forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)
    rfe = RFE(estimator=forest, n_features_to_select=n_compounds, step=0.2).fit(np_data, labels)
    # selected_features = [feature for feature, selected in zip(no_labels.columns, rfe.support_) if selected]
    feature_ranks = rfe.ranking_
    output = pd.DataFrame({'Compound': no_labels.columns, 'Rank': feature_ranks})
    output.set_index('Compound', inplace=True)
    return output
