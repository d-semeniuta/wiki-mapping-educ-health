import pandas as pd
import util_data
import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.orca.config.use_xvfb = True
from sklearn.decomposition import PCA

OUT_DIR = os.path.join(os.path.curdir, 'visualization', 'out', 'pca')

def get_embedding_matrix(embedding_dir):
    embs = []
    for filename in os.listdir(embedding_dir):
        # article_idx = filename.split('.')[0]
        emb = np.load(os.path.join(embedding_dir, filename))
        embs.append(emb)
    emb_mat = np.matrix(embs)
    return emb_mat

def run_pca(emb_mat, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(emb_mat)
    return pca

def plot_pca(pc1s, pc2s, scores, task, country, img_format='svg'):
    fig = go.Figure(data=go.Scatter(
        x = pc1s,
        y = pc2s,
        mode='markers',
        marker=dict(
            size=16,
            color=scores, #set color equal to a variable
            colorscale='Viridis', # one of plotly colorscales
            showscale=True,
            colorbar_title = 'Maternal Education Score' if task=='MatEd' else 'Infant Mortality Rate',
            cmin = 0,
            cmax = 3 if task=='MatEd' else 1
        )
    ))
    title = '{} in {}'.format(task, country)
    fig.update_layout(title=title,
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title='First Principal Component'),
                    yaxis=dict(title='Second Principal Component'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Gray')
    img_name = title.replace(' ', '_').lower()
    fig.write_image(os.path.join(OUT_DIR, '{}_pca.{}'.format(img_name, img_format)))

    fig.show()

def do_country_level_analysis(dataset, task, country, pca):
    pc1s, pc2s, scores = [], [], []
    coords = set()
    for cluster in dataset:
        article_embs, label = cluster['x'], cluster['y']
        article_embs = article_embs.reshape((-1, 300)).mean(axis=0, keepdims=True)
        pc1, pc2 = pca.transform(article_embs)[0]
        coords.add((pc1,pc2))
        pc1s.append(pc1)
        pc2s.append(pc2)
        if task == 'MatEd':
            ed_score = 0
            for i in range(4):
                ed_score += i * label[i]
            scores.append(ed_score)
        else:
            scores.append(label[0])
    plot_pca(pc1s, pc2s, scores, task, country)

def pca_on_all_articles():
    train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
    article_embeddings_dir = os.path.join(os.curdir, 'raw', 'wikipedia', 'doc2vec_embeddings')
    cluster_article_rank_dist_path = os.path.join(os.curdir, 'processed', 'ClusterNearestArticles_Rank_Dist.csv')

    emb_mat = get_embedding_matrix(article_embeddings_dir)
    pca = run_pca(emb_mat)

    NUM_ARTICLES = 5
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    for task in ['MatEd', 'IMR']:
        for country in countries:
            dataset = util_data.DHS_Wiki_Dataset(DHS_csv_file=train_path,
                        emb_root_dir=article_embeddings_dir, cluster_rank_csv_path=cluster_article_rank_dist_path,
                        emb_dim=300, n_articles=NUM_ARTICLES, include_dists=False,
                        country_subset=[country], task=task,
                        transforms=None)
            do_country_level_analysis(dataset, task, country, pca)

def do_country_level_analysis_internal(dataset, task, country):
    embs = []
    scores = []
    for cluster in dataset:
        article_embs, label = cluster['x'], cluster['y']
        # article_embs = article_embs.reshape((-1, 300)).mean(axis=0, keepdims=True)
        embs.append(article_embs)
        # pc1, pc2 = pca.transform(article_embs)[0]
        # coords.add((pc1,pc2))
        # pc1s.append(pc1)
        # pc2s.append(pc2)
        if task == 'MatEd':
            ed_score = 0
            for i in range(4):
                ed_score += i * label[i]
            scores.append(ed_score)
        else:
            scores.append(label[0])
    emb_mat = np.matrix(embs)
    pca = run_pca(emb_mat)
    pcs = pca.transform(emb_mat)
    pc1s, pc2s = emb_mat[:,0], emb_mat[:,1]
    plot_pca(pc1s, pc2s, scores, task, country)


def pca_on_relevant_articles():
    train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
    article_embeddings_dir = os.path.join(os.curdir, 'raw', 'wikipedia', 'doc2vec_embeddings')
    cluster_article_rank_dist_path = os.path.join(os.curdir, 'processed', 'ClusterNearestArticles_Rank_Dist.csv')

    NUM_ARTICLES = 5
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    for task in ['MatEd', 'IMR']:
        for country in countries:
            print(country, task)
            dataset = util_data.DHS_Wiki_Dataset(DHS_csv_file=train_path,
                        emb_root_dir=article_embeddings_dir, cluster_rank_csv_path=cluster_article_rank_dist_path,
                        emb_dim=300, n_articles=NUM_ARTICLES, include_dists=False,
                        country_subset=[country], task=task,
                        transforms=None)
            do_country_level_analysis_internal(dataset, task, country)

def main():
    # pca_on_all_articles()
    pca_on_relevant_articles()

if __name__ == '__main__':
    main()
