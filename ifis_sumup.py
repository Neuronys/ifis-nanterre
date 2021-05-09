#import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import connected_components
import numpy as np
import pysbd
import streamlit as st
#from string import punctuation
#from spacy.lang.fr.stop_words import STOP_WORDS
#from string import punctuation



# LEXRANK code
def degree_centrality_scores(similarity_matrix, threshold=None, increase_power=True):
    if not (
            threshold is None
            or isinstance(threshold, float)
            and 0 <= threshold < 1
    ):
        raise ValueError(
            '\'threshold\' should be a floating-point number '
            'from the interval [0, 1) or None',
        )

    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)

    else:
        markov_matrix = create_markov_matrix_discrete(
            similarity_matrix,
            threshold,
        )

    scores = stationary_distribution(
        markov_matrix,
        increase_power=increase_power,
        normalized=False,
    )

    return scores


def _power_method(transition_matrix, increase_power=True):
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        return eigenvector

    transition = transition_matrix.transpose()

    while True:
        eigenvector_next = np.dot(transition, eigenvector)

        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next

        eigenvector = eigenvector_next

        if increase_power:
            transition = np.dot(transition, transition)


def connected_nodes(matrix):
    _, labels = connected_components(matrix)

    groups = []

    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)

    return groups


def create_markov_matrix(weights_matrix):
    n_1, n_2 = weights_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'weights_matrix\' should be square')

    row_sum = weights_matrix.sum(axis=1, keepdims=True)

    return weights_matrix / row_sum


def create_markov_matrix_discrete(weights_matrix, threshold):
    discrete_weights_matrix = np.zeros(weights_matrix.shape)
    ixs = np.where(weights_matrix >= threshold)
    discrete_weights_matrix[ixs] = 1

    return create_markov_matrix(discrete_weights_matrix)


def graph_nodes_clusters(transition_matrix, increase_power=True):
    clusters = connected_nodes(transition_matrix)
    clusters.sort(key=len, reverse=True)

    centroid_scores = []

    for group in clusters:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        centroid_scores.append(eigenvector / len(group))

    return clusters, centroid_scores


def stationary_distribution(transition_matrix, increase_power=True, normalized=True):
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'transition_matrix\' should be square')

    distribution = np.zeros(n_1)

    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector

    if normalized:
        distribution /= n_1

    return distribution



#@st.cache(suppress_st_warning=True)
def hl1(raw_text, ratio, highlighter = ['<b>','</b>']):
    #segment text to sentences
    seg = pysbd.Segmenter(language="fr", clean=True)
    sentences = seg.segment(raw_text)
    nb_sentences = len(sentences)
    ref_text = ''.join(sentences)
    mess = "Finding " + str(nb_sentences) + " sentences"
    st.code(mess)

    #compute sentence embedding
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1', device="cpu")
    embeddings = embedder.encode(sentences, convert_to_tensor=True)

    #Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()

    #Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

    #We argsort so that the first element is the sentence with the highest score
    most_central_sentence_indices = np.argsort(-centrality_scores)

    lc=len(most_central_sentence_indices)
    rc = round(lc*ratio)
    hl = []
    mess = "indices:"
    for i in most_central_sentence_indices[:rc]:
        #mess =  mess + str(most_central_sentence_indices[i]) + ","
        hl.append(most_central_sentence_indices[i])
    #st.code(mess)

    highlighted = []
    for i in range(nb_sentences):
        if i in hl:
            #st.code(str(i))
            highlighted.append(highlighter[0]+sentences[i]+highlighter[1]+' ')
        else:
            highlighted.append(sentences[i]+' ')
    html = '\n'.join(highlighted)

    ret = {}    
    ret["raw_text"] = raw_text
    ret["highlighted_html"] = html
    return (ret)


#@st.cache(suppress_st_warning=True)
def hl2(raw_text, ratio, nb_clusters = 0, highlighter = ['<b>','</b>']):
    #segment text to sentences
    seg = pysbd.Segmenter(language="fr", clean=True)
    sentences = seg.segment(raw_text)
    nb_sentences = len(sentences)
    ref_text = ''.join(sentences)
    mess = "Finding " + str(nb_sentences) + " sentences"
    st.code(mess)

    #compute sentence embedding
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1', device="cpu")
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    # Normalize the embeddings to unit length
    body_embeddings = embeddings /  np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    if (nb_clusters > 0):
        k = nb_clusters
    else:
        k = round(nb_sentences*(ratio/3))+1
    mess = "Finding " + str(k) + " clusters"
    st.code(mess)

    clustering_model = KMeans(n_clusters=k,random_state=0)
    clustering_model.fit(body_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(k)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(sentences[sentence_id])

    clustered_sentence_ids = [[] for i in range(k)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentence_ids[cluster_id].append(sentence_id)

    centroids = clustering_model.cluster_centers_
    ordered_ids = []
    for j, centroid in enumerate(centroids):
        centroid_min = 1e10
        temp_values = []
        temp_ids = []

        for i in range(len(clustered_sentence_ids[j])):
            value = np.linalg.norm(body_embeddings[clustered_sentence_ids[j][i]] - centroid)
            if centroid_min == 1e10:
                temp_values.append(value)
                temp_ids.append(clustered_sentence_ids[j][i])
                centroid_min = value
            else:
                if value < centroid_min:
                    temp_values.insert(0,value)
                    temp_ids.insert(0,clustered_sentence_ids[j][i])
                    centroid_min = value
                else:
                    for k in range(len(temp_ids)):
                        if value < temp_values[k]:
                            temp_values.insert(k,value)
                            temp_ids.insert(k,clustered_sentence_ids[j][i])
                            break
                        
                        if k == len(temp_ids) - 1:
                            temp_values.append(value)
                            temp_ids.append(clustered_sentence_ids[j][i])

        ordered_ids.append(temp_ids)
    ordered_ids.sort(key=len, reverse=True)

    args = []
    for j in range(len(ordered_ids)):
        l = round(len(ordered_ids[j])*ratio)
        if l > 0:
            for i in range(l):
                args.append(ordered_ids[j][i])
    sorted_values = sorted(args)

    highlighted = []
    for i in range(nb_sentences):
        if i in sorted_values:
            #st.code(str(i))
            highlighted.append(highlighter[0]+sentences[i]+highlighter[1]+' ')
        else:
            highlighted.append(sentences[i]+' ')
    html = '\n'.join(highlighted)

    ret = {}    
    ret["raw_text"] = raw_text
    ret["highlighted_html"] = html
    return (ret)

