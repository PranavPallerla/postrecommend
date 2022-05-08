
def recommended_posts(query_index,rating_popular_post_pivot, model_knn_pkl):
    distances, indices = model_knn_pkl.kneighbors(rating_popular_post_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    return distances,indices
