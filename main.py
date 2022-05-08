import pandas as pd
import pickle
import numpy as np


from flask import Flask, request, jsonify
from flask import render_template

from recommendation_engine import recommended_posts #Importing engine function

app = Flask(__name__)

# #Avoid switching the order of 'title' and 'confidence' keys
# app.config['JSON_SORT_KEYS'] = False

users_data_df = pd.read_csv(r'user_data.csv')
post_data_df = pd.read_csv(r'post_data.csv')
view_data_df = pd.read_csv(r'view_data.csv')

dataframe = pd.DataFrame(view_data_df)
dataframe["Valuable"] = np.random.randint(1, 6, len(dataframe))
df = pd.merge(dataframe, post_data_df ,on='post_id')
data = df.drop(['time_stamp', 'category'], axis=1)
combine_post_rating = data.dropna(axis = 0, subset = ['title'])

post_ratingCount = (combine_post_rating.
     groupby(by = ['title'])['Valuable'].
     count().
     reset_index().
     rename(columns = {'Valuable': 'totalValuableCount'})
     [['title', 'totalValuableCount']]
    )
rating_with_totalValuableCount = combine_post_rating.merge(post_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
popularity_threshold = 13
rating_popular_post = rating_with_totalValuableCount.query('totalValuableCount >= @popularity_threshold')
rating_popular_post = rating_popular_post.drop_duplicates(['user_id', 'title'])
rating_popular_post_pivot = rating_popular_post.pivot(index = 'title', columns = 'user_id', values = 'Valuable').fillna(0)




model_knn_pkl = pickle.load(open('model_knn.pickle', 'rb'))


#API endpoint
@app.route('/')
def index():

    return render_template('underconstruction.html')

#API endpoint
@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      query_index = int(result["query_index"])
      #Call recommendation engine
      (distances,indices) = recommended_posts(query_index, rating_popular_post_pivot, model_knn_pkl)
      l = []
      for i in range(0, len(distances.flatten())):
        if i == 0:
            l.append('Recommendations for {0}:\n'.format(rating_popular_post_pivot.index[query_index]))
        else:
            l.append('{0}: {1}, with distance of {2}:'.format(i, rating_popular_post_pivot.index[indices.flatten()[i]], distances.flatten()[i]))

      return render_template('result.html', l=l)




if __name__ == '__main__':

    app.run()