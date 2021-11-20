import numpy as np
import pandas as pd
import recommender_functions as rec_f
import sys # can use sys to take command line arguments

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''

    def fit(self, train_data_path, movies_path, latent_features=12, learning_rate=0.005, iters=20):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization
        
        INPUT:
        - train_data_path: path for the training data (csv file)
        - movies_path: path of movies data (csv file)
        - latent_features: number of latent features to use in prediciton, set to 12 per default
        - learning_rate: learning rate at each step of gradient descent, set to 0.005 per default
        - iters: number of iterations of gradient descent, set to 20 per default
        
        OUTPUT:
        Function has no output. It stores the following variables that can be looked at:
        - train_df: train dataframe used in fitting FunkSVD
        - movies_df: movies dataframe showing description for each movie
        - train_user_item: numpy array of user by movie showing associated ratings
        - n_users: number of users in the training data
        - n_movies: number of movies in the training data
        - num_rating: number of valid ratings (not nan)
        - user_mat: U matrix from SVD
        - movie_mat: V matrix from SVD
        '''
        # initiate variables
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters
        
        # Set up useful values to be used through the rest of the function
        self.reviews_df = pd.read_csv(train_data_path)
        self.movies_df = pd.read_csv(movies_path)
        
        # Create user-by-item matrix
        self.train_user_item_df = self.reviews_df[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.train_user_item_df = self.train_user_item_df.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.train_user_item = np.array(self.train_user_item_df)  
        self.user_by_index = np.array(self.train_user_item_df.index)
        self.movie_by_index = np.array(self.train_user_item_df.columns)        
       
        # get the number of users, movies and valid ratings (not nan)     
        self.n_users = self.train_user_item.shape[0]
        self.n_movies = self.train_user_item.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.train_user_item))
        
        # initialize the user and movie matrices with random values
        self.user_mat = np.random.rand(self.n_users, self.latent_features)
        self.movie_mat = np.random.rand(self.latent_features, self.n_movies)
        
        # initialize sse at 0 for first iteration
        sse_accum = 0
        
        # header for running results
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")
        
        # for each iteration
        for iteration in range(self.iters):
            # update our sse
            sse_accum = 0
            
            # For each user-movie pair
            for user in range(self.n_users):
                for movie in range(self.n_movies):                
                    # if the rating exists
                    rating = self.train_user_item[user][movie]
                    if not pd.isna(rating):
                        # print('user {}, movie {}, rating {}'.format(user, movie, rating))
                        
                        # compute the error as the actual minus the dot product of the user and movie latent features
                        error = rating - np.dot(self.user_mat[user], self.movie_mat[:, movie])
    
                        # Keep track of the total sum of squared errors for the matrix
                        sse_accum =+ error
                        
                        # update the values in each matrix in the direction of the gradient
                        update_user_mat = self.user_mat[user] + learning_rate * 2 * error * self.movie_mat[:, movie]
                        self.user_mat[user] = update_user_mat
                        update_movie_mat = self.movie_mat[:, movie] + learning_rate * 2 * error * self.user_mat[user]
                        self.movie_mat[:, movie] = update_movie_mat
    
            # print results for iteration
            try: 
                print('Step: {}, error is: {}.'.format(iteration, error))
            except:
                print('Rating was nan.')
                
        # Knowledge based fit
        self.ranked_movies = rec_f.create_ranked_df(self.movies_df, self.reviews_df)

    def predict_rating(self, user_id, movie_id):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''
        # initiate variables
        self.user_id = user_id
        self.movie_id = movie_id
        
        # User row and Movie Column
        row = np.where(self.user_by_index == user_id)[0][0]      
        column = np.where(self.movie_by_index == movie_id)[0][0]
        
        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(self.user_mat[row],  self.movie_mat[:, column])
       
        return pred

    def make_recs(self, _id, _id_type='movie',  rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)
        
        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies by id                  
        rec_names - (array) a list or numpy array of recommended movies by name
        '''
        
        # initiate variables
        self._id = _id
        self._id_type = _id_type
        self.rec_num = rec_num
        
        rec_ids = list()
        rec_names = list()
        
        if _id_type == 'movie':
            rec_ids.extend(rec_f.find_similar_movies(self._id, self.movies_df))
            rec_ids = rec_ids[:rec_num]
        elif _id_type == 'user':
            # get row and column for funk svd
            idx = np.where(self.user_by_index == _id)[0][0]
            # get prediction as mutiplication of idx row in user_mat and movie_mat -> get prediciton for every movie 
            predictions = np.dot(self.user_mat[idx, :], self.movie_mat)
            # sort it and get associated indices
            predictions = np.argsort(predictions)
            # get last x (as it's sorted ascending)
            rec_ids = self.movie_by_index[-rec_num:].tolist()
            
            # if we dont have enough ratings, try collaborative filtering on top:
            if len(rec_ids) < rec_num:
                rec_ids.extend(rec_f.popular_recommendations(_id, rec_num, self.ranked_movies))
                rec_ids = rec_ids[: rec_num]
        
        rec_names.extend(rec_f.get_movie_names(rec_ids, self.movies_df))
        
        return rec_ids, rec_names


#if __name__ == '__main__':
    # test different parts to make sure it works
