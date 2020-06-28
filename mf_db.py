"""
Apply Matrix Factorization to movie lens data set.
"""
import numpy as np
import pandas as pd
import random

DATA_PATH = '/Users/aljo/research/matrix_factorization/data/ml-latest-small'

N_EPCOH = 10
BATCH_SIZE = 64
VEC_SIZE = 4


# MF.
class MF:
    def __init__(self, n_users, n_movies, vec_size):
        self.n_users, self.n_movies = n_users, n_movies
        self.learning_rate = 0.01
        self.lambda_ = 0.01
        self.vec_size = vec_size

        self.user_vec = np.random.normal(scale=1 / vec_size, size=(self.n_users, self.vec_size))
        self.movies_vec = np.random.normal(scale=1 / vec_size, size=(self.n_movies, self.vec_size))

    def __call__(self, batch_data, run_type):
        batch_loss = 0
        predictions = []
        for x in batch_data:
            y = x['rating']
            user_index = x['user_index']
            movie_index = x['movie_index']

            # compute loss
            y_pred = np.dot(self.user_vec[user_index].T, self.movies_vec[movie_index])

            if run_type == 'train':
                # compute loss.
                loss = (y_pred - y) ** 2

                # compute gradient.
                user_grad_vec = (y_pred - y) * self.movies_vec[movie_index]
                movie_grad_vec = (y_pred - y) * self.user_vec[user_index]

                # update gradient.
                self.user_vec[user_index] -= self.learning_rate * user_grad_vec
                self.movies_vec[movie_index] -= self.learning_rate * movie_grad_vec
                batch_loss += loss
            elif run_type == 'predict':
                predictions.append(y_pred)
            else:
                raise ValueError('oops! incorrect run type. ', run_type)

        return {'mean_loss': batch_loss / len(data), 'predictions': predictions}

    def predict(self):
        estimated_R = np.dot(self.user_vec, self.movies_vec.T)
        return estimated_R


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# train.
def train(data):
    ratings_data = data['ratings']
    n_batches = int(len(ratings_data) / BATCH_SIZE)

    model = MF(n_users=len(data['user_dict']), n_movies=len(data['movie_dict']), vec_size=VEC_SIZE)
    for epoch in range(N_EPCOH):
        random.shuffle(ratings_data)
        batch_no = 0
        for batch_data in chunks(ratings_data, BATCH_SIZE):
            batch_no += 1
            out = model(batch_data, run_type='train')
            loss = out['mean_loss']
            print(f'epoch {epoch}, batch {batch_no}/{n_batches}, loss {round(loss, 4)}')
        print(f'--------- END OF EPOCH {epoch} ------------')
    return model


# load data from csv.
def get_data():
    ratings_df = pd.read_csv(DATA_PATH + '/ratings.csv')

    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()

    user_dict = {uid: i for i, uid in enumerate(unique_users)}
    movie_dict = {mid: i for i, mid in enumerate(unique_movies)}
    processed_data = []
    for index, row in ratings_df.iterrows():
        processed_data.append({'user_index': user_dict[row['userId']], 'movie_index': movie_dict[row['movieId']],
                               'rating': row['rating']})

    return {'ratings': processed_data, 'user_dict': user_dict, 'movie_dict': movie_dict}


# main
if __name__ == '__main__':
    print('started.')

    data = get_data()
    print('data loaded and processed.')

    train(data=data)
