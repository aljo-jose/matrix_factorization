# Matrix factorization algorithm.
import numpy as np
import random


class MF:
    def __init__(self, n_users, n_movies, vec_size):
        self.n_users, self.n_movies = n_users, n_movies
        self.learning_rate = 0.01
        self.lambda_ = 0.01
        self.vec_size = vec_size

        self.user_vec = np.random.normal(scale=1 / vec_size, size=(self.n_users, self.vec_size))
        self.movies_vec = np.random.normal(scale=1 / vec_size, size=(self.n_movies, self.vec_size))

    def __call__(self, data, run_type):
        batch_loss = 0
        predictions = []
        for x in data:
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

        return {'mean_loss': batch_loss/len(data), 'predictions': predictions}

    def predict_all_ratings(self):
        estimated_R = np.dot(self.user_vec, self.movies_vec.T)
        return estimated_R


def train(R):
    # init model.
    n_users, n_movies = R.shape
    vec_size = 10
    model = MF(n_users=n_users, n_movies=n_movies, vec_size=vec_size)

    # make training samples - choose only observations with rating.
    train_data = []
    for i in range(n_users):
        for j in range(n_movies):
            if R[i][j] != 0:
                train_data.append({'user_index': i, 'movie_index': j, 'rating': R[i, j]})

    n_iterations = 100
    training_results = []
    for i in range(n_iterations):
        random.shuffle(train_data)
        loss = model(data=train_data, run_type='train')
        training_results.append(loss)
        print(f' iteration - {i}, loss : {loss}')
    return model

def get_data():
    R = np.asarray([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ])
    # R = np.asarray([
    #     [3,1,1,3,1],
    #     [1,2,4,1,3],
    #     [3,1,1,3,1],
    #     [4,3,5,4,4]
    #     ])
    return R


if __name__ == '__main__':
    data = get_data()

    # do training and get the model.
    mf_model = train(data)

    # predictions.
    rating_estimates = mf_model.predict_all_ratings()
    print('predicted ratings : \n', rating_estimates)
    print('actual ratings : \n', data)
