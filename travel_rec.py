from __future__ import division
from __future__ import print_function

import time
import sys
from pylab import *
import numpy as np
import tensorflow as tf

def factorize(ratings_train, ratings_val, user_bias, item_bias, rank, max_iter, threshold=0.5, lda=0.1, lr=0.01,
              result_processor=None):
    # Extract infos from the training data
    rating_values = np.array(ratings_train[:, 2], dtype=float32)
    user_indices = ratings_train[:, 0]
    item_indices = ratings_train[:, 1]
    num_ratings = len(item_indices)
    num_users = user_bias.shape[0]
    num_items = item_bias.shape[0]
    # Ratings mean.
    mean_rating = mean(ratings_train[:, 2])

    # Extract infos the validation data
    rating_values_val = np.array(ratings_val[:, 2], dtype=float32)
    user_indices_val = ratings_val[:, 0]
    item_indices_val = ratings_val[:, 1]
    num_ratings_val = len(item_indices_val)

    # Shaping bias vectors
    user_bias = user_bias.reshape((num_users, 1))
    item_bias = item_bias.reshape((1, num_items))

    # Initialize the matrix factors from random normals with mean 0. W will
    # represent users and H will represent items.
    # W_plus_bias = 
    # | user_id/feature | feature1 | feature2 | bias | ones
    # |       1         |     6    |    3     | 0.27 |  1
    # |       2         |     2    |    0     | 1.67 |  1
    #
    # H_plus_bias = 
    # | feature/item1   |     3    |    1  
    # |    feature1     |     6    |    3
    # |    feature2     |     2    |    0
    # |      bias       |    0.23  |   2.3
    # |      ones       |     1    |    1
    
    W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.2, mean=0, seed=123), name="users")
    H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.2, mean=0, seed=123), name="items")

    # Add bias vectors to factor matrices
    W_plus_bias = tf.concat(1, [W, tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"),
                                tf.ones((num_users, 1), dtype=float32, name="item_bias_ones")])
    H_plus_bias = tf.concat(0, [H, tf.ones((1, num_items), name="user_bias_ones", dtype=float32),
                                tf.convert_to_tensor(item_bias, dtype=float32, name="item_bias")])
    # Multiply the factors to get our result as a dense matrix
    result = tf.matmul(W_plus_bias, H_plus_bias)

    # result_values = result[user_indices, item_indices]
    result_values = tf.gather(tf.reshape(result, [-1]), user_indices * tf.shape(result)[1] + item_indices,
                              name="extract_training_ratings")

    result_values_val = tf.gather(tf.reshape(result, [-1]), user_indices_val * tf.shape(result)[1] + item_indices_val,
                                  name="extract_validation_ratings")

    # Calculate the difference between the predicted ratings and the actual
    # ratings. The predicted ratings are the values obtained form the matrix
    # multiplication with the mean rating added on.
    diff_op = tf.sub(tf.add(result_values, mean_rating, name="add_mean"), rating_values, name="raw_training_error")
    diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_val,
                         name="raw_validation_error")

    with tf.name_scope("training_cost") as scope:
        base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")
        # Add regularization.
        regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
        cost = tf.div(tf.add(base_cost, regularizer), num_ratings * 2, name="average_error")

    with tf.name_scope("validation_cost") as scope:
        cost_val = tf.div(
            tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"),
            num_ratings_val * 2, name="average_error")

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)

    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cost, global_step=global_step)

    with tf.name_scope("training_accuracy") as scope:
        # To compute accuracy we just measure the absolute difference against the threshold
        good = tf.less(tf.abs(diff_op), threshold)

        accuracy_tr = tf.div(tf.reduce_sum(tf.cast(good, tf.float32)), num_ratings)
        accuracy_tr_summary = tf.scalar_summary("accuracy_tr", accuracy_tr)

    with tf.name_scope("validation_accuracy") as scope:
        # Validation set accuracy:
        good_val = tf.less(tf.abs(diff_op_val), threshold)
        accuracy_val = tf.reduce_sum(tf.cast(good_val, tf.float32)) / num_ratings_val
        accuracy_val_summary = tf.scalar_summary("accuracy_val", accuracy_val)

    # Create a TensorFlow session and initialize variables
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Make sure summaries get written to the logs so we can debug it with TensorBoard
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph)

    # Let the game start
    for i in range(max_iter):
        if i % 500 == 0:
            res = sess.run([summary_op, accuracy_tr, accuracy_val, cost, cost_val])
            summary_str = res[0]
            acc_tr = res[1]
            acc_val = res[2]
            cost_ev = res[3]
            cost_val_ev = res[4]
            writer.add_summary(summary_str, i)
            print("Training accuracy at step %s: %s" % (i, acc_tr))
            print("Validation accuracy at step %s: %s" % (i, acc_val))
            print("Training cost: %s" % (cost_ev))
            print("Validation cost: %s" % (cost_val_ev))
        else:
            sess.run(train_step)

    with tf.name_scope("final_model") as scope:
        # At the end we want to get the final ratings matrix
        add_mean_final = tf.add(result, mean_rating, name="add_mean_final")
        if result_processor == None:
            final_matrix = add_mean_final
        else:
            final_matrix = result_processor(add_mean_final)
        final_res = sess.run([final_matrix])

    finalAcc = accuracy_val.eval(session=sess)
    sess.close()
    return final_res[0], finalAcc

# Now we're going to test the recommender on a dataset
f = open('u.data', 'r')
data = np.array([ map(int,line.split("\t")) for line in f ])
ratings = data[:, 2]
user_size = np.unique(data[:, 0]).shape[0]
item_size = np.unique(data[:, 1]).shape[0]
user_indices = data[:, 0]
item_indices = data[:, 1]
# using average bias
# user_bias = np.array([ np.mean(ratings[user_indices == user_id]) for user_id in range(1, user_size + 1) ])
# item_bias = np.array([ np.mean(ratings[item_indices == item_id]) for item_id in range(1, item_size + 1) ])
# using random bias
user_bias = np.random.uniform(-0.5, 0.5, size=user_size)
item_bias = np.random.uniform(-0.5, 0.5, size=item_size)


res = factorize(data[:990, :], data[990:1000, :], user_bias, item_bias, 2, 500)
while True:
    print("Enter user id")
    user_id = input()
    print("Enter item id")
    item_id = input()
    print(res[0][int(user_id - 1), int(item_id - 1)])
