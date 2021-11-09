# -- coding: utf-8 --
import pandas
import pandas as pd
import seaborn as sns
sns.set(context="notebook",style="whitegrid",palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# get features
def get_X(df):
    '''
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    '''
    ones = pd.DataFrame({'ones':np.ones(len(df))})
    data = pd.concat([ones,df],axis=1)
    print(data.iloc[:, :-1].values[:5])
    return data.iloc[:, :-1].values

def get_y(df):
    # assume the last column is the target
    return np.array(df.iloc[:,-1])

def normalize_frature(df):
    return df.apply(lambda column:(column - column.mean()) / column.std())

def linear_regression(X_data,y_data,alpha,epoch,optimzier=tf.train.GradientDescentOptimizer):
    # placeholder for graph input
    X = tf.placeholder(tf.float32,shape=X_data.shape)
    y = tf.placeholder(tf.float32,shape=y_data.shape)

    # construct the graph
    with tf.variable_scope('linear-regression'):
        W = tf.get_variable("weights",(X_data.shape[1],1),initializer=tf.constant_initializer)
        y_pred = tf.matmul(X,W) # m*n @ n*1 -> m*1
        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
        opt = optimzier(learning_rate=alpha)
        opt_operation = opt.minimize(loss)

        # run the session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_data = []

            for i in range(epoch):
                _, loss_val, W_val = sess.run([opt_operation,loss,W],feed_dict={X:X_data,y:y_data})

                loss_data.append(loss_val[0,0])

                if len(loss_data) > 1 and np.abs(loss_data[-1] - loss_data[-2]) < 10 ** -9:
                    print("converged at epoch {}".format(i))
                    break

        # clear the graph
        tf.reset_default_graph()
        return {'loss':loss_data,'parameters':W_val}

def lr_cost(theta,X,y):
    '''
    :param theta:  R(n), 线性回归的参数
    :param X: R(m*n), m 样本数, n 特征数
    :param y:R(m)
    :return:
    '''
    m = X.shape[0] # 样本数

    # R(m*1)，X @ theta等价于X.dot(theta)
    inner = X @ theta - y

    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost


def gradient(theta,X,y):
    m = X.shape[0]

    # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)
    inner = X.T @ (X @ theta - y)

    return inner / m

def batch_gradient_decent(theta,X,y,epoch,alpha=0.01):
    '''
    拟合线性回归，返回参数和cost
    :param theta:
    :param X:
    :param y:
    :param epoch: 批处理的轮数
    :param alpha:
    :return:
    '''

    cost_data = [lr_cost(theta,X,y)]
    _theta = theta.copy()

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta,X,y)
        cost_data.append(lr_cost(_theta,X,y))

    return _theta,cost_data


if __name__ == '__main__':

    # load data
    df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
    print(df.head())
    print(df.info())

    # see the rawdata
    sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
    plt.show()

    X = get_X(df)
    print(X.shape,type(X))

    y = get_y(df)
    print(y.shape,type(y))

    theta = np.zeros(X.shape[1])

    print(lr_cost(theta,X,y))

    epoch = 500
    final_theta,cost_data = batch_gradient_decent(theta,X,y,epoch)

    print(final_theta)
    print(cost_data)

    # final cost
    print(lr_cost(final_theta,X,y))

    # visualize cost data
    ax = sns.tsplot(cost_data,time=np.arange(epoch + 1))
    ax.set_xlabel('epoch')
    ax.set_ylabel('cost')
    plt.show()

    b = final_theta[0]
    m = final_theta[1]

    plt.scatter(df.population, df.profit,label="Training data")
    plt.plot(df.population,df.population * m + b, label="Prediction")
    plt.legend(loc = 2)
    plt.show()

