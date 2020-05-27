import numpy as np
import sys
import csv
import argparse
import matplotlib.pyplot as plt

def train(x_train, y_train, epoch, lr, size, model):
    best = 1000000000000
    w = np.zeros([size, 1])
    w_best = np.zeros([size, 1])
    # square of previous gradients
    w_pre_grad = 0
    for i in range(epoch):
        # x_transpose size = (163, 5751)
        x_train_t = x_train.transpose()
        # calculate gradient
        # y size = (5751, 1), w size = (163, 1), w_grad size = (163, 1)
        w_grad = -2 * x_train_t.dot(y_train - x_train.dot(w))
        # calculate sigma_t
        w_pre_grad += w_grad.transpose().dot(w_grad)
        w_ada = w_pre_grad ** 0.5
        # gradient descent
        w -= w_grad * lr / w_ada
        # predict
        # x size = (5751, 163)
        ans = x_train.dot(w)
        # Root Mean Square Error
        RMSE = ((y_train - ans).transpose().dot(y_train - ans) / y_train.shape[0]) ** 0.5
        RMSE = np.squeeze(RMSE)
        if RMSE < best:
            best = RMSE
            w_best = w.copy()
        if i % 1000 == 0:
            print("Training: epochs = {}, RMSE = {:.4f}".format(i, best))
    np.save(model, w_best)


def eval(x_test, y_test, model):
    x = list(range(600))
    # load model
    w = np.load(model)
    # predict
    ans = x_test.dot(w)
    # Root Mean Square Error
    RMSE = ((y_test - ans).transpose().dot(y_test - ans) / y_test.shape[0]) ** 0.5
    RMSE = np.squeeze(RMSE) 
    plt.plot(x, ans, color="blue", label="answer")
    plt.plot(x, y_test, color="red", label="predict")
    # plt.text(0, 0, f"Root mean square error {RMSE}")
    plt.xlabel("")
    plt.ylabel("pm2.5")
    plt.legend(loc="best")
    plt.savefig(f"mix.png")
    plt.cla()
    plt.clf()
    plt.plot(x, ans, color="blue", label="answer")
    # plt.text(0, 0, f"Root mean square error {RMSE}")
    plt.xlabel("")
    plt.ylabel("pm2.5")
    plt.legend(loc="best")
    plt.savefig(f"answer.png")
    plt.cla()
    plt.clf()
    plt.plot(x, y_test, color="red", label="predict")
    # plt.text(0, 0, f"Root mean square error {RMSE}")
    plt.xlabel("")
    plt.ylabel("pm2.5")
    plt.legend(loc="best")
    plt.savefig(f"predict.png")
    plt.cla()
    plt.clf()
    print("Testing: RMSE = {:.4f}".format(RMSE))

def data_preparation(doc):
    # read data
    x = []
    with open(doc, newline = '', encoding = "big5") as csvfile:
        rows = csv.reader(csvfile)
        i = 0
        for row in rows:
            for j in range(len(row)):
                if row[j] == 'NR':
                    row[j] = 0
            if i == 0:
                i += 1
                continue
            if int((i-1) / 18) == 0:
                x.append(row[3:])
            else:
                for k in row[3:]:
                    x[(i-1) % 18].append(k)
            i += 1
    x_train_pre = np.array(x, dtype = 'float32')

    # extract features
    x_train = []
    y_train = []
    for data_num in range(len(x_train_pre[0]) - 9):
        c = x_train_pre[:, data_num:data_num + 9].copy()
        d = x_train_pre[9, data_num + 9]

        c_re = np.reshape(c, (c.shape[0] * c.shape[1]))
        c_re = list(c_re)
        # add bias at first dimension
        c_re.insert(0, 1)
        c_re = np.array(c_re)

        x_train.append(c_re)
        y_train.append(d)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    # split train & test
    x_train, y_train, x_test, y_test = x_train[:-600], y_train[:-600], x_train[-600:], y_train[-600:]
    return x_train, y_train, x_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--lr", default=1e-1, type=float,
                        help="The initial learning rate for Adagrad.")
    parser.add_argument("--num_train_epochs", default=100000, type=int,
                        help="Total number of training epochs to perform.")
    args = parser.parse_args()

    doc = "../data/train.csv"
    model = "model.npy"

    # prepare data
    x_train, y_train, x_test, y_test = data_preparation(doc)
    size = x_train.shape[1]

    # training
    if args.do_train:
        train(x_train, y_train, args.num_train_epochs, args.lr, size, model)
    if args.do_eval:
        eval(x_test, y_test, model)

if __name__ == "__main__":
    main()