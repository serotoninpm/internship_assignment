import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE):
    train = read('./result/train_val_loss_log/train_loss-{0}-{1}-{2}.txt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE))
    test = read('./result/train_val_loss_log/valid_loss-{0}-{1}-{2}.txt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE))
    plt.clf()
    plt.plot(train, 'r', label='train')
    plt.plot(test, 'b', label='validation')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel("loss")
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig("./result/loss_graph/" + "loss_graph-{0}-{1}-{2}.png".format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE))


if __name__ == '__main__':
    draw()
