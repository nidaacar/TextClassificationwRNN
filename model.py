from rnn import RNN
import matplotlib.pyplot as plt
import numpy as np
import random


class Model(RNN):
    def __init__(self, train_data, test_data):
        # Create the vocabulary.
        self.train_data = train_data
        self.test_data = test_data

        vocab = list(set([w for text in train_data.keys()
                          for w in text.split(' ')]))
        vocab2 = list(set([w for text in test_data.keys()
                           for w in text.split(' ')]))
        vocab = vocab+vocab2

        self.vocab_size = len(vocab)

        # Initialize the RNN
        super(Model, self).__init__(self.vocab_size, 2)
        print('%d unique words found' % self.vocab_size)

        # Assign indices to each word.
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}
        print(self.word_to_idx['good'])
        print(self.idx_to_word[0])

    def createInputs(self, text):
        '''
        Returns an array of one-hot vectors representing the words in the input text string.
        - text is a string
        - Each one-hot vector has shape (vocab_size, 1)
        '''

        inputs = []
        for w in text.split(' '):
            v = np.zeros((self.vocab_size, 1))
            if w in self.word_to_idx:
                v[self.word_to_idx[w]] = 1
                inputs.append(v)
        return inputs

    def softmax(self, xs):
        # Applies the Softmax Function to the input array.
        return np.exp(xs) / sum(np.exp(xs))

    def processData(self, data, backprop=True):
        '''
        This function will train the RNN on a given dataset.
        - data is a dictionary mapping text to True or False.
        - backprop determines if the backward phase should be run.
        '''
        items = list(data.items())
        random.shuffle(items)

        loss = 0
        num_correct = 0

        y_true = []
        y_pred = []

        for x, y in items:
            inputs = self.createInputs(x)
            target = int(y)

            # Forward
            out, _ = self.forward(inputs)
            probs = self.softmax(out)

            # Calculate loss / accuracy
            loss -= np.log(probs[target])
            num_correct += int(np.argmax(probs) == target)

            y_true.append(target)
            y_pred.append(np.argmax(probs))

            if backprop:
                # Build dL/dy
                d_L_d_y = probs
                d_L_d_y[target] -= 1

                # Backward
                self.backprop(d_L_d_y)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        f1_score = self.compute_f1_score(y_true, y_pred)

        return loss / len(data), num_correct / len(data), f1_score

    def train(self, num_epoch=1000, plot_results=True):
        print("*")
        train_info = []
        for epoch in range(num_epoch):
            train_loss, train_acc, train_f1 = self.processData(self.train_data)
            test_loss, test_acc, test_f1 = self.processData(
                self.test_data, backprop=False)

            if epoch % 100 == 99:
                print('--- Epoch %d' % (epoch + 1))
                print('Train:\tLoss: %.3f | Accuracy: %.3f | F1 Score: %.3f' %
                      (train_loss, train_acc, train_f1))
                print('Test:\tLoss: %.3f | Accuracy: %.3f | F1 Score: %.3f' %
                      (test_loss, test_acc, test_f1))

                train_info.append(
                    [train_loss, train_acc, test_loss, test_acc, epoch, train_f1, test_f1])

        if plot_results:
            epoches = []
            train_loss = []
            test_loss = []
            train_acc = []
            test_acc = []
            train_f1 = []
            test_f1 = []

            for i in train_info:
                train_loss.append(i[0])
                train_acc.append(i[1])
                test_loss.append(i[2])
                test_acc.append(i[3])
                epoches.append(i[4])
                train_f1.append(i[5])
                test_f1.append(i[6])

            # Plot 1: Train Loss vs Epoches
            plt.figure(1)
            plt.plot(epoches, train_loss)
            plt.title("Train Loss vs Epoches")
            plt.ylabel('Train Loss')
            plt.xlabel('Epoches')
            plt.savefig('Plots/Train Loss vs Epoches')

            # Plot 2: Train Accuracy vs Epoches
            plt.figure(2)
            plt.plot(epoches, train_acc)
            plt.title("Train Accuracy vs Epoches")
            plt.ylabel('Train Accuracy')
            plt.xlabel('Epoches')
            plt.savefig('Plots/Train Accuracy vs Epoches')

            # Plot 3: Test Loss vs Epoches
            plt.figure(3)
            plt.plot(epoches, test_loss)
            plt.title("Test Loss vs Epoches")
            plt.ylabel('Test Loss')
            plt.xlabel('Epoches')
            plt.savefig('Plots/Test Loss vs Epoches')

            # Plot 4: Test Accuracy vs Epoches
            plt.figure(4)
            plt.plot(epoches, test_acc)
            plt.title("Test Accuracy vs Epoches")
            plt.ylabel('Test Accuracy')
            plt.xlabel('Epoches')
            plt.savefig('Plots/Test Accuracy vs Epoches')

            # Plot 5: F1 score (Train) vs Epoches
            plt.figure(5)
            plt.plot(epoches, train_f1)
            plt.title("F1 score (Train) vs Epoches")
            plt.ylabel('F1 Score')
            plt.xlabel('Epoches')
            plt.savefig('Plots/F1 score (Train) vs Epoches')

            # Plot 6: F1 score (Test) vs Epoches
            plt.figure(6)
            plt.plot(epoches, test_f1)
            plt.title("F1 score (Test) vs Epoches")
            plt.ylabel('F1 Score')
            plt.xlabel('Epoches')
            plt.savefig('Plots/F1 score (Test) vs Epoches')

            plt.show()
        return train_info

    def compute_tp_tn_fn_fp(self, y_true, y_pred):
        '''
        True positive - actual = 1, predicted = 1
        False positive - actual = 1, predicted = 0
        False negative - actual = 0, predicted = 1
        True negative - actual = 0, predicted = 0
        '''
        tp = sum((y_true == 1) & (y_pred == 1))
        tn = sum((y_true == 0) & (y_pred == 0))
        fn = sum((y_true == 1) & (y_pred == 0))
        fp = sum((y_true == 0) & (y_pred == 1))
        return tp, tn, fp, fn

    def compute_precision(self, tp, fp):
        '''
        Precision = TP  / FP + TP 

        '''
        np.seterr(divide='ignore', invalid='ignore')
        return (tp * 100) / float(tp + fp)

    def compute_recall(self, tp, fn):
        '''
        Recall = TP /FN + TP 

        '''
        np.seterr(divide='ignore', invalid='ignore')
        return (tp * 100) / float(tp + fn)

    def compute_f1_score(self, y_true, y_pred):
        # calculates the F1 score
        np.seterr(divide='ignore', invalid='ignore')
        tp, tn, fp, fn = self.compute_tp_tn_fn_fp(y_true, y_pred)
        precision = self.compute_precision(tp, fp) / 100
        recall = self.compute_recall(tp, fn) / 100
        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score

    def predict(self, sentence):
        out, _ = self.forward(self.createInputs(sentence))
        probs = self.softmax(out)
        y = np.argmax(probs)
        if y == 1:
            print("Sentiment is positive, sentiment is {:.2f}% positive".format(
                float(probs[y][0]) * 100))
        else:
            print("Sentiment is negative, sentiment is {:.2f}% negative".format(
                float(probs[y][0]) * 100))
        return y
