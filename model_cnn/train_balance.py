import  numpy as np
import predata as pre
import model_cnn as model
from tensorflow.contrib import learn
import tensorflow as tf
import datetime

from  sklearn.metrics import  accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from  sklearn.metrics import recall_score
from  sklearn.metrics import precision_score
from  sklearn.metrics import f1_score
from gensim.models.keyedvectors import KeyedVectors
import random

lr = 0.005
window_size=31
batch_size=2056
filter_size=150
embedding_size=300
class_number=34
filter_sizes='2,3,4,5'
l2_lambad=3.0
drop_keep_out=0.5
epochs=600
pretrain_emb_path="../GoogleNews-vectors-negative300.bin"

def getava(true_label,prediction_label):
    real_trigger_number = 0
    for tt in true_label:
        if tt != 0:
            real_trigger_number = real_trigger_number + 1
    pre_trigger_number = 0
    for tt in prediction_label:
        if tt != 0:
            pre_trigger_number = pre_trigger_number + 1
    true_number = 0
    for i in range(len(true_label)):
        if true_label[i] != 0 and true_label[i] == prediction_label[i]:
            true_number = true_number + 1
    if pre_trigger_number != 0:
        precision = true_number / pre_trigger_number
    else:
        precision = 0
    if real_trigger_number != 0:
        recall = true_number / real_trigger_number
    else:
        recall = 0
    if precision * recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
   # print(real_trigger_number,pre_trigger_number,true_number)
    return precision, recall, f1


def getlabel(predictions,labels):
	true_label = []
	prediction_label = predictions
	for i, yy in enumerate(labels):
		true_label.append(np.argmax(yy))

	return  true_label,prediction_label

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 # 1124/64
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def loadEmbMatrix(pretrain_emb_path,text,embed_size):
    print('Indexing word vectors.')
   # print(text)
    model = KeyedVectors.load_word2vec_format(pretrain_emb_path, binary=True)
    #print(model['None'])
    embedding_matrix = np.zeros((len(text), embed_size), dtype='float32')
    word_dict = {}
    for i, tt in enumerate(text):
        if i % 3000 == 0:
            print(i)
        try:
            word_dict[tt] = i
            embedding_matrix[i] = model[tt]
        except KeyError:
            print(tt)

    return embedding_matrix, word_dict


def main():
    train_sentences, test_sentences, dev_sentences, train_trigger_labels, test_trigger_labels, dev_trigger_labels, \
	train_entity_labls, test_entity_labls, dev_entity_labls,\
    train_sequences,test_sequences,dev_sequences,\
    train_positions,tset_positions,dev_positions,\
    train_entitys,test_entitys,dev_entitys,\
    train_labels,test_labels,dev_labels=pre.predata()

    print(len(train_sequences),len(test_sequences),len(dev_sequences))   #[[[w1][w2]...[wn]][][]...[]]

    words =train_sentences+test_sentences+dev_sentences
    print(len(words))
    words_new = ['None']  #没有重复的  出现过得 词
    for ww in words:
       # print(ww)
        for www in ww:
            if www not in words_new:
                words_new.append(www)
    #words_new.append('None')

    emb_matrix, word_dict = loadEmbMatrix(pretrain_emb_path, words_new, embedding_size)
   # print(emb_matrix[0],emb_matrix[1])


    x_train = []  #[0,1,2,...]
    for tt in train_sequences:
        se_sentence=[]
        for ttt in tt:
            se = [word_dict[tttt] for tttt in ttt]
            se_sentence.append(se)
        x_train.append(se_sentence)
    #print(x_train[0][0],train_sequences[0][0])

    x_dev = []
    for tt in dev_sequences:
        se_sentence = []
        for ttt in tt:
            se = [word_dict[tttt] for tttt in ttt]
            se_sentence.append(se)
        x_dev.append(se_sentence)

    x_test = []
    for tt in test_sequences:
        se_sentence = []
        for ttt in tt:
            se = [word_dict[tttt] for tttt in ttt]
            se_sentence.append(se)
        x_test.append(se_sentence)

    gcn = model.Gcns(  #实际是cnn  懒得改了。
        sequence_length=window_size,
        num_classes=class_number,
        vocab_size=len(words_new),
        embedding_size=embedding_size,
        filter_sizes=list(map(int, filter_sizes.split(","))),
        num_filters=filter_size,
        emb_matrix=emb_matrix,
        l2_reg_lambda=l2_lambad
    )

    train_op = tf.train.AdamOptimizer(learning_rate=lr,epsilon=1e-6).minimize(gcn.loss)
    print(lr)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    def train_step(x_batch, position_indexs,entity_indexs,y_batch):
        """
        A single training step
        """
        feed_dict = {
            gcn.input_x: x_batch,
            gcn.input_y: y_batch,
            gcn.position_index: position_indexs,
            gcn.entity_index: entity_indexs,
            gcn.dropout_keep_prob: drop_keep_out
        }
        _, loss, accuracy, predictions = sess.run(
            [train_op, gcn.loss, gcn.accuracy, gcn.predictions], feed_dict)
        return loss, accuracy, predictions

    def dev_step(x_batch, position_indexs,entity_indexs, y_batch):

        feed_dict = {
            gcn.input_x: x_batch,
            gcn.input_y: y_batch,
            gcn.position_index: position_indexs,
            gcn.entity_index:entity_indexs,
            gcn.dropout_keep_prob: 1.0
        }
        loss, accuracy, predictions = sess.run([gcn.loss, gcn.accuracy, gcn.predictions], feed_dict)

        return loss, accuracy, predictions

    x = []
    x_position = []
    x_entity= []
    y= []

    for xx in x_train:
        for xxx in xx:
            x.append(xxx)
    for xx in train_positions:
        for xxx in xx:
            x_position.append(xxx)
    for xx in train_entitys:
        for xxx in xx:
            x_entity.append(xxx)
    for yy in train_labels:
        for yyy in yy:
            y.append(yyy)


    x_batch_se_dev = []
    x_position_se_dev = []
    x_entity_se_dev = []
    y_batch_se_dev = []

    for xx in x_dev:
        for xxx in xx:
            x_batch_se_dev.append(xxx)
    for xx in dev_positions:
        for xxx in xx:
            x_position_se_dev.append(xxx)
    for xx in dev_entitys:
        for xxx in xx:
            x_entity_se_dev.append(xxx)
    for yy in dev_labels:
        for yyy in yy:
            y_batch_se_dev.append(yyy)


    # Generate batches
    batches = batch_iter(
        list(zip(x,x_position,x_entity,y)), batch_size, epochs)
    best_f1=0.0
    # Training loop. For each batch...
    for k, batch in enumerate(batches):
       # print(k)
        x_batch,x_position_batch,x_entity_batch,y_batch = zip(*batch)

        number = []
        number_trigger = []
        number_trigger_13 = []
        for i, ll in enumerate(y_batch):
            if ll == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0]:
                number.append(i)
            else:
                if ll == [0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0]:
                    number_trigger_13.append(i)
                number_trigger.append(i)  # qule13
       # print(len(number), len(number_trigger), len(number_trigger_13))

        x_balance = []
        x_position_balance = []
        x_entity_balance = []
        y_balance = []
        random.shuffle(number)
        for i in range(int(len(number_trigger_13)*2.5)+0):  # 控制非trigger的数量
            x_balance.append(x_batch[number[i]])
            x_position_balance.append(x_position_batch[number[i]])
            x_entity_balance.append(x_entity_batch[number[i]])
            y_balance.append(y_batch[number[i]])
        for i in range(len(number_trigger)):  # qu13
            x_balance.append(x_batch[number_trigger[i]])
            x_position_balance.append(x_position_batch[number_trigger[i]])
            x_entity_balance.append(x_entity_batch[number_trigger[i]])
            y_balance.append(y_batch[number_trigger[i]])

      # print(len(x_balance))

        loss_train, acc_train, prediction_bath = train_step(x_balance,x_position_balance,x_entity_balance, y_balance)

        if k % (int(len(x) / batch_size / 2)) == 0:

            loss_dev, acc_dev, prediction_dev = dev_step(x_batch_se_dev,x_position_se_dev,x_entity_se_dev,y_batch_se_dev)
            true_label_dev, prediction_label_dev = getlabel(prediction_dev,y_batch_se_dev)
            prec_dev, recall_dev, f1_dev = getava(true_label_dev, prediction_label_dev)

            if f1_dev>best_f1:  #如果f1有提升
                best_f1 = f1_dev
                print("epoch {:g} train loss {:g}, acc {:g}".format( k / (int(len(x) / batch_size / 2)),loss_train, acc_train))
                print(prediction_label_dev[0:20])
                print(true_label_dev[0:20])
                print("dev loss {:g}, acc {:g} precison {:g} recall {:g} f1 {:g}".format(loss_dev, acc_dev, prec_dev,
                                                                                     recall_dev, f1_dev))


if __name__ == '__main__':
    main()

