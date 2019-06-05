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
from sklearn.metrics import classification_report
lr = 0.001
window_size=31
batch_size=8000
filter_size=200
embedding_size=300
class_number=34
filter_sizes='2,3,4,5'
l2_lambad=3
drop_keep_out=0.5
epochs=2000
np.random.seed(4567)
pretrain_emb_path="../GoogleNews-vectors-negative300.bin"

def getava(true_label,prediction_label):
    real_trigger_number = 0
    real_trigger_label=[]
    pre_trigger_label=[]
    for tt in true_label:
        if tt != 0:
            real_trigger_number = real_trigger_number + 1
            real_trigger_label.append(tt)
    pre_trigger_number = 0
    pre_trigger_number_index=[]
    wrong_trigger_number_index=[]

    for i,tt in enumerate(prediction_label):
        if tt != 0:
            pre_trigger_number = pre_trigger_number + 1
            pre_trigger_label.append(tt)
    true_number = 0
    for i in range(len(true_label)):
        if true_label[i] != 0 and true_label[i] == prediction_label[i]:
            true_number = true_number + 1
            pre_trigger_number_index.append(i)
        elif true_label[i] != 0 and true_label[i] != prediction_label[i]:
            wrong_trigger_number_index.append(i)
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
    #print(real_trigger_number,pre_trigger_number,true_number)
   # print(classification_report(true_label, prediction_label))
    return precision, recall, f1,pre_trigger_number_index,wrong_trigger_number_index,real_trigger_label,pre_trigger_label


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
        if tt=='None':
            word_dict[tt]=i
        else:
            if i % 3000 == 0:
                print(i)
            try:
                word_dict[tt] = i
                embedding_matrix[i] = model[tt]
            except KeyError:
                continue

    return embedding_matrix, word_dict


def main():
    train_sentences, test_sentences, dev_sentences, train_trigger_labels, test_trigger_labels, dev_trigger_labels, \
	train_entity_labls, test_entity_labls, dev_entity_labls,\
    train_sequences,test_sequences,dev_sequences,\
    train_positions,test_positions,dev_positions,\
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
    print(len(word_dict))

    emb_matrix2=[]
    emb_matrix2.append(emb_matrix[0].tolist())
    print(emb_matrix2)

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
        emb_matrix1=emb_matrix[1:],
        emb_matrix2=emb_matrix2,
        l2_reg_lambda=l2_lambad
    )



    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(gcn.loss)
    print(lr)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    def train_step(x_batch, position_indexs,entity_indexs,y_batch,batch_size):
        """
        A single training step
        """
        feed_dict = {
            gcn.input_x: x_batch,
            gcn.input_y: y_batch,
            gcn.position_index: position_indexs,
            gcn.entity_index: entity_indexs,
            gcn.dropout_keep_prob: drop_keep_out,
            gcn.batch_size:batch_size,
        }
        _, loss, accuracy, predictions,scores,W_train = sess.run(
            [train_op, gcn.loss, gcn.accuracy, gcn.predictions,gcn.pred_probas,gcn.W], feed_dict)

        return loss, accuracy, predictions,scores,W_train

    def dev_step(x_batch, position_indexs,entity_indexs, y_batch,batch_size):

        feed_dict = {
            gcn.input_x: x_batch,
            gcn.input_y: y_batch,
            gcn.position_index: position_indexs,
            gcn.entity_index:entity_indexs,
            gcn.dropout_keep_prob:1,
            gcn.batch_size:batch_size,
        }
        loss, accuracy, predictions,scores = sess.run([gcn.loss, gcn.accuracy, gcn.predictions,gcn.pred_probas], feed_dict)

        return loss, accuracy, predictions,scores

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

    x_batch_se_test = []
    x_position_se_test = []
    x_entity_se_test = []
    y_batch_se_test = []

    for xx in x_test:
        for xxx in xx:
            x_batch_se_test.append(xxx)
    for xx in test_positions:
        for xxx in xx:
            x_position_se_test.append(xxx)
    for xx in test_entitys:
        for xxx in xx:
            x_entity_se_test.append(xxx)
    for yy in test_labels:
        for yyy in yy:
            y_batch_se_test.append(yyy)

    print(len(x),len(x_position),len(x_entity),len(y))
    print(x[0])

    # Generate batches
    batches = batch_iter(
        list(zip(x, x_position, x_entity, y)), batch_size, epochs)
    best_f1=0.0
    # Training loop. For each batch...
    for k, batch in enumerate(batches):
       # print(k)
        x_batch,x_position_batch,x_entity_batch,y_batch = zip(*batch)
       # print(len(x_batch),len(x_position_batch))

        number_triggers=[]
        for i in range(34):
            number_triggers.append([])
        for i, ll in enumerate(y_batch):
            for j in range(34):
                label=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0]
                label[j]=1
                if ll == label:
                    number_triggers[j].append(i)
       # print(number_triggers)

        x_balance = []
        x_position_balance = []
        x_entity_balance = []
        y_balance = []
        np.random.shuffle(number_triggers[0])
        if number_triggers[13]!=0:
            for i in range(int(len(number_triggers[13])*2.5)+0):  # 控制非trigger的数量
                x_balance.append(x_batch[number_triggers[0][i]])
                x_position_balance.append(x_position_batch[number_triggers[0][i]])
                x_entity_balance.append(x_entity_batch[number_triggers[0][i]])
                y_balance.append(y_batch[number_triggers[0][i]])
            for i in range(34):
                if i!=0:
                    try:
                        if int(len(number_triggers[13])/len(number_triggers[i])/3)==0:
                            for j in range(len(number_triggers[i])):
                                x_balance.append(x_batch[number_triggers[i][j]])
                                x_position_balance.append(x_position_batch[number_triggers[i][j]])
                                x_entity_balance.append(x_entity_batch[number_triggers[i][j]])
                                y_balance.append(y_batch[number_triggers[i][j]])
                        else:
                            for m in range(int(len(number_triggers[13])/len(number_triggers[i]))):
                                for j in range(len(number_triggers[i])):
                                    x_balance.append(x_batch[number_triggers[i][j]])
                                    x_position_balance.append(x_position_batch[number_triggers[i][j]])
                                    x_entity_balance.append(x_entity_batch[number_triggers[i][j]])
                                    y_balance.append(y_batch[number_triggers[i][j]])
                    except:
                        continue
        else:
            x_balance=x_batch
            x_position_balance=x_position_batch
            x_entity_balance=x_entity_batch
            y_balance=y_batch



        #print(len(x_balance))

        loss_train, acc_train, prediction_bath,scores_train,W_train= train_step(x_balance,x_position_balance,x_entity_balance, y_balance,len(x_balance))

       # print(W_train[0])

        if k % (int(len(x) / batch_size /10)) == 0:
            if k%500==0:
                print(k)
            loss_dev, acc_dev, prediction_dev,scores_dev = dev_step(x_batch_se_dev, x_position_se_dev, x_entity_se_dev,
                                                         y_batch_se_dev,len(x_batch_se_dev))


            true_label_dev, prediction_label_dev = getlabel(prediction_dev, y_batch_se_dev)
            prec_dev, recall_dev, f1_dev,pre_trigger_dev,wrong_trigger_dev,real_trigger_label_dev,pre_trigger_label_dev = getava(true_label_dev, prediction_label_dev)

            if f1_dev>best_f1 :  # 如果f1有提升
                best_f1 = f1_dev
                if True:
                    print("epoch {:g} train loss {:g}, acc {:g}".format(k / (int(len(x) / batch_size / 10)),
                                                                        loss_train,
                                                                        acc_train))
#
                    print("dev loss {:g}, acc {:g} precison {:g} recall {:g} f1 {:g}".format(loss_dev, acc_dev,
                                                                                             prec_dev,
                                                                                             recall_dev, f1_dev))
                    loss_test, acc_test, prediction_test,scores_test = dev_step(x_batch_se_test, x_position_se_test,
                                                                    x_entity_se_test,
                                                                    y_batch_se_test,len(x_batch_se_test))
                    true_label_test, prediction_label_test = getlabel(prediction_test, y_batch_se_test)

                    prec_test, recall_test, f1_test,pre_trigger_test,wrong_trigger_test,real_trigger_label_test,pre_trigger_label_test = getava(true_label_test, prediction_label_test)
                    if f1_test>0.0:

                        print(
                        "test loss {:g}, acc {:g} precison {:g} recall {:g} f1 {:g}".format(loss_test, acc_test,
                                                                                            prec_test,
                                                                                           recall_test, f1_test))

                        right_words=[]
                        wrong_words=[]

                        for ind in pre_trigger_test:
                            right_words.append(list (word_dict.keys()) [list (word_dict.values()).index (x_batch_se_test[ind][15])])
                        #real_trigger_label_test.append(ind)
                        for ind in wrong_trigger_test:
                            wrong_words.append(list (word_dict.keys()) [list (word_dict.values()).index (x_batch_se_test[ind][15])])
                        #wrong_labels.append()
                   # print(right_words)
                       # print(wrong_words)
                      #  print(classification_report(true_label_test, prediction_label_test))




if __name__ == '__main__':
    main()

