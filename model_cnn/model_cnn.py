import tensorflow as tf

from tensorflow.python.ops import array_ops
class Gcns(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, emb_matrix1,emb_matrix2, l2_reg_lambda=0.0, ):

        # Placeholders for input, output and dropout
        self.position_index = tf.placeholder(tf.int32, [None, sequence_length], name="position")
        self.entity_index = tf.placeholder(tf.int32, [None,sequence_length], name="entity")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #word
            self.W2 = tf.Variable(initial_value=emb_matrix2, name='embedding_matrix2', dtype=tf.float32, trainable=False)
            self.W1= tf.Variable(initial_value=emb_matrix1, name='embedding_matrix1',dtype=tf.float32, trainable=True)
            self.W=tf.concat([self.W2,self.W1],axis=0)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)#31*300
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #postion
            self.position =  tf.Variable(tf.random_uniform(shape=[sequence_length,50], minval=-0.1,maxval=0.1),name='position_embedding', dtype=tf.float32, trainable=True)
            self.position_embedding = tf.nn.embedding_lookup(self.position, self.position_index)

            self.position_embedding_expanded = tf.expand_dims(self.position_embedding, -1)
            #entity
            self.entity = tf.Variable(tf.random_uniform(shape=[15, 50], minval=-0.1, maxval=0.1),
                                        name='entity_embedding', dtype=tf.float32, trainable=True)
            self.entity_embedding = tf.nn.embedding_lookup(self.entity, self.entity_index)
            self.entity_embedding_expanded = tf.expand_dims(self.entity_embedding, -1)
            #pinjie  400维度
            self.word_position_embedding_qu1=tf.concat([self.embedded_chars,self.position_embedding,self.entity_embedding],axis=2)
            self.word_position_embedding=tf.concat([self.embedded_chars_expanded,self.position_embedding_expanded,self.entity_embedding_expanded],axis=2)
           # print(self.word_position_embedding)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size+100, 1, num_filters] # (2,300,1,100)

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W") #W是我们的滤波器矩阵
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.word_position_embedding,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")#h是将非线性应用于卷积输出的结果
                # Maxpooling over the outputs
                print(h)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropoutDropout层随机地“禁用”其神经元的一部分
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        #h_drop 为(batch_size,num_fiters_total)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.pred_probas = tf.nn.softmax(self.scores, name='class_proba')
            self.predictions = tf.argmax(self.pred_probas, axis=-1, name='class_prediction')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.labels=tf.argmax(self.input_y, 1)
            self.labels=tf.cast(self.labels,dtype=tf.int32)
          #  self.loss = focal_loss(self.scores, self.input_y, alpha=0.75) + l2_reg_lambda * l2_loss
            losses=ranking_loss(self.labels,self.scores,self.batch_size)
           # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
           # self.loss = focal_loss(self.scores,self.input_y,alpha=0.75) + l2_reg_lambda * l2_loss

        tvars = tf.trainable_variables()
        print(tvars)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

def ranking_loss( labels, logits, batch_size):
    lm = tf.constant(1.0)  # lambda
    m_plus = tf.constant(2.5)
    m_minus = tf.constant(0.5)

    L = tf.constant(0.0)
    i = tf.constant(0)
    cond = lambda i, L: tf.less(i, batch_size)

    def loop_body(i, L):
        cplus = labels[i]  # positive class label index
        # taking most informative negative class, use 2nd argmax
        _, cminus_indices = tf.nn.top_k(logits[i, :], k=2)
        cminus = tf.cond(tf.equal(cplus, cminus_indices[0]),#如果第二大的是正确索引则为第三大的
                         lambda: cminus_indices[1], lambda: cminus_indices[0])

        splus = logits[i, cplus]  # score for gold class
        sminus = logits[i, cminus]  # score for negative class

        szere=logits[i,0]
        ###splus++  sminus--
        l = tf.log((1.0 + tf.exp((lm * (m_plus - splus))))) + \
            tf.log((1.0 + tf.exp((lm * (m_minus + sminus)))))

        return [tf.add(i, 1), tf.add(L, l)]

    _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])
    nbatch = tf.to_float(batch_size)
    L = L
    return L



