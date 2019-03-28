import tensorflow as tf

class Gcns(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, emb_matrix, l2_reg_lambda=0.0, ):

        # Placeholders for input, output and dropout
        self.position_index = tf.placeholder(tf.int32, [None, 31], name="position")
        self.entity_index = tf.placeholder(tf.int32, [None, 31], name="entity")
        self.input_x = tf.placeholder(tf.int32, [None, 31], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #word
            self.W= tf.Variable(initial_value=emb_matrix, name='embedding_matrix',dtype=tf.float32, trainable=True)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #postion
            self.position =  tf.Variable(tf.random_uniform(shape=[16,50], minval=0,maxval=1),name='position_embedding', dtype=tf.float32, trainable=True)
            self.position_embedding = tf.nn.embedding_lookup(self.position, self.position_index)
            self.position_embedding = tf.expand_dims(self.position_embedding, -1)
            #entity
            self.entity = tf.Variable(tf.random_uniform(shape=[14, 50], minval=0, maxval=1),
                                        name='entity_embedding', dtype=tf.float32, trainable=True)
            self.entity_embedding = tf.nn.embedding_lookup(self.entity, self.entity_index)
            self.entity_embedding = tf.expand_dims(self.entity_embedding, -1)
            #pinjie  400维度
            self.word_position_embedding=tf.concat([self.embedded_chars_expanded,self.position_embedding,self.entity_embedding],axis=2)
            print(self.word_position_embedding)



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
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        tvars = tf.trainable_variables()
        print(tvars)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

