import tensorflow as tf
import numpy as np
import os, sys
from data_utils.DataManager import DataManager

class KDModel(object):
    def __init__(self, dataset_name, teacher_model, students_model):
        """
        The class of Knowledge Distillation Model

        dataset_name:   The name of your dataset, candidates are 'CIFAR-10', 'CIFAR-100', 'MNIST' 
                        (You can add new dataset by modifying the DataManager class).

        teacher_model:  The model function of your teacher model, signature should be 
                        teacherName(input_tensor, trainable, is_train, temp)
                        - input_tensor:     Input of your teacher model
                        - trainable:        If true, we will train your teacher model.
                        - is_train:         If true means that we are in training process, this will determine dropout.
                        - temp:             The temperature parameter.
        
        student_model:  The model function of your student model, signature should be
                        studentName(input_tensor, is_train), the meaning of parameters is same as above.
        """
        self.data_manager = DataManager(dataset_name)
        self.dataset_name = dataset_name
        self.teacher_model = teacher_model
        self.student_model = students_model

    def _writeRecord(self, path, name, data):
        """
        Write the record of some variables during some process into a file.
        :param path:    The path store your record files.
        :param name:    The name of file.
        :param data:    A list contains data you want to store.
        """
        file_path = os.path.join(path, name)
        with open(file_path, 'w') as f:
            for item in data:
                f.write(str(item)+'\t')
            f.write('\n')

    def TrainTeacher(self, model_name, **kwargs):
        """
        Train your teacher model.
        model_name: the name of your model. You can use the hyper-parameters with dataset name to name your model.

        kwargs:
        - batch_size:           Size of batch
        - model_save_path:      The path saving your teacher model value
        - num_epoch:            How many epochs in training process.
        - basic_learning_rate:  The initial learning rate
        - record_save_path:     The path saving training process record.
        - is_dev(dev_mode):     Development mode(i.e. using small dataset)
        - learning_rate_decay:  The decay rate of learning rate, here we use exp decay.
        - reg_scale:            The l2 regularization strength.
        - verbose:              Print some debug information.
        """
        batch_size = kwargs.pop("batch_size", 64)
        model_save_path = kwargs.pop('model_save_path', "./checkpoints/teacher/")
        num_epoch = kwargs.pop("num_epoch", 10)
        basic_learning_rate = kwargs.pop("basic_learning_rate", 5e-4)
        record_save_path = kwargs.pop("record_save_path", "./records/teacher")
        is_dev = kwargs.pop("dev_mode", False)
        learning_rate_decay = kwargs.pop("learning_rate_decay", 0.01)
        reg_scale = kwargs.pop("reg_scale", 1e-1)
        verbose = kwargs.pop("verbose", False)

        # Do some check
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(record_save_path):
            os.makedirs(record_save_path)
        model_save_path = os.path.join(model_save_path, "{}.ckpt".format(model_name))
        
        tf.reset_default_graph()
        
        # Get training dataset
        if is_dev:
            train_data, train_label = self.data_manager.dev_data, self.data_manager.dev_label
        else:
            train_data, train_label = self.data_manager.train_data, self.data_manager.train_label
        
        num_train_data = train_data.shape[0]

        # The input of teacher model
        X = tf.placeholder(train_data.dtype, [None]+list(train_data.shape[1:]), name="input_data")
        y = tf.placeholder(train_label.dtype, [None]+list(train_label.shape[1:]), name="input_label")
        is_train = tf.placeholder(tf.bool, name="is_train")
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=8000)
        batched_dataset = dataset.batch(batch_size)

        iterator = batched_dataset.make_initializable_iterator()
        batch_data, batch_label = iterator.get_next()

        # Get the teacher model
        #logits, probs = self.teacher_models[self.dataset_name](batch_data, is_train=is_train, reg_scale=reg_scale)
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
        with tf.variable_scope('teacher_model', regularizer=regularizer):
            logits, probs = self.teacher_model(batch_data, is_train=is_train)
        result = tf.argmax(logits, axis=1)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(result, tf.argmax(batch_label, axis=1)), tf.float32))
        saver = tf.train.Saver()

        # Training part
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label, name="cross_entropy_loss"))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'teacher_model'))
        loss += reg_loss
        global_step = tf.get_variable('global_step', initializer=0.0, trainable=False)
        learning_rate = tf.train.natural_exp_decay(
            basic_learning_rate, global_step,
            decay_rate=learning_rate_decay,
            name='learning_rate', decay_steps=1
        )
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        global_step_add = tf.assign_add(global_step, 1)

        train_acc_hist = []
        val_acc_hist = []
        train_loss_hist = []
        best_acc = 0.0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_epoch):
                sess.run(iterator.initializer, feed_dict={X:train_data, y:train_label})
                cnt = 0
                total_correct_cnt = 0
                total_loss, acc = 0.0, 0.0
                while True:
                    try:
                        curr_loss, train, right_num, curr_result = sess.run(
                            [loss, train_op, correct_num, result],
                            feed_dict={is_train: True}
                        )
                        total_correct_cnt += right_num
                        total_loss += curr_loss
                        cnt += 1
                    except tf.errors.OutOfRangeError:
                        acc = total_correct_cnt * 1.0 / num_train_data
                        last_loss = total_loss / cnt 
                        if verbose:
                            div = "==========================="
                            print("{}\nEpoch {}/{}\t\tloss: {}\t\tacc: {}".format(div, i+1, num_epoch, last_loss, acc))
                        train_acc_hist.append(acc)
                        train_loss_hist.append(last_loss)
                        sess.run([global_step_add])
                        if verbose:
                            last_global_step, last_learning_rate = sess.run([global_step, learning_rate])
                            print("learning_rate: {}".format(last_learning_rate))
                        break
                    
                # Validation
                sess.run(iterator.initializer, feed_dict={X:self.data_manager.val_data, y:self.data_manager.val_label})
                acc = 0.0
                total_correct_cnt = 0
                while True:
                    try:
                        right_num = sess.run([correct_num], feed_dict={is_train:False})
                        total_correct_cnt += right_num[0]
                    except tf.errors.OutOfRangeError:
                        acc = total_correct_cnt * 1.0 / self.data_manager.val_data.shape[0]
                        if verbose:
                            print("Validation acc: {}".format(acc))
                        val_acc_hist.append(acc)
                        if acc > best_acc:
                            best_acc = acc
                            saver.save(sess, model_save_path)
                        break
        # Write train process record
        self._writeRecord(record_save_path, "{}_train_accuracy".format(model_name), train_acc_hist)
        self._writeRecord(record_save_path, "{}_validation_accuracy".format(model_name), val_acc_hist)
        self._writeRecord(record_save_path, "{}_train_loss".format(model_name), train_loss_hist)
        if verbose:
            print("Finish Training Teacher Model! The Best Validation Accuracy is: {}".format(best_acc))

    
    def TestTeacher(self, model_name, **kwargs):
        """
        Test your teacher model.

        model_name:             The name of your pretrained teacher model.
        kwargs:
        - batch_size:           Size of batch
        - model_save_path:      The path that you store your pretrained teacher model.
        - record_save_path:     The path saving training process record.
        - verbose:              Print some debug information.
        """
        batch_size = kwargs.pop("batch_size", 256)
        model_save_path = kwargs.pop("model_save_path", "./checkpoints/teacher/")
        record_save_path = kwargs.pop("record_save_path", "./records/teacher")
        verbose = kwargs.pop("verbose", False)

        # Do some check
        if not os.path.exists(model_save_path):
            raise RuntimeError("No pretrained model exists in '{}'".format(model_save_path))
        if not os.path.exists(record_save_path):
            os.makedirs(record_save_path)
        model_save_path = os.path.join(model_save_path, "{}.ckpt".format(model_name))

        tf.reset_default_graph()

        # Get dataset
        test_data, test_label = self.data_manager.test_data, self.data_manager.test_label
        num_test_data = test_data.shape[0]

        X = tf.placeholder(test_data.dtype, shape=[None]+list(test_data.shape[1:]), name="input_data")
        y = tf.placeholder(test_label.dtype, shape=[None]+list(test_label.shape[1:]), name="input_label")
        is_train = tf.placeholder(tf.bool, name="is_train")

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        batched_dataset = dataset.batch(batch_size)
        
        iterator = batched_dataset.make_initializable_iterator()
        batch_data, batch_label = iterator.get_next()

        # Get the teacher model
        with tf.variable_scope('teacher_model'):
            logits, probs = self.teacher_model(batch_data, is_train=is_train)
        result = tf.argmax(logits, axis=1)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(result, tf.argmax(batch_label, axis=1)), tf.float32))
        saver = tf.train.Saver()

        # Test process
        with tf.Session() as sess:
            sess.run(iterator.initializer, feed_dict={X:test_data, y:test_label})
            saver.restore(sess, model_save_path)
            total_correct_cnt = 0
            while True:
                try:
                    right_num = sess.run([correct_num], feed_dict={is_train:False})
                    total_correct_cnt += right_num[0]
                except tf.errors.OutOfRangeError:
                    acc = total_correct_cnt * 1.0 / num_test_data
                    if verbose:
                        print("Test accuracy: {}".format(acc))
                    break
        acc_hist = [acc]
        self._writeRecord(record_save_path, "{}_test_accuracy".format(model_name), acc_hist)


    def TrainStudent(self, model_name, teacher_model_name, **kwargs):
        """
        Train your student model.
        
        model_name:             The name of your student model.
        teacher_model_name:     The name of your teacher model.
        
        kwargs:
        - batch_size:           Size of batch
        - model_save_path:      The path saving your teacher model value
        - teacher_model_path:   The path that contains your pretrained teacher model.
        - temp:                 The temperature parameter.
        - num_epoch:            How many epochs in training process.
        - basic_learning_rate:  The initial learning rate
        - record_save_path:     The path saving training process record.
        - is_dev(dev_mode):     Development mode(i.e. using small dataset)
        - learning_rate_decay:  The decay rate of learning rate, here we use exp decay.
        - reg_scale:            The l2 regularization strength.
        - soft_target_scale:    The mix weight of soft target denoted by lambda, loss = hard_target_loss + lambda * soft_target_loss 
        - verbose:              Print some debug information.
        """
        batch_size = kwargs.pop("batch_size", 64)
        model_save_path = kwargs.pop('model_save_path', "./checkpoints/student/")
        teacher_model_path = kwargs.pop("teacher_model_path", "./checkpoints/teacher/")
        temp = kwargs.pop("temp", 10)
        num_epoch = kwargs.pop("num_epoch", 20)
        basic_learning_rate = kwargs.pop("basic_learning_rate", 5e-4)
        record_save_path = kwargs.pop("record_save_path", "./records/student")
        is_dev = kwargs.pop("dev_mode", False)
        learning_rate_decay = kwargs.pop("learning_rate_decay", 0.01)
        reg_scale = kwargs.pop("reg_scale", 1e-1)
        soft_target_scale = kwargs.pop("soft_target_scale", 1)
        verbose = kwargs.pop("verbose", False)

        # Do some check
        if not os.path.exists(teacher_model_path):
            raise RuntimeError("Cannot find pretrained teacher model in '{}'".format(teacher_model_path))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(record_save_path):
            os.makedirs(record_save_path)

        model_save_path = os.path.join(model_save_path, "{}.ckpt".format(model_name))
        teacher_model_path = os.path.join(teacher_model_path, "{}.ckpt".format(teacher_model_name))

        tf.reset_default_graph()
        
        # Get training dataset
        if is_dev:
            train_data, train_label = self.data_manager.dev_data, self.data_manager.dev_label
        else:
            train_data, train_label = self.data_manager.train_data, self.data_manager.train_label
        
        num_train_data = train_data.shape[0]

        # The input of model
        X = tf.placeholder(train_data.dtype, [None]+list(train_data.shape[1:]), name="input_data")
        y = tf.placeholder(train_label.dtype, [None]+list(train_label.shape[1:]), name="input_label")
        is_train = tf.placeholder(tf.bool, name="is_train")
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=8000)
        batched_dataset = dataset.batch(batch_size)

        iterator = batched_dataset.make_initializable_iterator()
        batch_data, batch_label = iterator.get_next()

        # Get the teacher and student model
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
        with tf.variable_scope('student_model', regularizer=regularizer):
            logits, probs = self.student_model(batch_data, is_train=is_train)

        with tf.variable_scope('teacher_model'):
            teacher_logits, teacher_probs = self.teacher_model(batch_data, is_train=False, trainable=False, temp=temp)

        result = tf.argmax(logits, axis=1)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(result, tf.argmax(batch_label, axis=1)), tf.float32))

        teacher_variabels = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="teacher_model")
        student_variabels = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="student_model")
        teacher_loader = tf.train.Saver(teacher_variabels)
        student_saver = tf.train.Saver(student_variabels)
        
        # Training part
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label, name="hard_loss"))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'teacher_model'))
        loss += reg_loss
        soft_target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=teacher_probs, name="soft_loss"))
        loss += soft_target_scale * soft_target_loss
        
        global_step = tf.get_variable('global_step', initializer=0.0, trainable=False)
        learning_rate = tf.train.natural_exp_decay(
            basic_learning_rate, global_step,
            decay_rate=learning_rate_decay,
            name='learning_rate', decay_steps=1
        )
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        global_step_add = tf.assign_add(global_step, 1)

        train_acc_hist = []
        val_acc_hist = []
        train_loss_hist = []
        best_acc = 0.0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            teacher_loader.restore(sess, teacher_model_path)
            for i in range(num_epoch):
                sess.run(iterator.initializer, feed_dict={X:train_data, y:train_label})
                cnt = 0
                total_correct_cnt = 0
                total_loss, acc = 0.0, 0.0
                while True:
                    try:
                        curr_loss, train, right_num, curr_result = sess.run(
                            [loss, train_op, correct_num, result],
                            feed_dict={is_train: True}
                        )
                        total_correct_cnt += right_num
                        total_loss += curr_loss
                        cnt += 1
                    except tf.errors.OutOfRangeError:
                        acc = total_correct_cnt * 1.0 / num_train_data
                        last_loss = total_loss / cnt 
                        if verbose:
                            div = "==========================="
                            print("{}\nEpoch {}/{}\t\tloss: {}\t\tacc: {}".format(div, i+1, num_epoch, last_loss, acc))
                        train_acc_hist.append(acc)
                        train_loss_hist.append(last_loss)
                        sess.run([global_step_add])
                        if verbose:
                            last_global_step, last_learning_rate = sess.run([global_step, learning_rate])
                            print("learning_rate: {}".format(last_learning_rate))
                        break
                    
                # Validation
                sess.run(iterator.initializer, feed_dict={X:self.data_manager.val_data, y:self.data_manager.val_label})
                acc = 0.0
                total_correct_cnt = 0
                while True:
                    try:
                        right_num = sess.run([correct_num], feed_dict={is_train:False})
                        total_correct_cnt += right_num[0]
                    except tf.errors.OutOfRangeError:
                        acc = total_correct_cnt * 1.0 / self.data_manager.val_data.shape[0]
                        if verbose:
                            print("Validation acc: {}".format(acc))
                        val_acc_hist.append(acc)
                        if acc > best_acc:
                            best_acc = acc
                            student_saver.save(sess, model_save_path)
                        break
        # Write train process record
        self._writeRecord(record_save_path, "{}_train_accuracy".format(model_name), train_acc_hist)
        self._writeRecord(record_save_path, "{}_validation_accuracy".format(model_name), val_acc_hist)
        self._writeRecord(record_save_path, "{}_train_loss".format(model_name), train_loss_hist)
        if verbose:
            print("Finish Training Student Model! The Best Validation Accuracy is: {}".format(best_acc))

    def TestStudent(self, model_name, **kwargs):
        """
        Test your student model.

        model_name:             The name of your student model.
        kwargs:
        - batch_size:           Size of batch
        - model_save_path:      The path that you store your pretrained student model.
        - record_save_path:     The path saving training process record.
        - verbose:              Print some debug information.
        """
        batch_size = kwargs.pop("batch_size", 256)
        model_save_path = kwargs.pop("model_save_path", "./checkpoints/student/")
        record_save_path = kwargs.pop("record_save_path", "./records/student")
        verbose = kwargs.pop("verbose", False)

        # Do some check
        if not os.path.exists(model_save_path):
            raise RuntimeError("No pretrained model exists in '{}'".format(model_save_path))
        if not os.path.exists(record_save_path):
            os.makedirs(record_save_path)
        model_save_path = os.path.join(model_save_path, "{}.ckpt".format(model_name))

        tf.reset_default_graph()

        # Get dataset
        test_data, test_label = self.data_manager.test_data, self.data_manager.test_label
        num_test_data = test_data.shape[0]

        X = tf.placeholder(test_data.dtype, shape=[None]+list(test_data.shape[1:]), name="input_data")
        y = tf.placeholder(test_label.dtype, shape=[None]+list(test_label.shape[1:]), name="input_label")
        is_train = tf.placeholder(tf.bool, name="is_train")

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        batched_dataset = dataset.batch(batch_size)
        
        iterator = batched_dataset.make_initializable_iterator()
        batch_data, batch_label = iterator.get_next()

        # Get the student model
        with tf.variable_scope('student_model'):
            logits, probs = self.student_model(batch_data, is_train=is_train)
        result = tf.argmax(logits, axis=1)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(result, tf.argmax(batch_label, axis=1)), tf.float32))
        saver = tf.train.Saver()

        # Test process
        with tf.Session() as sess:
            sess.run(iterator.initializer, feed_dict={X:test_data, y:test_label})
            saver.restore(sess, model_save_path)
            total_correct_cnt = 0
            while True:
                try:
                    right_num = sess.run([correct_num], feed_dict={is_train:False})
                    total_correct_cnt += right_num[0]
                except tf.errors.OutOfRangeError:
                    acc = total_correct_cnt * 1.0 / num_test_data
                    if verbose:
                        print("Test accuracy: {}".format(acc))
                    break
        acc_hist = [acc]
        self._writeRecord(record_save_path, "{}_test_accuracy".format(model_name), acc_hist)




        
        





        
            
                     
                        

