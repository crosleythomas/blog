import tensorflow as tf

########################
###   Define model   ###
########################

def model_fn(features, labels, mode, params):
    # 1. Define model structure
    for l in range(params.num_layers):
        lparams = params.layers[l]
        if l == 0:
            h = features['image']
        elif lparams['type'] == 'fc' and len(h.get_shape().as_list()) != 2:
            h = tf.contrib.layers.flatten(h)
        if lparams['type'] == 'conv':
            h = tf.contrib.layers.conv2d(h, lparams['num_outputs'], lparams['kernel_size'], lparams['stride'], activation_fn=lparams['activation'], weights_regularizer=lparams['regularizer'])
        elif lparams['type'] == 'pool':
            h = tf.contrib.layers.max_pool2d(h, lparams['kernel_size'], lparams['stride'])
        elif lparams['type'] == 'fc':
            h = tf.contrib.layers.fully_connected(h, lparams['num_outputs'], activation_fn=lparams['activation'])

    # 2. Generate predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'output': h}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # 3. Define the loss functions
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=h))
    
    # 3.1 Additional metrics for monitoring
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
          labels=labels, predictions=tf.argmax(h, axis=-1))}
    
    # 4. Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    # 5. Return training/evaluation EstimatorSpec
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

###################################
###   Define Hyper Parameters   ###
###################################

params = tf.contrib.training.HParams(
    layers = [
        {'type': 'conv', 'num_outputs' : 96, 'kernel_size' : 11, 'stride' : 4, 'activation' : tf.nn.relu, 'regularizer' : tf.nn.l2_loss}, 
        {'type': 'pool', 'kernel_size' : 3, 'stride' : 2},
        {'type': 'conv', 'num_outputs' : 256, 'kernel_size' : 5, 'stride' : 1, 'activation' : tf.nn.relu, 'regularizer' : tf.nn.l2_loss},
        {'type': 'pool', 'kernel_size' : 3, 'stride' : 2},
        {'type': 'conv', 'num_outputs' : 384, 'kernel_size' : 3, 'stride' : 1, 'activation' : tf.nn.relu, 'regularizer' : tf.nn.l2_loss},
        {'type': 'conv', 'num_outputs' : 384, 'kernel_size' : 3, 'stride' : 1, 'activation' : tf.nn.relu, 'regularizer' : tf.nn.l2_loss},
        {'type': 'conv', 'num_outputs' : 256, 'kernel_size' : 3, 'stride' : 1, 'activation' : tf.nn.relu, 'regularizer' : tf.nn.l2_loss},
        {'type': 'pool', 'kernel_size' : 3, 'stride' : 2},
        {'type': 'fc', 'num_outputs' : 4096, 'activation' : tf.nn.relu},
        {'type': 'fc', 'num_outputs' : 2048, 'activation' : tf.nn.relu},
        {'type': 'fc', 'num_outputs' : 50, 'activation' : None}
    ],
    learning_rate = 0.001,
    train_epochs = 30,
    batch_size = 32,
    image_height = 300,
    image_width = 200,
    image_depth = 3
)
params.add_hparam('num_layers', len(params.layers))

# Run Configuration
config = tf.contrib.learn.RunConfig(
    tf_random_seed=0,
    save_checkpoints_steps=250,
    save_checkpoints_secs=None,
    save_summary_steps=10,
)