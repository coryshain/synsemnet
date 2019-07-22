import re
import tensorflow as tf


parse_initializer = re.compile('(.*_initializer)(_(.*))?')


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def get_activation(activation, session=None, training=True, from_logits=True, sample_at_train=True, sample_at_eval=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if isinstance(activation, str):
                    if activation.lower().startswith('cla'):
                        _, lb, ub = activation.split('_')
                        if lb in ['None', '-inf']:
                            lb = None
                        else:
                            lb = float(lb)
                        if ub in ['None', 'inf']:
                            ub = None
                        else:
                            ub = float(ub)
                        out = make_clipped_linear_activation(lb=lb, ub=ub, session=session)
                    elif activation.lower() == 'hard_sigmoid':
                        out = hard_sigmoid
                    elif activation.lower() == 'bsn':
                        def make_sample_fn(s, from_logits):
                            if from_logits:
                                def sample_fn(x):
                                    return bernoulli_straight_through(tf.sigmoid(x), session=s)
                            else:
                                def sample_fn(x):
                                    return bernoulli_straight_through(x, session=s)

                            return sample_fn

                        def make_round_fn(s, from_logits):
                            if from_logits:
                                def round_fn(x):
                                    return round_straight_through(tf.sigmoid(x), session=s)
                            else:
                                def round_fn(x):
                                    return round_straight_through(x, session=s)

                            return round_fn

                        sample_fn = make_sample_fn(session, from_logits)
                        round_fn = make_round_fn(session, from_logits)

                        if sample_at_train:
                            train_fn = sample_fn
                        else:
                            train_fn = round_fn

                        if sample_at_eval:
                            eval_fn = sample_fn
                        else:
                            eval_fn = round_fn

                        out = lambda x: tf.cond(training, lambda: train_fn(x), lambda: eval_fn(x))

                    elif activation.lower().startswith('slow_sigmoid'):
                        split = activation.split('_')
                        if len(split) == 2:
                            # Default to a slowness parameter of 1/2
                            scale = 0.5
                        else:
                            try:
                                scale = float(split[2])
                            except ValueError:
                                raise ValueError('Parameter to slow_sigmoid must be a valid float.')

                        out = lambda x: tf.sigmoid(0.5 * x)

                    else:
                        out = getattr(tf.nn, activation)
                else:
                    out = activation
            else:
                out = lambda x: x

    return out


def get_initializer(initializer, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(initializer, str):
                initializer_name, _, initializer_params = parse_initializer.match(initializer).groups()

                kwargs = {}
                if initializer_params:
                    kwarg_list = initializer_params.split('-')
                    for kwarg in kwarg_list:
                        key, val = kwarg.split('=')
                        try:
                            val = float(val)
                        except Exception:
                            pass
                        kwargs[key] = val

                tf.keras.initializers.he_normal()

                if 'identity' in initializer_name:
                    return tf.keras.initializers.Identity
                elif 'he_' in initializer_name:
                    return tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='normal')
                else:
                    out = getattr(tf, initializer_name)
                    if 'glorot' in initializer:
                        out = out()
                    else:
                        out = out(**kwargs)
            else:
                out = initializer

            return out


def get_regularizer(init, scale=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)
            else:
                raise ValueError('Unrecognized value "%s" for init parameter of get_regularizer()' %init)

            return out


def initialize_embeddings(categories, dim, default=0., name=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            categories = sorted(list(set(categories)))
            n_categories = len(categories)
            index_table = tf.contrib.lookup.index_table_from_tensor(
                tf.constant(categories),
                num_oov_buckets=1
            )
            embedding_matrix = tf.Variable(tf.fill([n_categories+1, dim], default), name=name)

            return index_table, embedding_matrix


def compose_lambdas(lambdas):
    def composed_lambdas(x, **kwargs):
        out = x
        for l in lambdas:
            out = l(out, **kwargs)
        return out

    return composed_lambdas


def make_lambda(layer, session=None, use_kwargs=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if use_kwargs:
                def apply_layer(x, **kwargs):
                    return layer(x, **kwargs)
            else:
                def apply_layer(x, **kwargs):
                    return layer(x)
            return apply_layer


def make_bi_rnn_layer(fwd, bwd, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            def bi_rnn(x, fwd=fwd, bwd=bwd, mask=None):
                f = fwd(x, mask=mask)
                if mask is not None:
                    mask = tf.reverse(mask, axis=[1])
                b = bwd(tf.reverse(x, axis=[1]), mask=mask)
                out = tf.concat([f, b], axis=-1)
                return out

            return bi_rnn


class DenseLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            activation=None,
            batch_normalization_decay=0.9,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.name = name

        self.dense_layer = None
        self.projection = None

        self.initializer = get_initializer(kernel_initializer, self.session)

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.units is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.units

            with self.session.as_default():
                with self.session.graph.as_default():
                    self.dense_layer = tf.layers.Dense(
                        out_dim,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        _reuse=self.reuse,
                        name=self.name
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.dense_layer(inputs)

                if self.normalize_weights:
                    self.w = self.dense_layer.kernel
                    self.g = tf.Variable(tf.ones(self.w.shape[1]), dtype=tf.float32)
                    self.v = tf.norm(self.w, axis=0)
                    self.dense_layer.kernel = self.v

                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
                    )
                if self.activation is not None:
                    H = self.activation(H)

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)


class DenseResidualLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias

        self.layers_inner = layers_inner
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation_inner = get_activation(activation_inner, session=self.session, training=self.training)
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.name = name

        self.dense_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.units is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.units

                    self.dense_layers = []

                    for i in range(self.layers_inner):
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        l = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )
                        self.dense_layers.append(l)

                    if self.project_inputs:
                        if self.name:
                            name = self.name + '_projection'
                        else:
                            name = None

                        self.projection = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                F = inputs
                for i in range(self.layers_inner - 1):
                    F = self.dense_layers[i](F)
                    if self.batch_normalization_decay:
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None,
                            reuse=self.reuse,
                            scope=name
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.dense_layers[-1](F)
                if self.batch_normalization_decay:
                    if self.name:
                        name = self.name + '_i%d' % (self.layers_inner - 1)
                    else:
                        name = None
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name
                    )

                if self.project_inputs:
                    x = self.projection(inputs)
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)


class RNNLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            activation=None,
            recurrent_activation='sigmoid',
            kernel_initializer='he_normal_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            return_sequences=True,
            batch_normalization_decay=None,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.recurrent_activation = get_activation(recurrent_activation, session=self.session, training=self.training)
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.batch_normalization_decay = batch_normalization_decay
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    RNN = tf.keras.layers.LSTM

                    if self.units:
                        output_dim = self.units
                    else:
                        output_dim = inputs.shape[-1]

                    self.rnn_layer = RNN(
                        output_dim,
                        return_sequences=self.return_sequences,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        name=self.name
                    )

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.rnn_layer(inputs, mask=mask)
                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)