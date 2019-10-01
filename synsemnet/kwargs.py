from functools import cmp_to_key

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr, aliases=None):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases

    def dtypes_str(self):
        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def in_settings(self, settings):
        out = False
        if self.key in settings:
            out = True

        if not out:
            for alias in self.aliases:
                if alias in settings:
                    out = True
                    break

        return out

    def kwarg_from_config(self, settings):
        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, None)

            if val is None:
                for alias in self.aliases:
                    val = {
                        str: settings.get,
                        int: settings.getint,
                        float: settings.getfloat,
                        bool: settings.getboolean
                    }[self.dtypes[0]](alias, self.default_value)
                    if val is not None:
                        break

            if val is None:
                val = self.default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = self.default_value
            else:
                parsed = False
                for x in reversed(self.dtypes):
                    if x == None:
                        if from_settings == 'None':
                            val = None
                            parsed = True
                            break
                    elif isinstance(x, str):
                        if from_settings == x:
                            val = from_settings
                            parsed = True
                            break
                    else:
                        try:
                            val = x(from_settings)
                            parsed = True
                            break
                        except TypeError:
                            pass

                assert parsed, 'Invalid value "%s" received for %s' %(from_settings, self.key)

        return val

    @staticmethod
    def type_comparator(a, b):
        '''
        Types precede strings, which precede ``None``
        :param a: First element
        :param b: Second element
        :return: ``-1``, ``0``, or ``1``, depending on outcome of comparison
        '''
        if isinstance(a, type) and not isinstance(b, type):
            return -1
        elif not isinstance(a, type) and isinstance(b, type):
            return 1
        elif isinstance(a, str) and not isinstance(b, str):
            return -1
        elif isinstance(b, str) and not isinstance(a, str):
            return 1
        else:
            return 0





SYN_SEM_NET_KWARGS = [

    # Global hyperparams
    Kwarg(
        'outdir',
        './synsemnet_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),

    # Optimization hyperparams
    Kwarg(
        'optim_name',
        'Adam',
        [str, None],
        """Name of the optimizer to use. Must be one of:
    
            - ``'SGD'``
            - ``'Momentum'``
            - ``'AdaGrad'``
            - ``'AdaDelta'``
            - ``'Adam'``
            - ``'FTRL'``
            - ``'RMSProp'``
            - ``'Nadam'``
            - ``None`` (DTSRBayes only; uses the default optimizer defined by Edward, which currently includes steep learning rate decay and is therefore not recommended in the general case)"""
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
    ),
    Kwarg(
        'epsilon',
        1e-3,
        float,
        "Epsilon to avoid boundary violations."
    ),
    Kwarg(
        'optim_epsilon',
        1e-8,
        float,
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.001,
        float,
        "Initial value for the learning rate."
    ),
    Kwarg(
        'learning_rate_min',
        0.,
        float,
        "Minimum value for the learning rate."
    ),
    Kwarg(
        'lr_decay_family',
        None,
        [str, None],
        "Functional family for the learning rate decay schedule (no decay if ``None``)."
    ),
    Kwarg(
        'lr_decay_rate',
        1.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_iteration_power',
        1,
        float,
        "Power to which the iteration number ``t`` should be raised when computing the learning rate decay."
    ),
    Kwarg(
        'lr_decay_steps',
        1,
        int,
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        bool,
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'ema_decay',
        None,
        [float, None],
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
    Kwarg(
        'minibatch_size',
        128,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        100000,
        [int, None],
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),
    Kwarg(
        'n_pretrain_steps',
        0,
        int,
        "Number of steps (minibatches if **streaming** is ``True``, otherwise iterations) during which to pre-train the decoder without backpropagating into the encoder."
    ),

    # Checkpoint settings
    Kwarg(
        'save_freq',
        1000,
        int,
        "Frequency with which to save model checkpoints. If **streaming**, frequency is in minibatches and model is saved after each iteration; otherwise it's in iterations."
    ),
    Kwarg(
        'eval_freq',
        1000,
        int,
        "Frequency with which to evaluate model. If **streaming**, frequency is in minibatches and model is evaluated after each iteration; otherwise it's in iterations."
    ),
    Kwarg(
        'log_freq',
        1000,
        int,
        "Frequency with which to log summary data. If **streaming**, frequency is in minibatches and data is logged after each iteration; otherwise it's in iterations."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard."
    ),

    # Data settings
    Kwarg(
        'os',
        False,
        bool,
        "Whether to use data containing sequence boundary tokens (``'-BOS-'``, ``'-EOS-'``)."
    ),
    Kwarg(
        'root',
        False,
        bool,
        "Whether to use data containing a designated ``ROOT`` depth label. Incompatible with **factor_parse_labels** because depth can no longer be treated as numeric."
    ),
    Kwarg(
        'factor_parse_labels',
        True,
        bool,
        "Whether to factor parse labels into their (numeric) depth and (categorical) ancestor components and predict each separately. If ``False``, depth and category information is merged and treated as atomic.",
    ),

    # MODEL SETTINGS
    # Encoder settings
    Kwarg(
        'word_emb_dim',
        None,
        [int, None],
        "Dimensionality of vocabulary embedding layer. If ``None`` or ``0``, no vocabulary embedding used."
    ),
    Kwarg(
        'bidirectional_encoder',
        False,
        bool,
        "Use bi-directional encoder."
    ),
    Kwarg(
        'project_word_embeddings',
        True,
        bool,
        "Whether to pass word embeddings through a linear projection."
    ),
    Kwarg(
        'project_encodings',
        True,
        bool,
        "Whether to pass encodings through a linear projection."
    ),
    Kwarg(
        'character_embedding_dim',
        None,
        [int, None],
        "Dimensionality of character embedding layer. If ``None`` or ``0``, no character embedding used."
    ),
    Kwarg(
        'n_layers_encoder',
        None,
        [int, None],
        "Number of layers to use for encoder. If ``None``, inferred from length of **n_units_encoder**."
    ),
    Kwarg(
        'n_units_encoder',
        None,
        [int, str, None],
        "Number of units to use in non-final encoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** - 1 space-delimited integers, one for each layer in order from bottom to top. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'encoder_activation',
        'tanh',
        [str, None],
        "Name of activation to use at the output of the encoders.",
    ),
    Kwarg(
        'encoder_recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for recurrent gates.",
    ),
    Kwarg(
        'encoder_projection_activation_inner',
        'elu',
        [str, None],
        "Name of activation to use for prefinal layers in projection function of encoder.",
    ),
    Kwarg(
        'encoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal encoder layers as residual layers with **encoder_resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),

    # Parsing decoder settings
    Kwarg(
        'n_layers_parsing_decoder',
        None,
        [int, None],
        "Number of layers to use for parsing decoder. If ``None``, inferred from length of **parsing_n_units_decoder**.",
        aliases=['n_layers_decoder']
    ),
    Kwarg(
        'n_units_parsing_decoder',
        None,
        [int, str, None],
        "Number of units to use in parsing decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **parsing_n_layers_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value.",
        aliases=['n_units_decoder']
    ),
    Kwarg(
        'parsing_decoder_activation',
        None,
        [str, None],
        "Name of activation to use at the output of the parsing decoder.",
    ),
    Kwarg(
        'parsing_decoder_recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for parsing decoder recurrent gates. Ignored unless parsing decoder is recurrent.",
    ),
    Kwarg(
        'parsing_decoder_activation_inner',
        'tanh',
        [str, None],
        "Name of activation to use for prefinal layers in the parsing decoder.",
    ),
    Kwarg(
        'parsing_decoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal parsing decoder layers as residual layers with **parsing_decoder_resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),

    # STS decoder settings
    Kwarg(
        'sts_decoder_type',
        'cnn',
        str,
        "STS decoder network type to use. One of ``['cnn', 'rnn']``.",
    ),
    Kwarg(
        'n_layers_sts_decoder',
        None,
        [int, None],
        "Number of layers to use for STS decoder. If ``None``, inferred from length of **sts_n_units_decoder**.",
        aliases=['n_layers_decoder']
    ),
    Kwarg(
        'n_units_sts_decoder',
        300,
        [int, str, None],
        "Number of units to use in STS decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **sts_n_layers_decoder**  space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value.",
        aliases=['n_units_decoder']
    ),
    Kwarg(
        'sts_conv_kernel_size',
        1,
        [int, str],
        "Size of kernel to use in convolutional STS decoder layers. Can be an ``int``, which will be used for all layers, or a ``str`` with **sts_n_layers_decoder**  space-delimited integers, one for each layer in order from top to bottom. Ignored unless **sts_decoder_type** is ``cnn``.",
        aliases=['conv_kernel_size']
    ),
    Kwarg(
        'bidirectional_sts_decoder',
        True,
        bool,
        "Use bi-directional STS decoder. Ignored unless **sts_decoder_type** is ``rnn``."
    ),
    Kwarg(
        'sts_decoder_activation',
        None,
        [str, None],
        "Name of activation to use at the output of the STS decoder.",
    ),
    Kwarg(
        'sts_decoder_recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for STS decoder recurrent gates. Ignored unless **sts_decoder_type** is ``rnn``.",
    ),
    Kwarg(
        'sts_decoder_activation_inner',
        'tanh',
        [str, None],
        "Name of activation to use for prefinal layers in the STS decoder.",
    ),
    Kwarg(
        'sts_projection_activation_inner',
        'elu',
        [str, None],
        "Name of activation to use for prefinal layers in projection function of STS decoder.",
    ),
    Kwarg(
        'project_sts_decodings',
        False,
        bool,
        "Whether to apply a linear projection at the output of the STS decoder. Ignored unless **sts_decoder_type** is ``rnn``."
    ),
    Kwarg(
        'sts_decoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal STS decoder layers as residual layers with **sts_decoder_resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),
    Kwarg(
        'n_layers_sts_classifier',
        None,
        [int, None],
        "Number of layers to use for STS classifier. If ``None``, inferred from length of **sts_n_units_classifier**.",
        aliases=['n_layers_decoder']
    ),
    Kwarg(
        'n_units_sts_classifier',
        300,
        [int, str, None],
        "Number of units to use in STS classifier layers. Can be an ``int``, which will be used for all layers, a ``str`` with **sts_n_layers_classifier** - 1 space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value.",
        aliases=['n_units_decoder']
    ),
    Kwarg(
        'sts_classifier_activation_inner',
        'tanh',
        [str, None],
        "Name of activation to use for prefinal layers in the STS classifier.",
    ),
    Kwarg(
        'sts_classifier_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal STS classifier layers as residual layers with **sts_classifier_resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),

    # Loss settings
    Kwarg(
        'parsing_loss_scale',
        1,
        [float, None],
        "Weight on parsing loss.",
        aliases=['loss_scale']
    ),
    Kwarg(
        'parsing_adversarial_loss_scale',
        1,
        [float, None],
        "Weight on adversarial parsing loss.",
        aliases=['adversarial_loss_scale']
    ),
    Kwarg(
        'sts_loss_scale',
        1,
        [float, None],
        "Weight on STS loss.",
        aliases=['loss_scale']
    ),
    Kwarg(
        'sts_adversarial_loss_scale',
        1,
        [float, None],
        "Weight on adversarial STS loss.",
        aliases=['adversarial_loss_scale']
    ),

    # Numeric settings
    Kwarg(
        'float_type',
        'float32',
        str,
        "``float`` type to use throughout the network."
    ),
    Kwarg(
        'int_type',
        'int32',
        str,
        "``int`` type to use throughout the network (used for tensor slicing)."
    ),
]


def synsemnet_kwarg_docstring():
    out = "**SynSemNet:**\n\n"

    for kwarg in SYN_SEM_NET_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    return out
