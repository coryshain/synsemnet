import sys
import os
import time
import pickle
import numpy as np
import tensorflow as tf

from .kwargs import SYN_SEM_NET_KWARGS
from .backend import *
from .data import get_evalb_scores, padded_concat
from .util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

class SynSemNet(object):

    ############################################################
    # Initialization methods
    ############################################################

    _INITIALIZATION_KWARGS = SYN_SEM_NET_KWARGS

    _doc_header = """
        Class implementing a SynSemNet.

    """
    _doc_args = "\n        :param vocab: ``list``; list of vocabulary items. Items outside this list will be treated as <unk>.\n"
    _doc_args += "        :param charset: ``list`` of characters or ``None``; Characters to use in character-level encoder. If ``None``, no character-level representations will be used.\n"
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for
                             x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    TEXT_TYPES = ['parsing', 'sts_s1', 'sts_s2']
    TASK_TYPES = ['parsing', 'wp', 'sts', 'bow']
    REP_TYPES = ['syn', 'sem']
    GRADIENT_TYPES = ['standard', 'adversarial', 'frozen']
    ENCODING_LEVELS = ['word_embeddings', 'word_encodings', 'sent_encoding']

    # def __init__(self, char_set=['a'], pos_label_set=['a'], parse_label_set=['a'], sts_label_set=['a'], **kwargs):
    def __init__(self, char_set, pos_label_set, parse_label_set, parse_depth_set, sts_label_set, **kwargs):
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self.char_set = char_set
        self.pos_label_set = pos_label_set
        self.parse_label_set = parse_label_set
        self.parse_depth_set = parse_depth_set
        self.sts_label_set = sts_label_set

        self._initialize_session()
        self._initialize_metadata()
        self.build()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.UINT_TF = getattr(np, 'u' + self.int_type)
        self.UINT_NP = getattr(tf, 'u' + self.int_type)
        self.regularizer_losses = []

        self.n_char = len(self.char_set)
        self.n_pos = len(self.pos_label_set)
        self.n_parse_label = len(self.parse_label_set)
        self.n_sts_label = len(self.sts_label_set)
        self.parse_depth_min = min(*self.parse_depth_set)
        self.parse_depth_max = max(*self.parse_depth_set)
        self.n_parse_depth = self.parse_depth_max - self.parse_depth_min + 1

        # Encoder layers and units
        assert not self.n_units_encoder is None, 'You must provide a value for **n_units_encoder** when initializing a SynSemNet model.'
        if isinstance(self.n_units_encoder, str):
            self.units_encoder = [int(x) for x in self.n_units_encoder.split()]
        elif isinstance(self.n_units_encoder, int):
            if self.n_layers_encoder is None:
                self.units_encoder = [self.n_units_encoder]
            else:
                self.units_encoder = [self.n_units_encoder] * self.n_layers_encoder
        else:
            self.units_encoder = self.n_units_encoder

        if self.n_layers_encoder is None:
            self.layers_encoder = len(self.units_encoder)
        else:
            self.layers_encoder = self.n_layers_encoder
        if len(self.units_encoder) == 1:
            self.units_encoder = [self.units_encoder[0]] * self.layers_encoder

        assert len(self.units_encoder) == self.layers_encoder, 'Misalignment in number of layers between n_layers_encoder and n_units_encoder.'

        # Parsing classifier layers and units
        if self.n_units_parsing_classifier is None:
            self.units_parsing_classifier = []
        elif isinstance(self.n_units_parsing_classifier, str):
            self.units_parsing_classifier = [int(x) for x in self.n_units_parsing_classifier.split()]
        elif isinstance(self.n_units_parsing_classifier, int):
            if self.n_layers_parsing_classifier is None:
                self.units_parsing_classifier = [self.n_units_parsing_classifier]
            else:
                self.units_parsing_classifier = [self.n_units_parsing_classifier] * self.n_layers_parsing_classifier
        else:
            self.units_parsing_classifier = self.n_units_parsing_classifier

        if self.n_layers_parsing_classifier is None:
            self.layers_parsing_classifier = len(self.units_parsing_classifier)
        else:
            self.layers_parsing_classifier = self.n_layers_parsing_classifier
        if len(self.units_parsing_classifier) == 1:
            self.units_parsing_classifier = [self.units_parsing_classifier[0]] * self.layers_parsing_classifier

        assert len(self.units_parsing_classifier) == self.layers_parsing_classifier, 'Misalignment in number of layers between n_layers_parsing_classifier and n_units_parsing_classifier.'

        # WP decoder layers and units

        self.wp_recurrent_decoder = self.n_units_wp_decoder is not None
        if self.n_units_wp_decoder is None:
            self.units_wp_decoder = []
        elif isinstance(self.n_units_wp_decoder, str):
            self.units_wp_decoder = [int(x) for x in self.n_units_wp_decoder.split()]
        elif isinstance(self.n_units_wp_decoder, int):
            if self.n_layers_wp_decoder is None:
                self.units_wp_decoder = [self.n_units_wp_decoder]
            else:
                self.units_wp_decoder = [self.n_units_wp_decoder] * self.n_layers_wp_decoder
        else:
            self.units_wp_decoder = self.n_units_wp_decoder

        if self.n_layers_wp_decoder is None:
            self.layers_wp_decoder = len(self.units_wp_decoder)
        else:
            self.layers_wp_decoder = self.n_layers_wp_decoder
        if len(self.units_wp_decoder) == 1:
            self.units_wp_decoder = [self.units_wp_decoder[0]] * self.layers_wp_decoder

        assert len(self.units_wp_decoder) == self.layers_wp_decoder, 'Misalignment in number of layers between n_layers_wp_decoder and n_units_wp_decoder.'

        # WP classifier layers and units
        if self.n_units_wp_classifier is None:
            self.units_wp_classifier = []
        elif isinstance(self.n_units_wp_classifier, str):
            self.units_wp_classifier = [int(x) for x in self.n_units_wp_classifier.split()]
        elif isinstance(self.n_units_wp_classifier, int):
            if self.n_layers_wp_classifier is None:
                self.units_wp_classifier = [self.n_units_wp_classifier]
            else:
                self.units_wp_classifier = [self.n_units_wp_classifier] * self.n_layers_wp_classifier
        else:
            self.units_wp_classifier = self.n_units_wp_classifier

        if self.n_layers_wp_classifier is None:
            self.layers_wp_classifier = len(self.units_wp_classifier)
        else:
            self.layers_wp_classifier = self.n_layers_wp_classifier
        if len(self.units_wp_classifier) == 1:
            self.units_wp_classifier = [self.units_wp_classifier[0]] * self.layers_wp_classifier

        assert len(self.units_wp_classifier) == self.layers_wp_classifier, 'Misalignment in number of layers between n_layers_wp_classifier and n_units_wp_classifier.'

        # STS decoder layers and units
        self.use_sts_decoder = True
        if self.n_units_sts_decoder is None:
            self.units_sts_decoder = []
            self.use_sts_decoder = False
        elif isinstance(self.n_units_sts_decoder, str):
            self.units_sts_decoder = [int(x) for x in self.n_units_sts_decoder.split()]
        elif isinstance(self.n_units_sts_decoder, int):
            if self.n_layers_sts_decoder is None:
                self.units_sts_decoder = [self.n_units_sts_decoder]
            else:
                self.units_sts_decoder = [self.n_units_sts_decoder] * self.n_layers_sts_decoder
        else:
            self.units_sts_decoder = self.n_units_sts_decoder

        if self.n_layers_sts_decoder is None:
            self.layers_sts_decoder = len(self.units_sts_decoder)
        else:
            self.layers_sts_decoder = self.n_layers_sts_decoder
        if len(self.units_sts_decoder) == 1:
            self.units_sts_decoder = [self.units_sts_decoder[0]] * self.layers_sts_decoder

        if isinstance(self.sts_conv_kernel_size, str):
            self.sts_kernel_size = [int(x) for x in self.sts_conv_kernel_size.split()]
        elif isinstance(self.sts_conv_kernel_size, int):
            self.sts_kernel_size  = [self.sts_conv_kernel_size] * self.layers_sts_decoder
        else:
            self.sts_kernel_size = self.sts_conv_kernel_size

        if self.use_sts_decoder:
            assert len(self.units_sts_decoder) == len(self.sts_kernel_size) == self.layers_sts_decoder, 'Misalignment in number of layers between n_layers_sts_decoder, sts_conv_kernel_size, and n_units_sts_decoder.'

        # STS classifier layers and units
        if self.n_units_sts_classifier is None:
            self.units_sts_classifier = []
        elif isinstance(self.n_units_sts_classifier, str):
            self.units_sts_classifier = [int(x) for x in self.n_units_sts_classifier.split()]
        elif isinstance(self.n_units_sts_classifier, int):
            if self.n_layers_sts_classifier is None:
                self.units_sts_classifier = [self.n_units_sts_classifier]
            else:
                self.units_sts_classifier = [self.n_units_sts_classifier] * self.n_layers_sts_classifier
        else:
            self.units_sts_classifier = self.n_units_sts_classifier

        if self.n_layers_sts_classifier is None:
            self.layers_sts_classifier = len(self.units_sts_classifier)
        else:
            self.layers_sts_classifier = self.n_layers_sts_classifier
        if len(self.units_sts_classifier) == 1:
            self.units_sts_classifier = [self.units_sts_classifier[0]] * self.layers_sts_classifier

        assert len(self.units_sts_classifier) == self.layers_sts_classifier, 'Misalignment in number of layers between n_layers_sts_classifier and n_units_sts_classifier.'

        # BOW classifier layers and units
        if self.n_units_bow_classifier is None:
            self.units_bow_classifier = []
        elif isinstance(self.n_units_bow_classifier, str):
            self.units_bow_classifier = [int(x) for x in self.n_units_bow_classifier.split()]
        elif isinstance(self.n_units_bow_classifier, int):
            if self.n_layers_bow_classifier is None:
                self.units_bow_classifier = [self.n_units_bow_classifier]
            else:
                self.units_bow_classifier = [self.n_units_bow_classifier] * self.n_layers_bow_classifier
        else:
            self.units_bow_classifier = self.n_units_bow_classifier

        if self.n_layers_bow_classifier is None:
            self.layers_bow_classifier = len(self.units_bow_classifier)
        else:
            self.layers_bow_classifier = self.n_layers_bow_classifier
        if len(self.units_bow_classifier) == 1:
            self.units_bow_classifier = [self.units_bow_classifier[0]] * self.layers_bow_classifier

        assert len(
            self.units_bow_classifier) == self.layers_bow_classifier, 'Misalignment in number of layers between n_layers_bow_classifier and n_units_bow_classifier.'
        
        self.predict_mode = False

    def _pack_metadata(self):
        md = {}
        md['char_set'] = self.char_set
        md['pos_label_set'] = self.pos_label_set
        md['parse_label_set'] = self.parse_label_set
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.char_set = md.get('char_set')
        self.pos_label_set = md.get('pos_label_set')
        self.parse_label_set = md.get('parse_label_set')
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def __getstate__(self):
        return self._pack_metadata()

    def __setstate__(self, state):
        self._unpack_metadata(state)
        self._initialize_session()
        self._initialize_metadata()




    ############################################################
    # Private model construction methods
    ############################################################

    def build(self, outdir=None, restore=True, verbose=True):
        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './synsemnet_model/'

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_inputs()

                #############################################################
                #
                # ENCODERS
                #
                #############################################################

                for s in self.REP_TYPES:
                    name = 'char_rnn_%s' % s
                    val = self._initialize_rnn_module(
                        1,
                        [self.word_emb_dim],
                        activation=self.encoder_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        bidirectional=self.bidirectional_encoder,
                        project_encodings=self.project_word_embeddings,
                        projection_activation_inner=self.encoder_projection_activation_inner,
                        resnet_n_layers_inner=self.encoder_resnet_n_layers_inner,
                        return_sequences=False,
                        name=name
                    )
                    setattr(self, name, val)

                    name = 'word_encoder_%s' % s
                    val = self._initialize_rnn_module(
                        self.layers_encoder,
                        self.units_encoder,
                        activation=self.encoder_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        bidirectional=self.bidirectional_encoder,
                        project_encodings=self.project_encodings,
                        projection_activation_inner=self.encoder_projection_activation_inner,
                        resnet_n_layers_inner=self.encoder_resnet_n_layers_inner,
                        return_sequences=True,
                        name=name
                    )
                    setattr(self, name, val)


                #############################################################
                #
                # ENCODINGS
                #
                #############################################################

                # Parsing encodings
                for text in self.TEXT_TYPES:
                    for enc in self.ENCODING_LEVELS:
                        for s in self.REP_TYPES:
                            for grad in self.GRADIENT_TYPES:
                                name = '%s_%s_%s' % (text, enc, s)
                                name_base = name
                                if grad == 'adversarial':
                                    name += '_adversarial'
                                    val = replace_gradient(
                                        tf.identity,
                                        lambda x: -x * self.adversarial_gradient_scale,
                                        session=self.sess
                                    )(getattr(self, name_base))
                                elif grad == 'frozen':
                                    name += '_frozen'
                                    val = tf.stop_gradient(getattr(self, name_base))
                                else: # grad == 'standard
                                    if enc == 'word_embeddings':
                                        char_name = '%s_char_embeddings_%s'% (text, s)
                                        rnn_name = 'char_rnn_%s' % s
                                        mask_name = '%s_char_mask' % text
                                        val = self._initialize_word_embeddings(
                                            getattr(self, char_name),
                                            getattr(self, rnn_name),
                                            character_mask=getattr(self, mask_name)
                                        )
                                    elif enc == 'word_encodings':
                                        emb_name = '%s_word_embeddings_%s' % (text, s)
                                        rnn_name = 'word_encoder_%s' % s
                                        mask_name = '%s_word_mask' % text
                                        val = self._initialize_encoding(
                                            getattr(self, emb_name),
                                            getattr(self, rnn_name),
                                            mask=getattr(self, mask_name)
                                        )
                                    elif enc == 'sent_encoding':
                                        word_enc_name = '%s_word_encodings_%s' % (text, s)
                                        mask_name = '%s_word_mask' % text
                                        method = self.sentence_aggregation
                                        val = self._aggregate_words(
                                            getattr(self, word_enc_name),
                                            mask=getattr(self, mask_name),
                                            method=method
                                        )
                                    else:
                                        raise ValueError('Unrecognized encoding type "%s".' % enc)
                                setattr(self, name, val)


                #############################################################
                #
                # OUTPUTS
                #
                #############################################################

                for task in self.TASK_TYPES:
                    for s in self.REP_TYPES:
                        module_name = '%s_%s' % (task, s)
                        if task == 'parsing':
                            word_enc_name = 'parsing_word_encodings_%s' % s
                            if s == 'sem':
                                word_enc_name += '_adversarial'
                            o = self._initialize_parsing_outputs(
                                getattr(self, word_enc_name),
                                residual_parser=self.residual_parser,
                                name=module_name
                            )
                            for l in ['logit', 'prediction']:
                                for v in ['pos_label', 'parse_label', 'parse_depth']:
                                    key = '%s_%s_%s' % (v, l, s)
                                    val = o['%s_%s' % (v, l)]
                                    setattr(self, key, val)
                        elif task == 'wp':
                            if self.wp_recurrent_decoder:
                                module = self._initialize_wp_decoder(
                                    teacher_forcing=self.wp_decoder_teacher_forcing,
                                    pe_type=self.wp_decoder_positional_encoding_type,
                                    pe_n_units=self.wp_decoder_positional_encoding_units,
                                    name=module_name
                                )
                                wp_mode = 'decoder'
                                clip = None
                            else:
                                module = self._initialize_wp_classifier(name=module_name)
                                wp_mode = 'classifier'
                                clip =  self.wp_n_pos
                            setattr(self, 'wp_module_%s' % s, module)
                            for text in ['parsing', 'sts_s1', 'sts_s2']:
                                sent_enc_name = '%s_sent_encoding_%s' % (text, s)
                                word_1h_name = '%s_words_one_hot' % text
                                word_emb_name = '%s_word_embeddings_%s' % (text, s)
                                mask_name = '%s_word_mask' % text
                                if s == 'sem':
                                    sent_enc_name += '_adversarial'
                                    word_emb_name += '_adversarial'
                                o = self._initialize_wp_outputs(
                                    module,
                                    getattr(self, sent_enc_name),
                                    getattr(self, word_1h_name),
                                    getattr(self, word_emb_name),
                                    mask=getattr(self, mask_name),
                                    clip=clip,
                                    reverse=self.wp_decoder_reverse_targets,
                                    mode=wp_mode
                                )
                                for l in ['logit', 'prediction']:
                                    setattr(self, 'wp_%s_%s_%s' % (text, l, s), o[l])
                        elif task == 'sts':
                            s1_enc_name = 'sts_s1_word_encodings_%s' % s
                            s2_enc_name = 'sts_s2_word_encodings_%s' % s
                            if s == 'syn':
                                s1_enc_name += '_adversarial'
                                s2_enc_name += '_adversarial'
                            s1 = getattr(self, s1_enc_name)
                            s2 = getattr(self, s2_enc_name)
                            if self.use_sts_decoder:
                                sts_decoder = self._initialize_sts_decoder(name='module_name')
                                if self.sts_decoder_type.lower() == 'rnn':
                                    s1 = sts_decoder(s1, mask=self.sts_s1_word_mask)
                                    s2 = sts_decoder(s2, mask=self.sts_s2_word_mask)
                                elif self.sts_decoder_type.lower() == 'cnn':
                                    s1 = sts_decoder(s1)
                                    s2 = sts_decoder(s2)
                                else:
                                    raise ValueError('Unrecognized sts_decoder_type "%s".' % self.sts_decoder_type)
                            else:
                                s1 = self._aggregate_words(
                                    s1,
                                    mask=self.sts_s1_word_mask,
                                    method=self.sentence_aggregation
                                )
                                s2 = self._aggregate_words(
                                    s2,
                                    mask=self.sts_s2_word_mask,
                                    method=self.sentence_aggregation
                                )
                            o = self._initialize_sts_outputs(
                                s1,
                                s2,
                                use_classifier=self.use_sts_classifier,
                                name=module_name
                            )
                            for l in ['logit', 'prediction']:
                                setattr(self, 'sts_%s_%s' % (l, s), o[l])
                        elif task == 'bow':
                            # TODO
                            pass
                        else:
                            raise ValueError('Unrecognized task "%s".' % task)


                #############################################################
                #
                # LOSSES
                #
                #############################################################

                # Parsing losses
                for task in self.TASK_TYPES:
                    for s in self.REP_TYPES:
                        if task == 'parsing':
                            if s == 'syn':
                                scale = self.parsing_loss_scale
                            else:
                                scale = self.parsing_adversarial_loss_scale
                            pos_label_logit_name = 'pos_label_logit_%s' % s
                            parse_label_logit_name = 'parse_label_logit_%s' % s
                            parse_depth_logit_name = 'parse_depth_logit_%s' % s
                            o = self._initialize_parsing_objective(
                                getattr(self, pos_label_logit_name),
                                getattr(self, parse_label_logit_name),
                                getattr(self, parse_depth_logit_name),
                                weights=self.parsing_word_mask,
                                well_formedness_loss_scale=self.well_formedness_loss_scale,
                                nonzero_scale=scale
                            )
                            for l in [
                                'loss',
                                'pos_label_loss',
                                'parse_label_loss',
                                'parse_depth_loss',
                                'zero_sum_loss',
                                'no_neg_loss'
                            ]:
                                if l == 'loss':
                                    key = 'parsing_loss_%s' % s
                                else:
                                    key = '%s_%s' % (l, s)
                                val = o[l]
                                setattr(self, key, val)
                        elif task == 'wp':
                            if s == 'syn':
                                scale = self.wp_loss_scale
                            else:
                                scale = self.wp_adversarial_loss_scale
                            for text in self.TEXT_TYPES:
                                wp_logit_name = 'wp_%s_logit_%s' % (text, s)
                                wp_target_name = '%s_words' % text
                                weights_name = '%s_word_mask' % text
                                o = self._initialize_wp_objective(
                                    getattr(self, wp_logit_name),
                                    getattr(self, wp_target_name),
                                    mode=wp_mode,
                                    weights=getattr(self, weights_name),
                                    nonzero_scale=scale
                                )
                                for l in ['loss', 'acc', 'n']:
                                    key = 'wp_%s_%s_%s' % (text, l, s)
                                    val = o[l]
                                    setattr(self, key, val)
                            setattr(
                                self,
                                'wp_loss_%s' % s,
                                sum([getattr(self, 'wp_%s_loss_%s' % (text, s)) for text in self.TEXT_TYPES])
                            )
                        elif task == 'sts':
                            if s == 'sem':
                                scale = self.sts_loss_scale
                            else:
                                scale = self.sts_adversarial_loss_scale
                            sts_logit_name = 'sts_logit_%s' % s
                            o = self._initialize_sts_objective(
                                getattr(self, sts_logit_name),
                                nonzero_scale=scale
                            )
                            setattr(self, 'sts_loss_%s' % s, o['loss'])
                        elif task == 'bow':
                            # TODO
                            self.bow_loss_syn = tf.convert_to_tensor(0.)
                            self.bow_loss_sem = tf.convert_to_tensor(0.)
                        else:
                            raise ValueError('Unrecognized task "%s".' % task)


                self.loss = self.parsing_loss_syn * self.parsing_loss_scale + \
                            self.wp_loss_syn * self.wp_loss_scale + \
                            self.sts_loss_sem * self.sts_loss_scale + \
                            self.bow_loss_sem * self.bow_loss_scale
                self.adversarial_loss = self.parsing_loss_sem * self.parsing_adversarial_loss_scale + \
                                        self.wp_loss_sem * self.wp_adversarial_loss_scale + \
                                        self.sts_loss_syn * self.sts_adversarial_loss_scale + \
                                        self.bow_loss_syn * self.bow_adversarial_loss_scale
                
                self.total_loss = self.loss + self.adversarial_loss

        self._initialize_train_op()
        self._initialize_ema()
        self._initialize_saver()
        self._initialize_logging()

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
        self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.training = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='training')
                self.task = tf.placeholder(self.INT_TF, shape=[None], name='task')

                if self.character_embedding_dim:
                    self.char_embedding_matrix_syn = tf.get_variable(
                        shape=[self.n_char + 1, self.character_embedding_dim],
                        dtype=self.FLOAT_TF,
                        initializer=get_initializer('he_normal_initializer', session=self.sess),
                        name='char_embedding_matrix_syn'
                    )
                    self.char_embedding_matrix_sem = tf.get_variable(
                        shape=[self.n_char + 1, self.character_embedding_dim],
                        dtype=self.FLOAT_TF,
                        initializer=get_initializer('he_normal_initializer', session=self.sess),
                        name='char_embedding_matrix_sem'
                    )

                self._initialize_parsing_inputs()
                self._initialize_sts_inputs()

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

    def _initialize_parsing_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.parsing_chars = tf.placeholder(self.INT_TF, shape=[None, None, None], name='parsing_chars')
                self.parsing_char_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='parsing_char_mask')
                self.parsing_words = tf.placeholder(self.INT_TF, shape=[None, None], name='parsing_words')
                self.parsing_words_one_hot = tf.one_hot(self.parsing_words, self.target_vocab_size)
                self.parsing_bow_true = tf.reduce_sum(self.parsing_words_one_hot, axis=-2)
                self.parsing_word_mask = tf.cast(tf.reduce_any(self.parsing_char_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                if self.character_embedding_dim:
                    self.parsing_char_embeddings_syn = tf.gather(self.char_embedding_matrix_syn, self.parsing_chars)
                    self.parsing_char_embeddings_sem = tf.gather(self.char_embedding_matrix_sem, self.parsing_chars)
                else:
                    self.parsing_char_one_hots = tf.one_hot(
                        self.parsing_chars,
                        self.n_char
                    )
                    self.parsing_char_embeddings_syn = self.parsing_char_one_hots
                    self.parsing_char_embeddings_sem = self.parsing_char_one_hots

                self.pos_label = tf.placeholder(self.INT_TF, shape=[None, None], name='pos_label')

                self.parse_label = tf.placeholder(self.INT_TF, shape=[None, None], name='parse_label')
                if self.factor_parse_labels:
                    if self.parse_depth_loss_type.lower() == 'mse':
                        self.parse_depth_src = tf.placeholder(self.FLOAT_TF, shape=[None, None], name='parse_depth')
                        self.parse_depth = self.parse_depth_src
                    elif self.parse_depth_loss_type.lower() == 'xent':
                        self.parse_depth_src = tf.placeholder(self.INT_TF, shape=[None, None], name='parse_depth')
                        parse_depth = self.parse_depth_src - self.parse_depth_min
                        self.parse_depth = tf.clip_by_value(parse_depth, 0, self.INT_TF.max)
                    else:
                        raise ValueError('Unrecognized value for parse_depth_loss_type: "%s".' % self.parse_depth_loss_type)


    def _initialize_sts_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.sts_s1_chars = tf.placeholder(self.INT_TF, shape=[None, None, None], name='sts_s1_chars')
                self.sts_s1_char_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='sts_s1_char_mask')
                self.sts_s1_words = tf.placeholder(self.INT_TF, shape=[None, None], name='sts_s1_words')
                self.sts_s1_words_one_hot = tf.one_hot(self.sts_s1_words, self.target_vocab_size)
                self.sts_s1_bow_true = tf.reduce_sum(self.sts_s1_words_one_hot, axis=-2)
                self.sts_s1_word_mask = tf.cast(tf.reduce_any(self.sts_s1_char_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                if self.character_embedding_dim:
                    self.sts_s1_char_embeddings_syn = tf.gather(self.char_embedding_matrix_syn, self.sts_s1_chars)
                    self.sts_s1_char_embeddings_sem = tf.gather(self.char_embedding_matrix_sem, self.sts_s1_chars)
                else:
                    self.sts_s1_char_one_hots = tf.one_hot(
                        self.sts_s1_chars,
                        self.n_char
                    )
                    self.sts_s1_char_embeddings_syn = self.sts_s1_char_one_hots
                    self.sts_s1_char_embeddings_sem = self.sts_s1_char_one_hots

                self.sts_s2_chars = tf.placeholder(self.INT_TF, shape=[None, None, None], name='sts_s2_chars')
                self.sts_s2_char_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='sts_s2_char_mask')
                self.sts_s2_words = tf.placeholder(self.INT_TF, shape=[None, None], name='sts_s2_words')
                self.sts_s2_words_one_hot = tf.one_hot(self.sts_s2_words, self.target_vocab_size)
                self.sts_s2_bow_true = tf.reduce_sum(self.sts_s2_words_one_hot, axis=-2)
                self.sts_s2_word_mask = tf.cast(tf.reduce_any(self.sts_s2_char_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                if self.character_embedding_dim:
                    self.sts_s2_char_embeddings_syn = tf.gather(self.char_embedding_matrix_syn, self.sts_s2_chars)
                    self.sts_s2_char_embeddings_sem = tf.gather(self.char_embedding_matrix_sem, self.sts_s2_chars)
                else:
                    self.sts_s2_char_one_hots = tf.one_hot(
                        self.sts_s2_chars,
                        self.n_char
                    )
                    self.sts_s2_char_embeddings_syn = self.sts_s2_char_one_hots
                    self.sts_s2_char_embeddings_sem = self.sts_s2_char_one_hots

                if self.sts_loss_type.lower() == 'mse':
                    self.sts_label = tf.placeholder(self.FLOAT_TF, shape=[None], name='sts_label')
                elif self.sts_loss_type.lower() == 'xent':
                    self.sts_label = tf.placeholder(self.INT_TF, shape=[None], name='sts_label')
                else:
                    raise ValueError('Unrecognized sts_loss_type "%s".' % self.sts_loss_type)

    def _initialize_dense_module(
            self,
            n_layers,
            n_units,
            kernel_initializer='he_normal_initializer',
            activation=None,
            activation_inner='elu',
            resnet_n_layers_inner=None,
            reuse=None,
            name='word_dense'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if reuse is None:
                    reuse = tf.AUTO_REUSE
                out = []
                for l in range(n_layers):
                    activation_inner_cur = activation_inner
                    if l < n_layers - 1:
                        activation_cur = activation_inner
                    else:
                        activation_cur = activation

                    units_cur = n_units[l]
                    
                    if resnet_n_layers_inner:
                        word_encoder_dense = DenseResidualLayer(
                            training=self.training,
                            units=units_cur,
                            layers_inner=resnet_n_layers_inner,
                            kernel_initializer=kernel_initializer,
                            activation_inner=activation_inner_cur,
                            activation=activation_cur,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_l%d' % l,
                            reuse=reuse
                        )
                    else:
                        word_encoder_dense = DenseLayer(
                            training=self.training,
                            units=units_cur,
                            kernel_initializer=kernel_initializer,
                            activation=activation_cur,
                            name=name + '_l%d' % l,
                            reuse=reuse,
                            session=self.sess
                        )

                    out.append(make_lambda(word_encoder_dense, session=self.sess))

                out = compose_lambdas(out)

                return out

    def _initialize_rnn_module(
            self,
            n_layers,
            n_units,
            kernel_initializer='he_normal_initializer',
            recurrent_initializer='orthogonal_initializer',
            activation='tanh',
            activation_inner='tanh',
            recurrent_activation='sigmoid',
            bidirectional=True,
            project_encodings=True,
            projection_activation_inner='elu',
            resnet_n_layers_inner=None,
            return_sequences=True,
            reuse=None,
            passthru=False,
            name='rnn'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if reuse is None:
                    reuse = tf.AUTO_REUSE
                out = []
                for l in range(n_layers):
                    if l < n_layers - 1:
                        activation_cur = activation_inner
                    else:
                        activation_cur = activation

                    if bidirectional:
                        assert n_units[l] % 2 == 0, 'Bidirectional RNNs must have an even number of hidden units. Saw %d.' % n_units[l]
                        units_cur = int(n_units[l] / 2)
                    else:
                        units_cur = n_units[l]

                    fwd_rnn = RNNLayer(
                        training=self.training,
                        units=units_cur,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        activation=activation_cur,
                        recurrent_activation=recurrent_activation,
                        return_sequences=return_sequences,
                        name=name + '_fwd_l%d' % l,
                        session=self.sess
                    )

                    if bidirectional:
                        bwd_rnn = RNNLayer(
                            training=self.training,
                            units=units_cur,
                            kernel_initializer=kernel_initializer,
                            recurrent_initializer=recurrent_initializer,
                            activation=activation_cur,
                            recurrent_activation=recurrent_activation,
                            return_sequences=return_sequences,
                            name=name + '_bwd_l%d' % l,
                            session=self.sess
                        )
                        rnn = make_bi_rnn_layer(fwd_rnn, bwd_rnn, session=self.sess)
                    else:
                        rnn = fwd_rnn
                    if passthru: # Concat inputs and outputs to allow passthru connections
                        def char_encoder_rnn(x, layer=rnn, mask=None):
                            below = x
                            above = layer(x, mask=mask)
                            out = tf.concat([below, above], axis=-1)
                            return out

                    out.append(make_lambda(rnn, session=self.sess, use_kwargs=True))

                if project_encodings:
                    if resnet_n_layers_inner:
                        projection = DenseResidualLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer=kernel_initializer,
                            layers_inner=resnet_n_layers_inner,
                            activation_inner=projection_activation_inner,
                            activation=None,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    else:
                        projection = DenseLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer=kernel_initializer,
                            activation=None,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    out.append(make_lambda(projection, session=self.sess))

                out = compose_lambdas(out)

                return out

    def _initialize_cnn_module(
            self,
            n_layers,
            kernel_size,
            n_units,
            padding='valid',
            kernel_initializer='he_normal_initializer',
            activation='elu',
            activation_inner='elu',
            project_encodings=True,
            resnet_n_layers_inner=None,
            projection_activation_inner='elu',
            max_pooling_over_time=True,
            reuse=None,
            name='word_cnn'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if reuse is None:
                    reuse = tf.AUTO_REUSE
                out = []
                for l in range(n_layers):
                    kernel_size_cur = kernel_size[l]

                    if l < n_layers - 1:
                        activation_cur = activation_inner
                    else:
                        activation_cur = activation

                    units_cur = n_units[l]

                    if resnet_n_layers_inner:
                        word_encoder_cnn = Conv1DResidualLayer(
                            training=self.training,
                            kernel_size=kernel_size_cur,
                            n_filters=units_cur,
                            kernel_initializer=kernel_initializer,
                            activation=activation_cur,
                            padding=padding,
                            layers_inner=resnet_n_layers_inner,
                            project_inputs=False,
                            name=name + '_l%d' % l,
                            reuse=reuse,
                            session=self.sess
                        )
                    else:
                        word_encoder_cnn = Conv1DLayer(
                            training=self.training,
                            kernel_size=kernel_size_cur,
                            n_filters=units_cur,
                            kernel_initializer=kernel_initializer,
                            activation=activation_cur,
                            padding=padding,
                            name=name + '_l%d' % l,
                            reuse=reuse,
                            session=self.sess
                        )

                    out.append(make_lambda(word_encoder_cnn, session=self.sess))

                    if max_pooling_over_time:
                        out.append(make_lambda(lambda x: tf.reduce_max(x, axis=1), session=self.sess))

                if project_encodings:
                    if resnet_n_layers_inner:
                        projection = DenseResidualLayer(
                            training=self.training,
                            units=n_units[-1],
                            layers_inner=resnet_n_layers_inner,
                            kernel_initializer=kernel_initializer,
                            activation_inner=projection_activation_inner,
                            activation=activation,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    else:
                        projection = DenseLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer=kernel_initializer,
                            activation=activation,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    out.append(make_lambda(projection, session=self.sess))

                out = compose_lambdas(out)

                return out

    def _initialize_word_embeddings(self, inputs, encoder, character_mask=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                B = tf.shape(inputs)[0]
                W = tf.shape(inputs)[1]
                C = tf.shape(inputs)[2]
                F = inputs.shape[3]

                inputs_flattened = tf.reshape(inputs, [B * W, C, F])
                character_mask_flattened = tf.reshape(character_mask, [B * W, C])

                word_embedding = encoder(inputs_flattened, mask=character_mask_flattened)

                word_embedding = tf.reshape(word_embedding, [B, W, self.word_emb_dim])

                return word_embedding

    def _initialize_encoding(self, inputs, encoder, mask=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                return encoder(inputs, mask=mask) * mask[..., None]

    def _aggregate_words(self, word_encodings, mask=None, method='final'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if method.lower() == 'average':
                    if mask is None:
                        out = tf.reduce_mean(word_encodings, axis=-2)
                    else:
                        out = tf.reduce_sum(word_encodings, axis=-2) / tf.reduce_sum(mask, axis=-1, keepdims=True)
                elif method.lower() == 'logsumexp':
                    out = reduce_logsumexp(
                        word_encodings,
                        axis=-2,
                        mask=mask,
                        float_type=self.FLOAT_TF,
                        int_type=self.INT_TF,
                        session=self.sess
                    )
                elif method.lower() == 'max':
                    out = reduce_max(
                        word_encodings,
                        axis=-2,
                        mask=mask,
                        float_type=self.FLOAT_TF,
                        int_type=self.INT_TF,
                        session=self.sess
                    )
                elif method.lower() == 'final':
                    out = word_encodings[..., -1, :]
                else:
                    raise ValueError('Unrecognized aggregation method "%s".' % method)

                return out

    def _initialize_parsing_outputs(self, s, residual_parser=True, name=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if name is None:
                    name = 'parsing'

                units = self.n_pos + self.n_parse_label
                if self.factor_parse_labels:
                    if self.parse_depth_loss_type.lower() == 'mse':
                        units += 1
                    elif self.parse_depth_loss_type.lower() == 'xent':
                        units += self.n_parse_depth
                    else:
                        raise ValueError('Unrecognized value for parse_depth_loss_type: "%s".' % self.parse_depth_loss_type)

                parsing_classifier = self._initialize_dense_module(
                    self.layers_parsing_classifier + 1,
                    self.units_parsing_classifier + [units],
                    activation=None,
                    activation_inner=self.parsing_classifier_activation_inner,
                    resnet_n_layers_inner=self.parsing_classifier_resnet_n_layers_inner,
                    name=name + '_classifier'
                )

                parsing_logit = parsing_classifier(s)
                pos_label_logit = parsing_logit[..., :self.n_pos]
                pos_label_prediction = tf.argmax(pos_label_logit, axis=2, output_type=self.INT_TF)
                parse_label_logit = parsing_logit[..., self.n_pos:self.n_pos + self.n_parse_label]
                if residual_parser:
                    parse_label_logit = tf.cumsum(parse_label_logit, axis=1)
                parse_label_prediction = tf.argmax(parse_label_logit, axis=2, output_type=self.INT_TF)
                if self.factor_parse_labels:
                    if self.parse_depth_loss_type.lower() == 'mse':
                        parse_depth_logit = parsing_logit[..., self.n_pos + self.n_parse_label]
                        if residual_parser:
                            parse_depth_logit = tf.cumsum(parse_depth_logit, axis=1)
                        parse_depth_prediction = tf.cast(tf.round(parse_depth_logit), dtype=self.INT_TF)
                    elif self.parse_depth_loss_type.lower() == 'xent':
                        parse_depth_logit = parsing_logit[..., self.n_pos + self.n_parse_label:]
                        if residual_parser:
                            parse_depth_logit = tf.cumsum(parse_depth_logit, axis=1)
                        parse_depth_prediction = tf.argmax(parse_depth_logit, axis=-1)
                        parse_depth_prediction += self.parse_depth_min
                else:
                    parse_depth_logit = None
                    parse_depth_prediction = None

                out = {
                    'parsing_classifier': parsing_classifier,
                    'parsing_logit': parsing_logit,
                    'pos_label_logit': pos_label_logit,
                    'pos_label_prediction': pos_label_prediction,
                    'parse_label_logit': parse_label_logit,
                    'parse_label_prediction': parse_label_prediction,
                    'parse_depth_logit': parse_depth_logit,
                    'parse_depth_prediction': parse_depth_prediction
                }

                return out

    def _initialize_wp_decoder(self, teacher_forcing=False, pe_type=None, pe_n_units=128, name=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if name is None:
                    name = 'wp'

                rnn = self._initialize_rnn_module(
                    self.layers_wp_decoder,
                    self.units_wp_decoder,
                    bidirectional=False,
                    activation=self.wp_decoder_activation,
                    activation_inner=self.wp_decoder_activation_inner,
                    recurrent_activation=self.wp_decoder_recurrent_activation,
                    project_encodings=False,
                    resnet_n_layers_inner=self.wp_decoder_resnet_n_layers_inner,
                    return_sequences=True,
                    name=name + '_decoder'
                )

                if self.project_wp_decodings:
                    projection = self._initialize_dense_module(
                        1,
                        [self.word_emb_dim],
                        activation=None,
                        activation_inner=self.wp_projection_activation_inner,
                        resnet_n_layers_inner=self.wp_decoder_resnet_n_layers_inner,
                    )

                    projection = make_lambda(projection, self.sess)

                    module = compose_lambdas([rnn, projection])
                else:
                    assert self.units_wp_decoder[-1] == self.word_emb_dim, 'If project_wp_decodings is False, the final dimension of n_units_wp_decoder must be equal to the word embedding size. Expected %d, saw %d.' % (self.word_emb_dim, self.units_wp_decoder[-1])
                    module = rnn

                def wp_decoder(
                        s,
                        w,
                        e,
                        module=module,
                        teacher_forcing=teacher_forcing,
                        pe_type=pe_type,
                        pe_n_units=pe_n_units,
                        mask=None
                ):
                    # s -> sentence encodings
                    # w -> word one-hots
                    # e -> word embeddings, keys
                    with self.sess.as_default():
                        with self.sess.graph.as_default():
                            t = tf.cast(tf.shape(w)[-2], dtype=self.INT_TF)

                            # Tile sentence encodings
                            tile_ix = tf.stack([1] * (len(s.shape) - 1) + [t, 1], axis=0)
                            s = tf.tile(
                                tf.expand_dims(s, -2),
                                tile_ix
                            )

                            if teacher_forcing:
                                # Shift targets right and append sentence encodings
                                x = w[..., :-1, :]
                                x_pad = [(0,0) for _ in range(len(x.shape) - 2)] + [(0, 1), (0, 0)]
                                x = tf.pad(x, x_pad)
                                x = [x, s]
                            else:
                                x = [s]

                            if pe_type and pe_type.lower() in ['periodic', 'transformer_pe']:
                                time = tf.cast(tf.range(1, t + 1), dtype=self.FLOAT_TF)[..., None]
                                n = pe_n_units // 2

                                if pe_type.lower() == 'periodic':
                                    coef = tf.exp(tf.linspace(-2., 2., n))[None, ...]
                                elif pe_type.lower() == 'transformer_pe':
                                    log_timescale_increment = np.log(10000) / (n - 1)
                                    coef = (tf.exp(tf.cast(tf.range(n), dtype=self.FLOAT_TF) * -log_timescale_increment))[None, ...]

                                shape_base = tf.shape(x[0])
                                tile_ix = [shape_base[i] for i in range(len(x[0].shape) - 2)] + [1, 1]

                                sin = tf.sin(time * coef)
                                while len(sin.shape) < len(w.shape):
                                    sin = sin[None, ...]
                                sin = tf.tile(sin, tile_ix)

                                cos = tf.cos(time * coef)
                                while len(cos.shape) < len(w.shape):
                                    cos = cos[None, ...]
                                cos = tf.tile(cos, tile_ix)

                                x += [sin, cos]

                            if len(x) == 1:
                                x = x[0]
                            else:
                                x = tf.concat(x, axis=-1)

                            q = module(x, mask=mask)
                            dist = scaled_dot_attn(
                                q,
                                e,
                                w,
                                mask=mask,
                                float_type=self.FLOAT_TF,
                                int_type=self.INT_TF,
                                normalize_repeats=True,
                                epsilon=self.epsilon,
                                session=self.sess
                            )
                            logits = tf.log(tf.clip_by_value(dist, self.epsilon, 1-self.epsilon))

                            return logits

                return wp_decoder

    def _initialize_wp_classifier(self, name=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if name is None:
                    name = 'wp'

                units = self.wp_n_pos

                wp_classifier = self._initialize_dense_module(
                    self.layers_wp_classifier + 1,
                    self.units_wp_classifier + [units],
                    activation=None,
                    activation_inner=self.wp_classifier_activation_inner,
                    resnet_n_layers_inner=self.wp_classifier_resnet_n_layers_inner,
                    name=name + '_classifier'
                )

                return wp_classifier

    def _initialize_wp_outputs(self, module, s, w, e, mask=None, clip=None, reverse=True, mode='decoder'):
        # s -> sentence encodings
        # w -> word one-hots
        # e -> word embeddings, keys
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if mode.lower() == 'decoder':
                    if clip is not None:
                        s = s[..., -clip:, :]
                        w = w[..., -clip:, :]
                        e = e[..., -clip:, :]
                        if mask is not None:
                            mask = mask[..., -clip:, :]
                    if reverse:
                        w = tf.reverse(w, axis=[-2])
                        e = tf.reverse(e, axis=[-2])
                        if mask is not None:
                            mask = tf.reverse(mask, axis=[-1])
                    logit = module(s, w, e, mask=mask)
                    if reverse:
                        logit = tf.reverse(logit, axis=[-2])
                elif mode.lower() == 'classifier':
                    # Append sentence encodings
                    x = e
                    tile_ix = [1] * (len(s.shape) - 1) + [tf.shape(x)[-2]] + [1]
                    s = tf.tile(
                        tf.expand_dims(s, -2),
                        tile_ix
                    )
                    x = tf.concat([x, s], axis=-1)
                    if clip:
                        x = x[..., -clip:, :]
                    logit = module(x)
                else:
                    raise ValueError('Unrecognized WP loss mode "%s".' % mode)

                prediction = tf.argmax(logit, axis=-1, output_type=self.INT_TF)

                out = {
                    'logit': logit,
                    'prediction': prediction
                }

                return out

    def _initialize_sts_decoder(self, name=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Define some new tensors for STS prediction from encodings.
                if name is None:
                    name = 'sts'

                if self.sts_decoder_type.lower() == 'cnn':
                    sts_decoder = self._initialize_cnn_module(
                        self.layers_sts_decoder,
                        self.sts_kernel_size,
                        self.units_sts_decoder,
                        activation=self.sts_decoder_activation,
                        activation_inner=self.sts_decoder_activation_inner,
                        padding='same',
                        project_encodings=self.project_sts_decodings,
                        projection_activation_inner=self.sts_projection_activation_inner,
                        resnet_n_layers_inner=self.sts_decoder_resnet_n_layers_inner,
                        max_pooling_over_time=True,
                        name=name + '_decoder'
                    )  # confirm hyperparams with the shao2017 paper: CNN: 1 layer, filter_height=1, 300 filters, relu activation, no dropout or regularization.  then fed to difference and hadamard and concatenated.  then FCNN: 2 layers, 300 units, tanh activation, no regularization or dropout
                elif self.sts_decoder_type.lower() == 'rnn':
                    sts_decoder = self._initialize_rnn_module(
                        self.layers_sts_decoder,
                        self.units_sts_decoder,
                        bidirectional=self.bidirectional_sts_decoder,
                        activation=self.sts_decoder_activation,
                        activation_inner=self.sts_decoder_activation_inner,
                        recurrent_activation=self.sts_decoder_recurrent_activation,
                        project_encodings=self.project_sts_decodings,
                        projection_activation_inner=self.sts_projection_activation_inner,
                        resnet_n_layers_inner=self.sts_decoder_resnet_n_layers_inner,
                        return_sequences=False,
                        name=name + '_decoder'
                    )
                else:
                    raise ValueError('Unrecognized STS decoder type "%s".' % self.sts_decoder_type)

                return sts_decoder

    def _initialize_sts_outputs(self, s1, s2, use_classifier=True, name=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Define some new tensors for STS prediction from encodings.
                if self.sts_loss_type.lower() == 'mse':
                    outdim = 1
                else:
                    outdim = self.n_sts_label

                if use_classifier:
                    #sts prediction from encoder
                    sts_features = []
                    # sts_difference_features = tf.subtract(
                    #     sts_latent_s1,
                    #     sts_latent_s2,
                    #     name=name + '_difference_features'
                    # )
                    sts_difference_features = tf.abs(
                        s1 - s2,
                        name=name + '_difference_features'
                    )
                    sts_features.append(sts_difference_features)
                    sts_squared_difference_features = tf.pow(
                        s1 - s2,
                        2,
                        name=name + '_squared_difference_features'
                    )
                    sts_features.append(sts_squared_difference_features)
                    sts_product_features = tf.multiply(
                        s1,
                        s2,
                        name=name + '_product_features'
                    )
                    sts_features.append(sts_product_features)
                    sts_sim_features = cosine_similarity(s1, s2, epsilon=self.epsilon, session=self.sess)
                    sts_features.append(sts_sim_features)
                    # sts_features += [s1, s2]
                    sts_features = tf.concat(
                        values=sts_features,
                        axis=-1,
                        name=name + '_features'
                    )
                    #sts_logit from sts_features with 2 denselayer (section 2 fcnn) from Shao 2017
                    sts_classifier = self._initialize_dense_module(
                        self.layers_sts_classifier + 1,
                        self.units_sts_classifier + [outdim],
                        activation=None,
                        activation_inner=self.sts_classifier_activation_inner,
                        resnet_n_layers_inner=self.sts_classifier_resnet_n_layers_inner,
                        name=name + '_classifier'
                    )
                    sts_logit = sts_classifier(sts_features)
                    if self.sts_loss_type.lower() == 'mse':
                        sts_logit = tf.squeeze(sts_logit, axis=-1)
                        sts_prediction = sts_logit
                    elif self.sts_loss_type.lower() == 'xent':
                        sts_prediction = tf.argmax(sts_logit, axis=-1, output_type=self.INT_TF)
                    else:
                        raise ValueError('Unrecognized sts_loss_type "%s".' % self.sts_loss_type)

                else:
                    sts_logit = cosine_similarity(s1, s2, epsilon=self.epsilon, session=self.sess)[..., 0] * 5 # Output is on a scale from 0 to 5
                    sts_prediction = sts_logit

                out = {
                    # 'sts_difference_features': sts_difference_features,
                    # 'sts_squared_difference_features': sts_squared_difference_features,
                    # 'sts_product_features': sts_product_features,
                    # 'sts_features': sts_features,
                    # 'sts_classifier': sts_classifier,
                    'logit': sts_logit,
                    'prediction': sts_prediction
                }

                return out

    def _initialize_parsing_objective(
            self,
            pos_label_logit,
            parse_label_logit,
            parse_depth_logit,
            pos_label=None,
            parse_label=None,
            parse_depth=None,
            weights=None,
            well_formedness_loss_scale=False,
            nonzero_scale=True
    ):
        if nonzero_scale:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if pos_label is None:
                        pos_label = self.pos_label
                    if parse_label is None:
                        parse_label = self.parse_label
                    if self.factor_parse_labels and parse_depth is None:
                        parse_depth = self.parse_depth

                    pos_label_loss = tf.losses.sparse_softmax_cross_entropy(
                        pos_label,
                        pos_label_logit,
                        weights=weights
                    )
                    parse_label_loss = tf.losses.sparse_softmax_cross_entropy(
                        parse_label,
                        parse_label_logit,
                        weights=weights
                    )

                    loss = pos_label_loss + parse_label_loss

                    if self.factor_parse_labels:
                        if self.parse_depth_loss_type.lower() == 'mse':
                            parse_depth_loss = tf.losses.mean_squared_error(
                                parse_depth,
                                parse_depth_logit,
                                weights=weights
                            )
                        elif self.parse_depth_loss_type.lower() == 'xent':
                            parse_depth_loss = tf.losses.sparse_softmax_cross_entropy(
                                parse_depth,
                                parse_depth_logit,
                                weights=weights
                            )
                        else:
                            raise ValueError(
                                'Unrecognized value for parse_depth_loss_type: "%s".' % self.parse_depth_loss_type)

                        loss += parse_depth_loss

                        if well_formedness_loss_scale and self.parse_depth_loss_type.lower() == 'mse':
                            # Define well-formedness losses.
                            #   ZERO SUM: In a valid tree, word-by-word changes in depth should sum to 0.
                            #             Encouraged by an L1 loss on the sum of the predicted depths.
                            #   NO NEG:   In a valid tree, no word should close more constituents than it has ancestors.
                            #             Encouraged by an L1 loss on negative cells in a cumsum over predicted depths.

                            # Normalize by the minibatch size
                            zero_sum_denom = tf.cast(tf.shape(parse_label)[0], dtype=self.FLOAT_TF)
                            # Normalize by the number of non-padding words
                            if weights is None:
                                no_neg_denom = tf.cast(tf.reduce_prod(tf.shape(parse_label)), dtype=self.FLOAT_TF)
                            else:
                                no_neg_denom = tf.reduce_sum(weights) + self.epsilon

                            masked_depth_logit = parse_depth_logit
                            if weights is not None:
                                masked_depth_logit *= weights
                            depth_abs_sums = tf.abs(tf.reduce_sum(masked_depth_logit, axis=1)) # Trying to make these 0
                            depth_abs_clipped_cumsums = tf.abs(tf.clip_by_value(tf.cumsum(masked_depth_logit), -np.inf, 0.))

                            zero_sum_loss = tf.reduce_sum(depth_abs_sums, axis=0) / zero_sum_denom
                            no_neg_loss = tf.reduce_sum(depth_abs_clipped_cumsums, axis=0) / no_neg_denom

                            loss += (zero_sum_loss + no_neg_loss) * well_formedness_loss_scale
                        else:
                            zero_sum_loss = tf.convert_to_tensor(0.)
                            no_neg_loss = tf.convert_to_tensor(0.)
                    else:
                        parse_depth_loss = tf.convert_to_tensor(0.)
                        zero_sum_loss = tf.convert_to_tensor(0.)
                        no_neg_loss = tf.convert_to_tensor(0.)

                    loss *= nonzero_scale
        else:
            loss = tf.convert_to_tensor(0.)
            pos_label_loss = tf.convert_to_tensor(0.)
            parse_label_loss = tf.convert_to_tensor(0.)
            parse_depth_loss = tf.convert_to_tensor(0.)
            zero_sum_loss = tf.convert_to_tensor(0.)
            no_neg_loss = tf.convert_to_tensor(0.)

        out = {
            'loss': loss,
            'pos_label_loss': pos_label_loss,
            'parse_label_loss': parse_label_loss,
            'parse_depth_loss': parse_depth_loss,
            'zero_sum_loss': zero_sum_loss,
            'no_neg_loss': no_neg_loss
        }

        return out

    def _initialize_wp_objective(
            self,
            logit,
            target=None,
            mode='decoder',
            weights=None,
            nonzero_scale=True
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if nonzero_scale:
                    if mode.lower() == 'decoder':
                        assert target is not None, 'Target must be provided to WP objective in decoder mode.'
                        loss = tf.losses.sparse_softmax_cross_entropy(
                            target,
                            logit,
                            weights=weights
                        )
                    elif mode.lower() == 'classifier':
                        target = tf.range(self.wp_n_pos, dtype=self.INT_TF)[None, ...]
                        if weights is not None:
                            start_ix = tf.cast(tf.reduce_sum(1 - weights, axis=-1, keepdims=True), dtype=self.INT_TF)
                            target = tf.maximum(0, target - start_ix)

                        loss = tf.losses.sparse_softmax_cross_entropy(
                            target,
                            logit
                        )
                    else:
                        raise ValueError('Unrecognized WP loss mode "%s".' % mode)

                    n = tf.reduce_sum(weights)
                    acc = tf.reduce_sum(
                        tf.cast(
                            tf.equal(
                                target,
                                tf.argmax(logit, axis=-1, output_type=self.INT_TF)
                            ),
                            dtype=self.FLOAT_TF
                        ) * weights
                    ) / tf.maximum(n, self.epsilon)
                else:
                    loss = tf.convert_to_tensor(0.)
                    acc = tf.convert_to_tensor(0.)
                    n = tf.convert_to_tensor(0.)

                out = {
                    'loss': loss,
                    'acc': acc,
                    'n': n
                }

                return out

    def _initialize_sts_objective(self, logit, label=None, nonzero_scale=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if nonzero_scale:
                    if label is None:
                        label = self.sts_label
                    if self.sts_loss_type.lower() == 'mse':
                        loss = tf.losses.mean_squared_error(
                            label,
                            logit
                        )
                    elif self.sts_loss_type.lower() == 'xent':
                        loss = tf.losses.sparse_softmax_cross_entropy(
                            label,
                            logit
                        )
                    else:
                        raise ValueError('Unrecognized sts_loss_type "%s".' % self.sts_loss_type)
                else:
                    loss = tf.convert_to_tensor(0.)

                out = {
                    'loss': loss
                }

                return out

    def _initialize_train_op(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_batch_step)

    def _initialize_optimizer(self, name):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase

                    if self.lr_decay_iteration_power != 1:
                        t = tf.cast(self.step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.step

                    if self.lr_decay_family.lower() == 'linear_decay':
                        if lr_decay_staircase:
                            decay = tf.floor(t / lr_decay_steps)
                        else:
                            decay = t / lr_decay_steps
                        decay *= lr_decay_rate
                        self.lr = lr - decay
                    else:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            t,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                return {
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip) if clip else tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip) if clip else tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip) if clip else tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip) if clip else tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip) if clip else tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    vars = [var for var in tf.get_collection('trainable_variables') if 'BatchNorm' not in var.name]

                    self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                    self.ema_op = self.ema.apply(vars)
                    self.ema_map = {}
                    for v in vars:
                        self.ema_map[self.ema.average_name(v)] = v
                    self.ema_saver = tf.train.Saver(self.ema_map)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.log_graph:
                    self.train_syn_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/train_syn', self.sess.graph)
                else:
                    self.train_syn_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/train_syn')
                self.dev_syn_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dev_syn')
                self.train_sem_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/train_sem')
                self.dev_sem_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dev_sem')

                self.parsing_loss_log_entries = self._initialize_parsing_loss_log_entries()
                self.parsing_eval_log_entries = self._initialize_parsing_eval_log_entries()
                self.wp_loss_log_entries = self._initialize_wp_loss_log_entries()
                self.wp_eval_log_entries = self._initialize_wp_eval_log_entries()
                self.sts_loss_log_entries = self._initialize_sts_loss_log_entries()
                self.sts_eval_log_entries = self._initialize_sts_eval_log_entries()

                self.parsing_loss_log_summaries = self._initialize_log_summaries(
                    self.parsing_loss_log_entries,
                    collection='parsing_loss'
                )
                self.parsing_eval_log_summaries = self._initialize_log_summaries(
                    self.parsing_eval_log_entries,
                    collection='parsing_eval'
                )
                self.wp_loss_log_summaries = self._initialize_log_summaries(
                    self.wp_loss_log_entries,
                    collection='wp_loss'
                )
                self.wp_eval_log_summaries = self._initialize_log_summaries(
                    self.wp_eval_log_entries,
                    collection='wp_eval'
                )
                self.sts_loss_log_summaries = self._initialize_log_summaries(
                    self.sts_loss_log_entries,
                    collection='sts_loss'
                )
                self.sts_eval_log_summaries = self._initialize_log_summaries(
                    self.sts_eval_log_entries,
                    collection='sts_eval'
                )

                self.parsing_loss_summary = tf.summary.merge_all(key='parsing_loss')
                self.parsing_eval_summary = tf.summary.merge_all(key='parsing_eval')
                self.wp_loss_summary = tf.summary.merge_all(key='wp_loss')
                self.wp_eval_summary = tf.summary.merge_all(key='wp_eval')
                self.sts_loss_summary = tf.summary.merge_all(key='sts_loss')
                self.sts_eval_summary = tf.summary.merge_all(key='sts_eval')

                # TODO: Cory, add BOW logging

    def _initialize_parsing_loss_log_entries(self):
        out = [
            'pos_label_loss',
            'parse_label_loss'
        ]
        if self.factor_parse_labels:
            out.append('parse_depth_loss')
        
        return out

    def _initialize_parsing_eval_log_entries(self):
        return [
            'f',
            'p',
            'r',
            'tag',
        ]

    def _initialize_wp_loss_log_entries(self):
        return [
            'wp_loss'
        ]

    def _initialize_wp_eval_log_entries(self):
        return [
            'wp_acc'
        ]

    def _initialize_sts_loss_log_entries(self):
        return ['sts_loss']

    def _initialize_sts_eval_log_entries(self):
        return ['sts_r']

    def _initialize_log_summaries(self, log_entries, collection=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                log_summaries = {}

                for x in log_entries:
                    log_summaries[x] = tf.placeholder(self.FLOAT_TF, shape=[], name=x + '_placeholder')
                    tf.summary.scalar('%s/%s' % (collection, x), log_summaries[x], collections=[collection])

                return log_summaries

    def _initialize_sts_metric_log_entries(self, syn=True, sem=True):
        pass

    def _initialize_sts_metric_log_summaries(self, log_entries, collection='parsing_metrics'):
        pass
        

    ############################################################
    # Private utility methods
    ############################################################

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__(*args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))
                return grads_and_vars

            def apply_gradients(self, grads_and_vars, **kwargs):
                if self.max_global_norm is None:
                    return grads_and_vars
                grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))

                return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

        return ClippedOptimizer

    def _get_n_batches(self, i):
        if self.eval_freq:
            eval_freq = self.eval_freq
        else:
            eval_freq = np.inf
        if self.save_freq:
            save_freq = self.save_freq
        else:
            save_freq = np.inf
        if self.log_freq:
            log_freq = self.log_freq
        else:
            log_freq = np.inf

        evaluate = ((i + 1) % eval_freq == 0) and (i > self.n_pretrain_steps)
        save = evaluate or ((i + 1) % save_freq == 0)
        log = (i + 1) % log_freq == 0

        if save_freq:
            next_save = save_freq - (i % save_freq)
        else:
            next_save = np.inf

        if log_freq:
            next_log = log_freq - (i % log_freq)
        else:
            next_log = np.inf

        if eval_freq:
            next_evaluate = eval_freq - (i % eval_freq)
        else:
            next_evaluate = np.inf

        n = min(
            next_save,
            next_log,
            next_evaluate
        )

        save = next_save == n
        log = next_log == n
        evaluate = next_evaluate == n

        return n, save, log, evaluate

    def _run_batches_inner(
            self,
            to_run,
            to_run_names,
            data_feed,
            n_minibatch,
            data_type='both',
            info_dict=None,
            update=False,
            return_syn_parsing_loss=False,
            return_syn_wp_loss=False,
            return_sem_sts_loss=False,
            return_syn_parsing_prediction=False,
            return_sem_parsing_prediction=False,
            return_syn_wp_prediction=False,
            return_sem_wp_prediction=False,
            return_syn_sts_prediction=False,
            return_sem_sts_prediction=False,
            verbose=True
    ):
        if info_dict is None:
            info_dict = {}
        gold_keys = set()
        for k in to_run_names:
            if 'loss' in k:
                info_dict[k] = 0.
            elif 'prediction' in k:
                info_dict[k] = []
                gold_key = k.replace('_syn', '').replace('_sem', '').replace('prediction', 'true')
                if not gold_key in info_dict:
                    gold_keys.add(gold_key)
                    info_dict[gold_key] = []

        if data_type.lower() in ['parsing', 'both'] and return_syn_parsing_prediction or return_sem_parsing_prediction:
            info_dict['parsing_text'] = []
            info_dict['parsing_text_mask'] = []
            gold_keys.add('parsing_text')
            gold_keys.add('parsing_text_mask')
            if return_syn_wp_prediction:
                info_dict['wp_parsing_true'] = []

        if data_type.lower() in ['sts', 'both'] and return_syn_sts_prediction or return_sem_sts_prediction:
            info_dict['sts_s1_text'] = []
            info_dict['sts_s1_text_mask'] = []
            info_dict['sts_s2_text'] = []
            info_dict['sts_s2_text_mask'] = []
            gold_keys.add('sts_s1_text')
            gold_keys.add('sts_s1_text_mask')
            gold_keys.add('sts_s2_text')
            gold_keys.add('sts_s2_text_mask')
            if return_syn_wp_prediction:
                info_dict['wp_sts_s1_true'] = []
                info_dict['wp_sts_s2_true'] = []

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if verbose:
                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                i = 0
                while i < n_minibatch:
                    batch = next(data_feed)
                    fd_minibatch = {}

                    if data_type.lower() in ['parsing', 'both']:
                        parsing_text_batch = batch['parsing_text']
                        parsing_normalized_text_batch = batch['parsing_normalized_text']
                        parsing_text_mask_batch = batch['parsing_text_mask']
                        pos_label_batch = batch['pos_label']
                        parse_label_batch = batch['parse_label']
                        parse_depth_batch = batch['parse_depth']

                        if 'parsing_text' in gold_keys:
                            info_dict['parsing_text'].append(parsing_text_batch)
                        if 'parsing_text_mask' in gold_keys:
                            info_dict['parsing_text_mask'].append(parsing_text_mask_batch)
                        if 'pos_label_true' in gold_keys:
                            info_dict['pos_label_true'].append(pos_label_batch)
                        if 'parse_label_true' in gold_keys:
                            info_dict['parse_label_true'].append(parse_label_batch)
                        if 'parse_depth_true' in gold_keys:
                            info_dict['parse_depth_true'].append(parse_depth_batch)
                        if 'wp_parsing_true' in gold_keys:
                            info_dict['wp_parsing_true'].append(parsing_normalized_text_batch)

                        fd_minibatch.update({
                            self.parsing_chars: parsing_text_batch,
                            self.parsing_words: parsing_normalized_text_batch,
                            self.parsing_char_mask: parsing_text_mask_batch,
                            self.pos_label: pos_label_batch,
                            self.parse_label: parse_label_batch,
                        })
                        if self.factor_parse_labels:
                            fd_minibatch[self.parse_depth_src] = parse_depth_batch
                    if data_type.lower() in ['sts', 'both']:
                        sts_s1_text_batch = batch['sts_s1_text']
                        sts_s1_normalized_text_batch = batch['sts_s1_normalized_text']
                        sts_s1_text_mask_batch = batch['sts_s1_text_mask']
                        sts_s2_text_batch = batch['sts_s2_text']
                        sts_s2_normalized_text_batch = batch['sts_s2_normalized_text']
                        sts_s2_text_mask_batch = batch['sts_s2_text_mask']
                        sts_label_batch = batch['sts_label']
                        if 'sts_s1_text' in gold_keys:
                            info_dict['sts_s1_text'].append(sts_s1_text_batch)
                        if 'sts_s1_text_mask' in gold_keys:
                            info_dict['sts_s1_text_mask'].append(sts_s1_text_mask_batch)
                        if 'sts_s2_text' in gold_keys:
                            info_dict['sts_s2_text'].append(sts_s2_text_batch)
                        if 'sts_s2_text_mask' in gold_keys:
                            info_dict['sts_s2_text_mask'].append(sts_s2_text_mask_batch)
                        if 'sts_true' in gold_keys:
                            info_dict['sts_true'].append(sts_label_batch)
                        if 'wp_sts_s1_true' in gold_keys:
                            info_dict['wp_sts_s1_true'].append(sts_s1_normalized_text_batch)
                        if 'wp_sts_s2_true' in gold_keys:
                            info_dict['wp_sts_s2_true'].append(sts_s2_normalized_text_batch)

                        fd_minibatch.update({
                            self.sts_s1_chars: sts_s1_text_batch,
                            self.sts_s1_words: sts_s1_normalized_text_batch,
                            self.sts_s1_char_mask: sts_s1_text_mask_batch,
                            self.sts_s2_chars: sts_s2_text_batch,
                            self.sts_s2_words: sts_s2_normalized_text_batch,
                            self.sts_s2_char_mask: sts_s2_text_mask_batch,
                            self.sts_label: sts_label_batch
                        })

                    out = self.sess.run(
                        to_run,
                        feed_dict=fd_minibatch
                    )

                    batch_dict = {}
                    for j, x in enumerate(out):
                        batch_dict[to_run_names[j]] = x

                    for k in batch_dict:
                        # if 'loss' in k or 'acc' in k:
                        if 'loss' in k:
                            info_dict[k] += batch_dict[k]
                        elif 'prediction' in k:
                            info_dict[k].append(batch_dict[k])

                    if verbose:
                        values = []
                        if update:
                            if data_type.lower() in ['parsing', 'both'] and return_syn_parsing_loss:
                                values += [
                                    ('pos', batch_dict['pos_label_loss_syn']),
                                    ('label', batch_dict['parse_label_loss_syn'])
                                ]
                                if self.factor_parse_labels:
                                    values += [
                                        ('depth', batch_dict['parse_depth_loss_syn'])
                                    ]
                            if data_type.lower() in ['sts', 'both'] and return_sem_sts_loss:
                                values += [
                                    ('sts', batch_dict['sts_loss_sem']),
                                ]
                            if return_syn_wp_loss:
                                values += [
                                    ('wp', batch_dict['wp_loss_syn'])
                                ]
                        pb.update(i + 1, values=values)

                    i += 1

                for k in to_run_names + sorted(list(gold_keys)):
                    # if 'loss' in k or 'acc' in k:
                    if 'loss' in k:
                        info_dict[k] /= n_minibatch
                    elif 'prediction' in k or k in gold_keys:
                        if len(info_dict[k]) > 0:
                            info_dict[k] = padded_concat(info_dict[k], axis=0)
                        else:
                            print('Empty list:')
                            print(k)
                            print()

        return info_dict

    def _run_batches(
            self,
            data,
            data_name='train',
            minibatch_size=None,
            n_minibatch=None,
            update=False,
            randomize=False,
            return_syn_parsing_losses=False,
            return_sem_parsing_losses=False,
            return_syn_parsing_prediction=False,
            return_sem_parsing_prediction=False,
            return_syn_wp_loss=False,
            return_sem_wp_loss=False,
            return_syn_wp_prediction=False,
            return_sem_wp_prediction=False,
            return_syn_sts_loss=False,
            return_sem_sts_loss=False,
            return_syn_sts_prediction=False,
            return_sem_sts_prediction=False,
            return_syn_bow_loss=False,
            return_sem_bow_loss=False,
            return_syn_bow_prediction=False,
            return_sem_bow_prediction=False,
            verbose=True
    ):
        parsing_loss_tensors, parsing_loss_tensor_names = self._get_parsing_loss_tensors(
            syn=return_syn_parsing_losses,
            sem=return_sem_parsing_losses
        )

        parsing_prediction_tensors, parsing_prediction_tensor_names = self._get_parsing_prediction_tensors(
            syn=return_syn_parsing_prediction,
            sem=return_sem_parsing_prediction
        )
        
        wp_loss_tensors, wp_loss_tensor_names = self._get_wp_loss_tensors(
            syn=return_syn_wp_loss,
            sem=return_sem_wp_loss
        )

        wp_prediction_tensors, wp_prediction_tensor_names = self._get_wp_prediction_tensors(
            syn=return_syn_wp_prediction,
            sem=return_sem_wp_prediction
        )

        sts_loss_tensors, sts_loss_tensor_names = self._get_sts_loss_tensors(
            syn=return_syn_sts_loss,
            sem=return_sem_sts_loss
        )

        sts_prediction_tensors, sts_prediction_tensor_names = self._get_sts_prediction_tensors(
            syn=return_syn_sts_prediction,
            sem=return_sem_sts_prediction
        )

        info_dict = {}

        if update:
            if verbose:
                batch_ix = self.global_batch_step.eval(session=self.sess)
                stderr('Running minibatches %d-%d...\n' % (batch_ix + 1, batch_ix + n_minibatch))

            to_run = [self.train_op]
            to_run_names = ['train_op']

            to_run += [
                self.loss
            ]
            to_run_names += [
                'loss'
            ]

            to_run += parsing_loss_tensors + \
                      parsing_prediction_tensors + \
                      wp_loss_tensors + \
                      wp_prediction_tensors + \
                      sts_loss_tensors + \
                      sts_prediction_tensors

            to_run_names += parsing_loss_tensor_names + \
                            parsing_prediction_tensor_names + \
                            wp_loss_tensor_names + \
                            wp_prediction_tensor_names + \
                            sts_loss_tensor_names + \
                            sts_prediction_tensor_names

            data_feed = data.get_training_data_feed(
                data_name,
                parsing=True,
                sts=True,
                integer_sts_targets=self.sts_loss_type.lower() == 'xent',
                minibatch_size=minibatch_size,
                randomize=randomize
            )

            info_dict = self._run_batches_inner(
                to_run,
                to_run_names,
                data_feed,
                n_minibatch,
                data_type='both',
                info_dict=info_dict,
                update=update,
                return_syn_parsing_loss=return_syn_parsing_losses,
                return_syn_wp_loss=return_syn_wp_loss,
                return_sem_sts_loss=return_sem_sts_loss,
                return_syn_parsing_prediction=return_syn_parsing_prediction,
                return_sem_parsing_prediction=return_sem_parsing_prediction,
                return_syn_wp_prediction=return_syn_wp_prediction,
                return_sem_wp_prediction=return_sem_wp_prediction,
                return_syn_sts_prediction=return_syn_sts_prediction,
                return_sem_sts_prediction=return_sem_sts_prediction,
                verbose=verbose
            )
        else:
            # Parsing
            wp_parsing_loss_tensors, wp_parsing_loss_tensor_names = self._get_wp_parsing_loss_tensors(
                syn=return_syn_wp_loss,
                sem=return_sem_wp_loss
            )
            wp_parsing_prediction_tensors, wp_parsing_prediction_tensor_names = self._get_wp_parsing_prediction_tensors(
                syn=return_syn_wp_prediction,
                sem=return_sem_wp_prediction
            )
            parsing_tensors = parsing_loss_tensors + \
                              parsing_prediction_tensors + \
                              wp_parsing_loss_tensors + \
                              wp_parsing_prediction_tensors
            parsing_tensor_names = parsing_loss_tensor_names + \
                                   parsing_prediction_tensor_names + \
                                   wp_parsing_loss_tensor_names + \
                                   wp_parsing_prediction_tensor_names
            if len(parsing_tensors) > 0:
                if verbose:
                    stderr('Extracting parse predictions...\n')
                if minibatch_size is None:
                    minibatch_size = self.minibatch_size
                if n_minibatch is None:
                    n_minibatch_cur = data.get_n_minibatch(data_name, minibatch_size, task='parsing')
                else:
                    n_minibatch_cur = n_minibatch

                to_run = parsing_tensors
                to_run_names = parsing_tensor_names

                data_feed = data.get_parsing_data_feed(
                    data_name,
                    minibatch_size=minibatch_size,
                    randomize=randomize
                )

                info_dict = self._run_batches_inner(
                    to_run,
                    to_run_names,
                    data_feed,
                    n_minibatch_cur,
                    data_type='parsing',
                    info_dict=info_dict,
                    update=update,
                    return_syn_parsing_loss=return_syn_parsing_losses,
                    return_syn_parsing_prediction=return_syn_parsing_prediction,
                    return_syn_wp_prediction=return_syn_wp_prediction,
                    return_sem_wp_prediction=return_sem_wp_prediction,
                    return_sem_parsing_prediction=return_sem_parsing_prediction,
                    verbose=verbose
                )

            # STS
            wp_sts_loss_tensors, wp_sts_loss_tensor_names = self._get_wp_sts_loss_tensors(
                syn=return_syn_wp_loss,
                sem=return_sem_wp_loss
            )
            wp_sts_prediction_tensors, wp_sts_prediction_tensor_names = self._get_wp_sts_prediction_tensors(
                syn=return_syn_wp_prediction,
                sem=return_sem_wp_prediction
            )
            sts_tensors = sts_loss_tensors + \
                          sts_prediction_tensors + \
                          wp_sts_loss_tensors + \
                          wp_sts_prediction_tensors
            sts_tensor_names = sts_loss_tensor_names + \
                               sts_prediction_tensor_names + \
                               wp_sts_loss_tensor_names + \
                               wp_sts_prediction_tensor_names
            if len(sts_tensors) > 0:
                if verbose:
                    stderr('Extracting STS predictions...\n')
                if minibatch_size is None:
                    minibatch_size = self.minibatch_size
                if n_minibatch is None:
                    n_minibatch_cur = data.get_n_minibatch(data_name, minibatch_size, task='sts')
                else:
                    n_minibatch_cur = n_minibatch

                to_run = sts_tensors
                to_run_names = sts_tensor_names

                data_feed = data.get_sts_data_feed(
                    data_name,
                    integer_targets=self.sts_loss_type.lower() == 'xent',
                    minibatch_size=minibatch_size,
                    randomize=randomize
                )

                info_dict = self._run_batches_inner(
                    to_run,
                    to_run_names,
                    data_feed,
                    n_minibatch_cur,
                    data_type='sts',
                    info_dict=info_dict,
                    update=update,
                    return_sem_sts_loss=return_sem_sts_loss,
                    return_syn_wp_prediction=return_syn_wp_prediction,
                    return_sem_wp_prediction=return_sem_wp_prediction,
                    return_syn_sts_prediction=return_syn_sts_prediction,
                    return_sem_sts_prediction=return_sem_sts_prediction,
                    verbose=verbose
                )

            if return_syn_wp_loss:
                wp_loss_syn = 0.
                if 'wp_parsing_loss_syn' in info_dict:
                    wp_loss_syn += info_dict['wp_parsing_loss_syn']
                if 'wp_sts_s1_loss_syn' in info_dict:
                    wp_loss_syn += info_dict['wp_sts_s1_loss_syn']
                if 'wp_sts_s2_loss_syn' in info_dict:
                    wp_loss_syn += info_dict['wp_sts_s2_loss_syn']
                info_dict['wp_loss_syn'] = wp_loss_syn
                
                wp_acc_syn = 0.
                n = 0
                if 'wp_parsing_acc_syn' in info_dict:
                    n += info_dict['wp_parsing_n_syn']
                    wp_acc_syn += info_dict['wp_parsing_acc_syn'] * info_dict['wp_parsing_n_syn']
                if 'wp_sts_s1_acc_syn' in info_dict:
                    n += info_dict['wp_sts_s1_n_syn']
                    wp_acc_syn += info_dict['wp_sts_s1_acc_syn'] * info_dict['wp_sts_s1_n_syn']
                if 'wp_sts_s2_acc_syn' in info_dict:
                    n += info_dict['wp_sts_s2_n_syn']
                    wp_acc_syn += info_dict['wp_sts_s2_acc_syn'] * info_dict['wp_sts_s1_n_syn']

                wp_acc_syn = wp_acc_syn / np.maximum(n, self.epsilon)
                info_dict['wp_acc_syn'] = wp_acc_syn

            if return_sem_wp_loss:
                wp_loss_sem = 0.
                if 'wp_parsing_loss_sem' in info_dict:
                    wp_loss_sem += info_dict['wp_parsing_loss_sem']
                if 'wp_sts_s1_loss_sem' in info_dict:
                    wp_loss_sem += info_dict['wp_sts_s1_loss_sem']
                if 'wp_sts_s2_loss_sem' in info_dict:
                    wp_loss_sem += info_dict['wp_sts_s2_loss_sem']
                info_dict['wp_loss_sem'] = wp_loss_sem

                wp_acc_sem = 0.
                n = 0
                if 'wp_parsing_acc_sem' in info_dict:
                    n += info_dict['wp_parsing_n_sem']
                    wp_acc_sem += info_dict['wp_parsing_acc_sem'] * info_dict['wp_parsing_n_sem']
                if 'wp_sts_s1_acc_sem' in info_dict:
                    n += info_dict['wp_sts_s1_n_sem']
                    wp_acc_sem += info_dict['wp_sts_s1_acc_sem'] * info_dict['wp_sts_s1_n_sem']
                if 'wp_sts_s2_acc_sem' in info_dict:
                    n += info_dict['wp_sts_s2_n_sem']
                    wp_acc_sem += info_dict['wp_sts_s2_acc_sem'] * info_dict['wp_sts_s1_n_sem']

                wp_acc_sem = wp_acc_sem / np.maximum(n, self.epsilon)
                info_dict['wp_acc_sem'] = wp_acc_sem

        return info_dict

    def _get_parsing_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            tensors += [
                self.parse_label_loss_syn,
                self.pos_label_loss_syn,
            ]
            tensor_names += [
                'parse_label_loss_syn',
                'pos_label_loss_syn',
            ]
            if self.factor_parse_labels:
                tensors.append(self.parse_depth_loss_syn)
                tensor_names.append('parse_depth_loss_syn')

        if sem:
            tensors += [
                self.parse_label_loss_sem,
                self.pos_label_loss_sem,
            ]
            tensor_names += [
                'parse_label_loss_sem',
                'pos_label_loss_sem',
            ]
            if self.factor_parse_labels:
                tensors.append(self.parse_depth_loss_sem)
                tensor_names.append('parse_depth_loss_sem')

        return tensors, tensor_names

    def _get_parsing_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            tensors += [
                self.pos_label_prediction_syn,
                self.parse_label_prediction_syn
            ]
            tensor_names += [
                'pos_label_prediction_syn',
                'parse_label_prediction_syn'
            ]
            if self.factor_parse_labels:
                tensors += [
                    self.parse_depth_prediction_syn
                ]
                tensor_names += [
                    'parse_depth_prediction_syn'
                ]
        if sem:
            tensors += [
                self.pos_label_prediction_sem,
                self.parse_label_prediction_sem
            ]
            tensor_names += [
                'pos_label_prediction_sem',
                'parse_label_prediction_sem'
            ]
            if self.factor_parse_labels:
                tensors += [
                    self.parse_depth_prediction_sem
                ]
                tensor_names += [
                    'parse_depth_prediction_sem'
                ]

        return tensors, tensor_names

    def _get_wp_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get WP loss tensors and names from syntactic encoder
            tensors += [
                self.wp_parsing_loss_syn,
                self.wp_sts_s1_loss_syn,
                self.wp_sts_s2_loss_syn,
                self.wp_loss_syn
            ]
            tensor_names += [
                'wp_parsing_loss_syn',
                'wp_sts_s1_loss_syn',
                'wp_sts_s2_loss_syn',
                'wp_loss_syn'
            ]
        if sem:
            # Get WP loss tensors and names from semantic encoder
            tensors += [
                self.wp_parsing_loss_sem,
                self.wp_sts_s1_loss_sem,
                self.wp_sts_s2_loss_sem,
                self.wp_loss_syn
            ]
            tensor_names += [
                'wp_parsing_loss_sem',
                'wp_sts_s1_loss_sem',
                'wp_sts_s2_loss_sem',
                'wp_loss_sem'
            ]

        return tensors, tensor_names

    def _get_wp_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get WP prediction tensors and names from syntactic encoder
            tensors += [
                self.wp_parsing_prediction_syn,
                self.wp_sts_s1_prediction_syn,
                self.wp_sts_s2_prediction_syn
            ]
            tensor_names += [
                'wp_parsing_prediction_syn',
                'wp_sts_s1_prediction_syn',
                'wp_sts_s2_prediction_syn'
            ]
        if sem:
            # Get WP prediction tensors and names from semantic encoder
            tensors += [
                self.wp_parsing_prediction_sem,
                self.wp_sts_s1_prediction_sem,
                self.wp_sts_s2_prediction_sem
            ]
            tensor_names += [
                'wp_parsing_prediction_sem',
                'wp_sts_s1_prediction_sem',
                'wp_sts_s2_prediction_sem'
            ]

        return tensors, tensor_names


    def _get_wp_parsing_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get WP loss tensors and names from syntactic encoder
            tensors += [
                self.wp_parsing_loss_syn
            ]
            tensor_names += [
                'wp_parsing_loss_syn'
            ]
        if sem:
            # Get WP loss tensors and names from semantic encoder
            tensors += [
                self.wp_parsing_loss_sem
            ]
            tensor_names += [
                'wp_parsing_loss_sem'
            ]

        return tensors, tensor_names

    def _get_wp_parsing_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get WP prediction tensors and names from syntactic encoder
            tensors += [
                self.wp_parsing_prediction_syn
            ]
            tensor_names += [
                'wp_parsing_prediction_syn'
            ]
        if sem:
            # Get WP prediction tensors and names from semantic encoder
            tensors += [
                self.wp_parsing_prediction_sem
            ]
            tensor_names += [
                'wp_parsing_prediction_sem'
            ]

        return tensors, tensor_names


    def _get_wp_sts_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get WP loss tensors and names from syntactic encoder
            tensors += [
                self.wp_sts_s1_loss_syn,
                self.wp_sts_s2_loss_syn
            ]
            tensor_names += [
                'wp_sts_s1_loss_syn',
                'wp_sts_s2_loss_syn'
            ]
        if sem:
            # Get WP loss tensors and names from semantic encoder
            tensors += [
                self.wp_sts_s1_loss_sem,
                self.wp_sts_s2_loss_sem
            ]
            tensor_names += [
                'wp_sts_s1_loss_sem',
                'wp_sts_s2_loss_sem'
            ]

        return tensors, tensor_names

    def _get_wp_sts_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get WP prediction tensors and names from syntactic encoder
            tensors += [
                self.wp_sts_s1_prediction_syn,
                self.wp_sts_s2_prediction_syn
            ]
            tensor_names += [
                'wp_sts_s1_prediction_syn',
                'wp_sts_s2_prediction_syn'
            ]
        if sem:
            # Get WP prediction tensors and names from semantic encoder
            tensors += [
                self.wp_sts_s1_prediction_sem,
                self.wp_sts_s2_prediction_sem
            ]
            tensor_names += [
                'wp_sts_s1_prediction_sem',
                'wp_sts_s2_prediction_sem'
            ]

        return tensors, tensor_names

    def _get_sts_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get STS loss tensors and names from syntactic encoder
            tensors += [self.sts_loss_syn]
            tensor_names += ['sts_loss_syn']
        if sem:
            # Get STS loss tensors and names from semantic encoder
            tensors += [self.sts_loss_sem]
            tensor_names += ['sts_loss_sem']

        return tensors, tensor_names

    def _get_sts_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get STS prediction tensors and names from syntactic encoder
            tensors += [self.sts_prediction_syn]
            tensor_names += ['sts_prediction_syn']
        if sem:
            # Get STS prediction tensors and names from semantic encoder
            tensors += [self.sts_prediction_sem]
            tensor_names += ['sts_prediction_sem']

        return tensors, tensor_names

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    stderr('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    if allow_missing:
                        reader = tf.train.NewCheckpointReader(path)
                        saved_shapes = reader.get_variable_to_shape_map()
                        model_var_names = sorted(
                            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                                 if var.name.split(':')[0] in saved_shapes])

                        model_var_names_set = set([x[1] for x in model_var_names])
                        ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                        missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                        if len(missing_in_ckpt) > 0:
                            stderr(
                                'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))
                        missing_in_model = ckpt_var_names_set - model_var_names_set
                        if len(missing_in_model) > 0:
                            stderr(
                                'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))

                        restore_vars = []
                        name2var = dict(
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

                        with tf.variable_scope('', reuse=True):
                            for var_name, saved_var_name in ckpt_var_names:
                                curr_var = name2var[saved_var_name]
                                var_shape = curr_var.get_shape().as_list()
                                if var_shape == saved_shapes[saved_var_name]:
                                    restore_vars.append(curr_var)

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

                        saver_tmp.restore(self.sess, path)
                    else:
                        raise err




    ############################################################
    # Public methods
    ############################################################

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None):

        assert not self.predict_mode, 'Cannot save while in predict mode, since this would overwrite the parameters with their moving averages.'

        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except Exception:
                        stderr('Write failure during save. Retrying...\n')
                        time.sleep(1)
                        i += 1
                if i >= 10:
                    stderr('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a DNN-Seg checkpoint and/or initialize the DNN-Seg model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                    tf.tables_initializer().run()
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
                else:
                    if predict:
                        stderr('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    reload = mode != self.predict_mode
                    if reload:
                        self.load(predict=mode)

                self.predict_mode = mode

    def report_settings(self, indent=0):
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        for kwarg in SYN_SEM_NET_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def report_parseval(self, parseval, indent=0):
        out = ''
        for e in ('syn', 'sem'):
            out += ' ' * indent + 'Parse eval (%s):\n' % e

            k = 'f_' + e
            if k in parseval:
                v = parseval[k]
            else:
                v = 0.
            out += ' ' * (indent + 2) + 'Brac F:   %.4f\n' % v

            k = 'p_' + e
            if k in parseval:
                v = parseval[k]
            else:
                v = 0.
            out += ' ' * (indent + 2) + 'Brac P:   %.4f\n' % v

            k = 'r_' + e
            if k in parseval:
                v = parseval[k]
            else:
                v = 0.
            out += ' ' * (indent + 2) + 'Brac R:   %.4f\n' % v

            k = 'tag_' + e
            if k in parseval:
                v = parseval[k]
            else:
                v = 0.
            out += ' ' * (indent + 2) + 'Tag Acc:  %.4f\n' % v

        return out

    def update_logs(self, info_dict, name='train', encoder='syn', task='parsing', update_type='loss'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not isinstance(name, list):
                    name = [name]
                if not isinstance(encoder, list):
                    encoder = [encoder]
                if not isinstance(task, list):
                    task = [task]
                if not isinstance(update_type, list):
                    update_type = [update_type]

                for n in name:
                    for e in encoder:
                        writer = getattr(self, '%s_%s_writer' % (n, e))
                        for t in task:
                            for u in update_type:
                                fd_summary = {}
                                # TODO: Cory, add WP and BOW tasks
                                if t.lower() == 'parsing':
                                    if u.lower() == 'loss':
                                        log_summaries = self.parsing_loss_log_summaries
                                        summary = self.parsing_loss_summary
                                    elif u.lower() == 'eval':
                                        log_summaries = self.parsing_eval_log_summaries
                                        summary = self.parsing_eval_summary
                                    else:
                                        raise ValueError('Unrecognized update_type "%s".' % u)
                                elif t.lower() == 'wp':
                                    if u.lower() == 'loss':
                                        log_summaries = self.wp_loss_log_summaries
                                        summary = self.wp_loss_summary
                                    elif u.lower() == 'eval':
                                        log_summaries = self.wp_eval_log_summaries
                                        summary = self.wp_eval_summary
                                    else:
                                        raise ValueError('Unrecognized update_type "%s".' % u)
                                elif t.lower() == 'sts':
                                    if u.lower() == 'loss':
                                        log_summaries = self.sts_loss_log_summaries
                                        summary = self.sts_loss_summary
                                    elif u.lower() == 'eval':
                                        log_summaries = self.sts_eval_log_summaries
                                        summary = self.sts_eval_summary
                                    else:
                                        raise ValueError('Unrecognized update_type "%s".' % u)
                                else:
                                    raise ValueError('Unrecognized task "%s".' % t)

                                for k in log_summaries:
                                    fd_summary[log_summaries[k]] = info_dict[k + '_' + e]

                                summary_out = self.sess.run(summary, feed_dict=fd_summary)
                                writer.add_summary(summary_out, self.global_batch_step.eval(session=self.sess))

    def report_n_params(self, indent=0):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
                out = ' ' * indent + 'TRAINABLE PARAMETERS:\n'
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
                    cur_params = np.prod(np.array(v_val).shape)
                    n_params += cur_params
                    out += ' ' * indent + '  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params)
                out += ' ' * indent + '  TOTAL: %d\n\n' % n_params

                return out

    def evaluate(
            self,
            data,
            gold_tree_path=None,
            data_name='dev',
            n_print=5,
            update_logs=True,
            verbose=True
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if verbose:
                    stderr('-' * 50 + '\n')
                    stderr('%s set evaluation\n\n' % data_name.upper())
        
                info_dict = self._run_batches(
                    data,
                    data_name=data_name,
                    minibatch_size=self.eval_minibatch_size,
                    update=False,
                    randomize=False,
                    return_syn_parsing_losses=True,
                    return_sem_parsing_losses=True,
                    return_syn_parsing_prediction=True,
                    return_sem_parsing_prediction=True,
                    return_syn_wp_loss=True,
                    return_sem_wp_loss=True,
                    return_syn_wp_prediction=True,
                    return_sem_wp_prediction=True,
                    return_syn_sts_loss=True,
                    return_sem_sts_loss=True,
                    return_syn_sts_prediction=True,
                    return_sem_sts_prediction=True,
                    verbose=verbose
                )
        
                if verbose:
                    stderr('Evaluating...\n')

                eval_metrics = self.eval_predictions(
                    data,
                    info_dict,
                    gold_tree_path=gold_tree_path,
                    syn=True,
                    sem=True,
                    name=data_name
                )
        
                if gold_tree_path is None:
                    task = ['wp', 'sts']
                else:
                    task = ['parsing', 'wp', 'sts']

                info_dict.update(eval_metrics)
        
                if update_logs:
                    self.update_logs(
                        info_dict,
                        name=data_name,
                        encoder=self.REP_TYPES,
                        task=task,
                        update_type=['loss', 'eval']
                    )
        
                if verbose:
                    parse_samples = data.pretty_print_parse_predictions(
                        text=info_dict['parsing_text'][:n_print],
                        pos_label_true=info_dict['pos_label_true'][:n_print],
                        pos_label_pred=info_dict['pos_label_prediction_syn'][:n_print],
                        parse_label_true=info_dict['parse_label_true'][:n_print],
                        parse_label_pred=info_dict['parse_label_prediction_syn'][:n_print],
                        parse_depth_true=info_dict['parse_depth_true'][:n_print] if self.factor_parse_labels else None,
                        parse_depth_pred=info_dict['parse_depth_prediction_syn'][:n_print] if self.factor_parse_labels else None,
                        mask=info_dict['parsing_text_mask'][:n_print]
                    )
                    stderr('Sample parsing prediction:\n\n' + parse_samples)

                    if self.wp_recurrent_decoder:
                        wp_samples = data.pretty_print_wp_predictions(
                            text=info_dict['wp_parsing_true'][:n_print],
                            pred=info_dict['wp_parsing_prediction_syn'][:n_print],
                            mask=np.any(info_dict['parsing_text_mask'][:n_print], axis=-1)
                        )
                        stderr('Sample word position prediction:\n\n' + wp_samples)
        
                    sts_samples = data.pretty_print_sts_predictions(
                        s1=info_dict['sts_s1_text'][:n_print],
                        s1_mask=info_dict['sts_s1_text_mask'][:n_print],
                        s2=info_dict['sts_s2_text'][:n_print],
                        s2_mask=info_dict['sts_s2_text_mask'][:n_print],
                        sts_true=info_dict['sts_true'][:n_print],
                        sts_pred=info_dict['sts_prediction_sem'][:n_print]
                    )
                    stderr('Sample STS prediction:\n\n' + sts_samples)
        
                    if gold_tree_path is not None:
                        stderr(self.report_parseval(info_dict))

                    stderr('\n')

                    stderr('WP acc (syn): %.4r\n' % info_dict['wp_acc_syn'])
                    stderr('WP acc (sem): %.4r\n\n' % info_dict['wp_acc_sem'])

                    stderr('STS Pearson correlation (sem): %.4f\n' % info_dict['sts_r_sem'])
                    stderr('STS Pearson correlation (syn): %.4f\n\n' % info_dict['sts_r_syn'])

    def fit(
            self,
            data,
            n_minibatch,
            train_gold_tree_path=None,
            dev_gold_tree_path=None,
            n_print=5,
            run_initial_eval=False,
            verbose=True
    ):
        if self.global_batch_step.eval(session=self.sess) == 0:
            if verbose:
                stderr('Saving initial weights...\n')
            self.save()

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)

        if verbose:
            stderr('*' * 100 + '\n')
            stderr(self.report_settings())
            stderr('\n')
            stderr(self.report_n_params())
            stderr('\n')
            stderr('*' * 100 + '\n\n')

        data_feed = data.get_parsing_data_feed(
            'train',
            minibatch_size=32,
            randomize=False
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if run_initial_eval and self.global_batch_step.eval(session=self.sess) == 0:
                    self.evaluate(
                        data,
                        gold_tree_path=train_gold_tree_path,
                        data_name='train',
                        n_print=n_print,
                        update_logs=True,
                        verbose=verbose
                    )

                    self.evaluate(
                        data,
                        gold_tree_path=dev_gold_tree_path,
                        data_name='dev',
                        n_print=n_print,
                        update_logs=True,
                        verbose=verbose
                    )

                i = self.global_batch_step.eval(session=self.sess)

                while i < n_minibatch:
                    n_minibatch_cur, save, log, evaluate = self._get_n_batches(i)

                    t0_iter = time.time()
                    if verbose:
                        stderr('-' * 50 + '\n')
                        stderr('Updating on training set...\n')

                    info_dict_train = self._run_batches(
                        data,
                        data_name='train',
                        minibatch_size=self.minibatch_size,
                        n_minibatch=n_minibatch_cur,
                        update=True,
                        randomize=True,
                        return_syn_parsing_losses=True,
                        return_sem_parsing_losses=True,
                        return_syn_wp_loss=True,
                        return_sem_wp_loss=True,
                        return_syn_sts_loss=True,
                        return_sem_sts_loss=True,
                        verbose=verbose
                    )

                    if save:
                        self.save()

                    if log:
                        self.update_logs(
                            info_dict_train,
                            name='train',
                            encoder=self.REP_TYPES,
                            task=['parsing', 'wp', 'sts'],
                            update_type='loss'
                        )

                    if evaluate:
                        self.evaluate(
                            data,
                            gold_tree_path=dev_gold_tree_path,
                            data_name='dev',
                            n_print=n_print,
                            update_logs=True,
                            verbose=verbose
                        )

                    if verbose:
                        t1_iter = time.time()
                        time_str = pretty_print_seconds(t1_iter - t0_iter)
                        stderr('Processing time: %s\n' % time_str)

                    i = self.global_batch_step.eval(session=self.sess)

    # TODO: Add STS prediction
    def predict(
            self,
            data,
            data_name='dev',
            from_syn=True,
            from_sem=True,
            verbose=True,
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                info_dict = self._run_batches(
                    data,
                    data_name=data_name,
                    minibatch_size=self.eval_minibatch_size,
                    update=False,
                    randomize=False,
                    return_syn_parsing_losses=False,
                    return_sem_parsing_losses=False,
                    return_syn_parsing_prediction=from_syn,
                    return_sem_parsing_prediction=from_sem,
                    verbose=verbose
                )
                return info_dict

    def predict_parses(
            self,
            data,
            data_name='dev',
            from_syn=True,
            from_sem=True,
            verbose=True,
    ):
        info_dict = self._run_batches(
            data,
            data_name=data_name,
            minibatch_size=self.eval_minibatch_size,
            update=False,
            randomize=False,
            return_syn_parsing_losses=False,
            return_sem_parsing_losses=False,
            return_syn_parsing_prediction=from_syn,
            return_sem_parsing_prediction=from_sem,
            verbose=verbose
        )

    def get_parse_seqs(
            self,
            data,
            info_dict,
            syn=True,
            sem=True
    ):
        parse_seqs = {}
        encoder = []
        if syn:
            encoder.append('syn')
        if sem:
            encoder.append('sem')
        for e in encoder:
            if not e in parse_seqs:
                parse_seqs[e] = {}
            numeric_chars = info_dict['parsing_text']
            numeric_pos = info_dict['pos_label_prediction_syn']
            numeric_parse_label = info_dict['parse_label_prediction_syn']
            mask = info_dict['parsing_text_mask']
            if self.factor_parse_labels:
                numeric_depth = info_dict['parse_depth_prediction_syn']
            else:
                numeric_depth = None

            seqs = data.parse_predictions_to_sequences(
                numeric_chars,
                numeric_pos,
                numeric_parse_label,
                numeric_depth=numeric_depth,
                mask=mask
            )

            parse_seqs[e] = seqs

        return parse_seqs

    def get_parse_trees(
            self,
            data,
            info_dict,
            syn=True,
            sem=True
    ):
        parse_trees = {}
        encoder = []
        if syn:
            encoder.append('syn')
        if sem:
            encoder.append('sem')
        for e in encoder:
            if not e in parse_trees:
                parse_trees[e] = {}
            numeric_chars = info_dict['parsing_text']
            numeric_pos = info_dict['pos_label_prediction_%s' % e]
            numeric_parse_label = info_dict['parse_label_prediction_%s' % e]
            mask = info_dict['parsing_text_mask']
            if self.factor_parse_labels:
                numeric_depth = info_dict['parse_depth_prediction_%s' % e]
            else:
                numeric_depth = None

            trees = data.parse_predictions_to_trees(
                numeric_chars,
                numeric_pos,
                numeric_parse_label,
                numeric_depth=numeric_depth,
                mask=mask,
                add_os=not self.os
            )

            parse_trees[e] = trees

        return parse_trees

    def print_parse_trees(
            self,
            data,
            info_dict,
            syn=True,
            sem=True,
            outdir=None,
            name=None
    ):
        if outdir is None:
            outdir = self.outdir
        if outdir[-1] != '/':
            outdir += '/'

        trees = self.get_parse_trees(
            data,
            info_dict,
            syn=syn,
            sem=sem
        )

        encoder = []
        if syn:
            encoder.append('syn')
        if sem:
            encoder.append('sem')
        for e in encoder:
            trees_cur = '\n'.join(trees[e])
            if name is not None:
                pred_path = outdir + name + '_parsed_trees_%s.txt' % e
            else:
                pred_path = outdir + 'parsed_trees_%s.txt' % e

            with open(pred_path, 'w') as f:
                f.write(trees_cur)

    def eval_trees(
            self,
            goldpath,
            syn=True,
            sem=True,
            outdir=None,
            name=None
    ):
        if outdir is None:
            outdir = self.outdir
        if outdir[-1] != '/':
            outdir += '/'

        evalb_path = os.path.abspath('tree2labels/EVALB/evalb')

        if os.path.exists(goldpath):
            parseval = {}
            encoder = []
            if syn:
                encoder.append('syn')
            if sem:
                encoder.append('sem')
            for e in encoder:
                if name is not None:
                    pred_path = outdir + name + '_parsed_trees_%s.txt' % e
                    eval_path = outdir + name + '_parseval_%s.txt' % e
                else:
                    pred_path = outdir + 'parsed_trees_%s.txt' % e
                    eval_path = outdir + 'parseval_%s.txt' % e

                exit_status = os.system('%s %s %s > %s' % (evalb_path, goldpath, pred_path, eval_path))
                
                assert exit_status == 0, 'Call to EVALB failed. See above for traceback.'

                parseval[e] = get_evalb_scores(eval_path)

            out = {}
            for e in parseval:
                for m in parseval[e]['all']:
                    out['_'.join((m, e))] = parseval[e]['all'][m]

        else:
            stderr('Path to gold trees provided (%s) does not exist.')
            out = {}

        return out

    def eval_parse_predictions(
            self,
            data,
            info_dict,
            goldpath,
            syn=True,
            sem=True,
            outdir=None,
            name=None
    ):
        self.print_parse_trees(
            data,
            info_dict,
            syn=syn,
            sem=sem,
            outdir=outdir,
            name=name
        )

        out = self.eval_trees(
            goldpath,
            syn=syn,
            sem=sem,
            outdir=outdir,
            name=name
        )

        return out

    def eval_wp_predictions(
            self,
            info_dict,
            syn=True,
            sem=True
    ):
        encoder = []
        if syn:
            encoder.append('syn')
        if sem:
            encoder.append('sem')

        out = {}
        for e in encoder:
            parsing_true = info_dict['wp_parsing_true']
            parsing_mask = np.any(info_dict['parsing_text_mask'], axis=-1)
            parsing_pred = info_dict['wp_parsing_prediction_%s' % e]
            
            sts_s1_true = info_dict['wp_sts_s1_true']
            sts_s1_mask = np.any(info_dict['sts_s1_text_mask'], axis=-1)
            sts_s1_pred = info_dict['wp_sts_s1_prediction_%s' % e]

            sts_s2_true = info_dict['wp_sts_s2_true']
            sts_s2_mask = np.any(info_dict['sts_s2_text_mask'], axis=-1)
            sts_s2_pred = info_dict['wp_sts_s2_prediction_%s' % e]

            parsing_correct = (np.equal(parsing_true, parsing_pred) * parsing_mask).sum()
            sts_s1_correct = (np.equal(sts_s1_true, sts_s1_pred) * sts_s1_mask).sum()
            sts_s2_correct = (np.equal(sts_s2_true, sts_s2_pred) * sts_s2_mask).sum()

            n_correct = parsing_correct + sts_s1_correct + sts_s2_correct
            n_total = parsing_mask.sum() + sts_s1_mask.sum() + sts_s2_mask.sum()

            acc = (n_correct / np.maximum(n_total, self.epsilon)) * 100

            out['wp_acc_%s' % e] = acc

        return out

    def eval_sts_predictions(
            self,
            info_dict,
            syn=True,
            sem=True
    ):
        encoder = []
        if syn:
            encoder.append('syn')
        if sem:
            encoder.append('sem')

        out = {}
        for e in encoder:
            r = np.corrcoef(info_dict['sts_true'], info_dict['sts_prediction_%s' % e])[1, 0]
            out['sts_r_%s' % e] = r

        return out

    def eval_predictions(
            self,
            data,
            info_dict,
            gold_tree_path=None,
            syn=True,
            sem=True,
            outdir=None,
            name=None
    ):
        out = {}
        if gold_tree_path is not None:
            out.update(self.eval_parse_predictions(
                    data,
                    info_dict,
                    gold_tree_path,
                    syn=syn,
                    sem=sem,
                    outdir=outdir,
                    name=name
            ))
        out.update(self.eval_wp_predictions(
            info_dict,
            syn=syn,
            sem=syn,
        ))
        out.update(self.eval_sts_predictions(
            info_dict,
            syn=syn,
            sem=syn,
        ))

        return out
    
    def print_parse_seqs(
            self,
            data,
            info_dict,
            syn=True,
            sem=True,
            outdir=None,
            name=None
    ):
        if outdir is None:
            outdir = self.outdir
        if outdir[-1] != '/':
            outdir += '/'

        seqs = self.get_parse_seqs(
            data,
            info_dict,
            syn=syn,
            sem=sem
        )

        encoder = []
        if syn:
            encoder.append('syn')
        if sem:
            encoder.append('sem')

        for e in encoder:
            if name is not None:
                pred_path = outdir + name + '_parsed_seqs_%s.txt' % e
            else:
                pred_path = outdir + '_parsed_seqs_%s.txt' % e

            with open(pred_path, 'w') as f:
                f.write(seqs[e])



