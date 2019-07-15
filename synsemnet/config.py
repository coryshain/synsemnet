import sys
import os
import shutil
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from .kwargs import SYN_SEM_NET_KWARGS


class Config(object):
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)

        # Data
        data = config['data']
        self.train_data_dir = data.get('train_data_dir', './')
        self.val_data_dir = data.get('val_data_dir', './')
        self.test_data_dir = data.get('test_data_dir', './')

        # SETTINGS
        # Output directory
        settings = config['settings']
        self.outdir = settings.get('outdir', None)
        if self.outdir is None:
            self.outdir = settings.get('logdir', None)
        if self.outdir is None:
            self.outdir = './synsemnet_model/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if os.path.realpath(path) != os.path.realpath(self.outdir + '/config.ini'):
            shutil.copy2(path, self.outdir + '/config.ini')

        # Process config settings
        self.model_settings = self.build_synsemnet_settings(settings)
        gpu_frac = settings.get('gpu_frac', None)
        if gpu_frac in [None, 'None']:
            gpu_frac = None
        else:
            try:
                gpu_frac = float(gpu_frac)
            except:
                raise ValueError('gpu_frac parameter invalid: %s' % gpu_frac)
        self.model_settings['gpu_frac'] = gpu_frac
        self.model_settings['use_gpu_if_available'] = settings.getboolean('use_gpu_if_available', True)


    def __getitem__(self, item):
        return self.model_settings[item]

    def build_synsemnet_settings(self, settings):
        out = {}

        # Parent class initialization keyword arguments
        out['outdir'] = self.outdir
        for kwarg in SYN_SEM_NET_KWARGS:
            out[kwarg.key] = kwarg.kwarg_from_config(settings)

        return out


