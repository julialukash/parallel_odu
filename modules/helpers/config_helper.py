import configparser
import datetime
from os import path, mkdir


class ConfigPaths(object):
    """Config class for input
    Auxiliary class for keeping input parameters of project

    Args:
        file_name: Name of input config file (must be in wd)
    """

    def __init__(self, file_name):
        if not path.exists(file_name):
            raise IOError('Input config file not found')

        cfg = configparser.ConfigParser()
        cfg.read(file_name)
        path_cfg = cfg['Paths']
        if path_cfg is None:
            raise AttributeError('Section <<Paths>> is not found into config file')

        home_dir = path_cfg.get('home_dir')
        if home_dir is None:
            raise AttributeError('Field <<home_dir>> is not found into <<Paths>> section')

        self.collection_name = path_cfg.get('collection_name')
        if self.collection_name is None:
            raise AttributeError('Field <<collection_name>> is  not found into <<Paths>> section')

        self.dataset_folder_name = path_cfg.get('dataset_folder_name')
        if self.dataset_folder_name is None:
            raise AttributeError('Field <<dataset_folder_name>> is  not found into <<Paths>> section')

        self.experiment_folder_name = path_cfg.get('experiment_folder_name')
        if self.experiment_folder_name is None:
            raise AttributeError('Field <<experiment_folder_name>> is  not found into <<Paths>> section')

        datase_rel_path = '..\\data\postnauka\\UCI_collections'
        self.dataset_path = path.join(home_dir, datase_rel_path, self.dataset_folder_name)
        if not path.exists(self.dataset_path):
            raise SystemError('Path ' + self.dataset_path + ' not found')
        self.vocabulary_path = path.join(self.dataset_path, 'vocab.' + self.collection_name + '.txt')
        if not path.isfile(self.vocabulary_path):
            raise SystemError('Vocabulary file ' + self.vocabulary_path + ' not found')

        output_batches_rel_dir = '..\\data\postnauka\\bigARTM_files'
        self.output_batches_path = path.join(home_dir, output_batches_rel_dir, self.dataset_folder_name)
        if not path.exists(self.output_batches_path):
            mkdir(self.output_batches_path)
        self.dictionary_path = path.join(home_dir, output_batches_rel_dir, self.dataset_folder_name, self.collection_name + '_dictionary')

        output_experiments_rel_dir = 'experiments'
        self.experiment_data_path = path.join(home_dir, output_experiments_rel_dir)
        if not path.exists(self.experiment_data_path ):
            mkdir(self.experiment_data_path)
        self.experiment_dataset_folder_name = path.join(self.experiment_data_path, self.dataset_folder_name)
        if not path.exists(self.experiment_dataset_folder_name):
            mkdir(self.experiment_dataset_folder_name)
        self.experiment_path = path.join(self.experiment_dataset_folder_name, self.experiment_folder_name)
        if not path.exists(self.experiment_path):
            mkdir(self.experiment_path)

        self.models_file_name = path.join(self.experiment_path, 'models.txt')

        self.models_archive_path = path.join(self.experiment_path, 'archive')
        if not path.exists(self.models_archive_path):
            mkdir(self.models_archive_path)

        
    # def __str__(self):
    #     return 'data_folder_path = {}, \ninput_folder_path = {}, \noutput_folder_path = {},\n' + \
    #            'pics_folder_path = {}, \nlog_file_path = {},\nlog_file_name = {}'\
    #             .format(self.data_folder_path, self.input_folder_path, self.output_folder_path,
    #                     self.pics_folder_path, self.log_folder_path, self.log_file_name)


