"""

Main function to call flair.
"""
import argparse
import configparser
import datetime
import os
from typing import List

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, WordEmbeddings, \
    BertEmbeddings, ELMoEmbeddings, FlairEmbeddings, CharacterEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import SGD, Adam


def get_corpus(config):
    columns = {0: 'text', 1: 'ner'}
    corpus: Corpus = ColumnCorpus(
        config['corpus']['dir'],
        columns,
        train_file=config['corpus']['train_file'],
        dev_file=config['corpus']['dev_file'],
        test_file=config['corpus']['test_file'],
    ).downsample(float(config['corpus']['downsample']))
    return corpus


def get_tagger(config, corpus, embeddings):
    tag_type = config['tagger']['tag_type']
    tag_dictionary = corpus.make_tag_dictionary(tag_type)
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=int(config['tagger']['hidden_size']),
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=config['tagger']['use_crf']
    )
    return tagger


def get_optimizer(config):
    # TODO other optimizers?
    optimizer_dict = {
        'SGD': SGD,
        'Adam': Adam
    }
    return optimizer_dict[config['trainer']['optimizer']]


def get_trainer(config, corpus, tagger):
    trainer: ModelTrainer = ModelTrainer(
        tagger,
        corpus,
        optimizer=get_optimizer(config)
    )
    return trainer


def get_embeddings(config):
    embedding_types: List[TokenEmbeddings] = []
    if config['embeddings']['char']:
        embedding_types.append(CharacterEmbeddings(
            hidden_size_char=int(config['embeddings']['hidden_size_char']),
            char_embedding_dim=int(config['embeddings']['char_embedding_dim'])))
    if config['embeddings']['word']:  # TODO other than glove
        embedding_types.append(WordEmbeddings(config['embeddings']['word']))
    if config['embeddings']['bert']:
        embedding_types.append(BertEmbeddings(config['embeddings']['bert']))
    if config['embeddings']['elmo']:
        embedding_types.append(ELMoEmbeddings(config['embeddings']['elmo']))
    if config['embeddings']['flair']:
        for i in config['embeddings']['flair'].strip().split():
            embedding_types.append(FlairEmbeddings(i))

    embeddings: StackedEmbeddings = StackedEmbeddings(
        embeddings=embedding_types)

    return embeddings


def train(config, trainer):
    trainer.train(
        config['trainer']['dir'],
        learning_rate=float(config['trainer']['learning_rate']),
        mini_batch_size=int(config['trainer']['mini_batch_size']),
        eval_mini_batch_size=int(config['trainer']['eval_mini_batch_size']),
        max_epochs=int(config['trainer']['max_epochs']),
        anneal_factor=float(config['trainer']['anneal_factor']),
        patience=int(config['trainer']['patience']),
        min_learning_rate=float(config['trainer']['min_learning_rate']),
        train_with_dev=False,
        monitor_train=config['trainer']['monitor_train'],
        monitor_test=config['trainer']['monitor_test'],
        embedding_storage_mode=config['trainer']['embedding_storage_mode'],
        checkpoint=True,
        save_final_model=True,
        anneal_with_restarts=True,
        shuffle=True,
        param_selection_mode=False,
        # num_workers=6,
        # sampler=None,
        summary_dir=config['trainer']['dir'],
    )


# TODO add find learning rate


def tune_hyperparameter(corpus):
    # tune hyperparameters, hard-code for now
    from hyperopt import hp
    from flair.hyperparameter.param_selection import SearchSpace, Parameter

    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
        [WordEmbeddings('glove')],
        [WordEmbeddings('glove'), CharacterEmbeddings()],
        [BertEmbeddings('bert-base-cased')],
        [BertEmbeddings('bert-base-uncased')],
        [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]
    ])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[64, 128,
                                                                256]),
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2]),
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5),
    search_space.add(Parameter.LEARNING_RATE, hp.choice,
                     options=[0.05, 0.1, 0.15, 0.2]),
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice,
                     options=[16, 32, 64, 128, 256])

    from flair.hyperparameter.param_selection import \
        SequenceTaggerParamSelector, OptimizationValue

    param_selector = SequenceTaggerParamSelector(
        corpus,
        'ner',
        'resources/results',
        max_epochs=50,
        training_runs=3,
        optimization_value=OptimizationValue.DEV_SCORE
    )

    param_selector.optimize(search_space)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with Flair')
    parser.add_argument('--config', help='Configuration File',
                        default='config.ini')
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(args.config)
    config.set('trainer', 'dir',
               'trainer_' + str(datetime.datetime.now()).replace(' ', '_'))
    corpus = get_corpus(config)

    # hyperopt hyperparameter tuning
    if config['trainer']['hyperopt']:
        tune_hyperparameter(corpus)
    else:

        # write all configs to trainer dir
        os.mkdir(config['trainer']['dir'])
        with open(os.path.join(config['trainer']['dir'], 'config.ini'),
                  'w') as f:
            config.write(f)

        embeddings = get_embeddings(config)
        tagger = get_tagger(config, corpus, embeddings)
        trainer = get_trainer(config, corpus, tagger)
        train(config, trainer)
