"""

Main function to call flair.
"""
import argparse
import configparser
import datetime
import os
from pathlib import Path
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
        use_crf=config['tagger'].get_boolean('use_crf'),
        dropout=float(config['tagger']['dropout'])
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
        optimizer=get_optimizer(config),
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
        monitor_train=config['trainer'].get_boolean('monitor_train'),
        monitor_test=config['trainer'].get_boolean('monitor_test'),
        embedding_storage_mode=config['trainer'].get_boolean(
            'embedding_storage_mode'),
        checkpoint=True,
        save_final_model=True,
        anneal_with_restarts=config['trainer'].get_boolean(
            'anneal_with_restarts'),
        shuffle=True,  # set True for aggressive lr update
        param_selection_mode=False,
        num_workers=12,  # 12 CPUs per GPU on current instance # TODO
        # sampler=None,
        summary_dir=config['trainer']['dir'],
        early_lr_update=config['trainer'].get_boolean('early_lr_update'),
        early_lr_start_batch=int(config['trainer']['early_lr_start_batch']),
        early_lr_stride_batch=int(config['trainer']['early_lr_stride_batch'])
    )


def tune_hyperparameter(corpus):
    # tune hyperparameters, hard-code for now
    from hyperopt import hp
    from flair.hyperparameter.param_selection import SearchSpace, Parameter
    from flair.hyperparameter.param_selection import \
        SequenceTaggerParamSelector, OptimizationValue

    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
        # StackedEmbeddings([WordEmbeddings('glove')]),
        # StackedEmbeddings([WordEmbeddings('glove'), CharacterEmbeddings()]),
        # StackedEmbeddings([BertEmbeddings('bert-base-cased')]),
        # StackedEmbeddings([BertEmbeddings('bert-base-uncased')]),
        StackedEmbeddings(
            [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])
    ])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[256, 512]),
    # search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2]),
    # search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5),
    search_space.add(Parameter.DROPOUT, hp.choice,
                     options=[0, 0.05, 0.1, 0.15, 0.2, 0.25]),
    search_space.add(Parameter.LEARNING_RATE, hp.choice,
                     options=[0, 0.05, 0.1, 0.15, 0.2, 0.25]),
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice,
                     options=[16, 32, 48, 64, 80])

    param_selector = SequenceTaggerParamSelector(
        corpus,
        'ner',
        'resources/results',
        max_epochs=100,
        training_runs=3,
        optimization_value=OptimizationValue.DEV_SCORE
    )

    param_selector.optimize(search_space)


def find_lr(trainer):
    # find best learning rate
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    learning_rate_tsv = trainer.find_learning_rate(
        config['trainer']['dir'], 'learning_rate.tsv')
    plotter.plot_learning_rate(learning_rate_tsv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with Flair')
    parser.add_argument('--config', help='Configuration File',
                        default='config.ini')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--checkpoint',
                        help='Checkpoint path to load from for resume mode',
                        default=None)
    """Modes:
    train: train a model from scratch
    resume: resume training from given checkpoint
    decode: TODO
    demo: TODO
    hyperopt: hyperparamter-tuning with hyperopt
    find_lr: find best learning rate
    """
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(args.config)
    config.set('trainer', 'dir',
               'trainer_' + str(datetime.datetime.now()).replace(' ', '_'))
    corpus = get_corpus(config)

    if args.mode == 'hyperopt':
        tune_hyperparameter(corpus)
    else:
        # write all configs to trainer dir
        os.mkdir(config['trainer']['dir'])
        with open(os.path.join(config['trainer']['dir'], 'config.ini'),
                  'w') as f:
            config.write(f)

        embeddings = get_embeddings(config)
        tagger = get_tagger(config, corpus, embeddings)

        if args.mode == 'resume':
            trainer = ModelTrainer.load_from_checkpoint(
                tagger.load_checkpoint(Path(args.checkpoint)),
                corpus,
            )
        elif args.mode == 'finetune':
            # TODO
            pass
        else:
            trainer = get_trainer(config, corpus, tagger)

        if args.mode == 'find_lr':
            find_lr(trainer)
        elif args.mode in ['train', 'resume', 'finetune']:
            train(config, trainer)
        elif args.mode == 'decode':
            # TODO
            pass
        elif args.mode == 'demo':
            # TODO
            pass
        else:
            raise NotImplementedError('No such mode as %s!' % args.mode)
