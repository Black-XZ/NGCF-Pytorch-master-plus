'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain (train from scratch), 1: Load pretrained model and test, 2: Continue training from pretrained model')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')

    parser.add_argument('--gpu_id', type=int, default=6)

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--valid_flag', type=int, default=0,
                        help='0: 不使用验证集, 1: 使用验证集做早停和模型选择')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值：连续多少个epoch验证集表现没有提升就停止训练')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--rating_file', nargs='?', default='',
                        help='Path to rating jsonl file (e.g., ../Books_10core.jsonl). If provided, ratings will be used as edge weights.')
    parser.add_argument('--sentiment_file', nargs='?', default='',
                        help='Path to sentiment jsonl file. If provided, sentiment scores will be used to adjust edge weights.')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Sentiment weight coefficient: R\' = rating * (1 + alpha * sentiment). Recommended range: 0.2-0.5')
    parser.add_argument('--pretrain_path', nargs='?', default='',
                        help='Path to pretrained model weights (e.g., model/9.pkl). Used when pretrain=1.')
    return parser.parse_args()
