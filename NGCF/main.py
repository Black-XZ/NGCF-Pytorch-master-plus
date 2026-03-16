'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time


if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    # 预训练模式：加载预训练权重并直接测试
    if args.pretrain == 1:
        if args.pretrain_path and os.path.exists(args.pretrain_path):
            print(f"\n=== 预训练模式：加载预训练模型 {args.pretrain_path} ===")
            model.load_state_dict(torch.load(args.pretrain_path))
            print("模型加载成功！")
        else:
            print(f"Error: pretrain=1 但找不到预训练模型: {args.pretrain_path}")
            exit(1)

        # 直接在测试集上评估
        print("\n=== 直接在测试集上评估 ===")
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        print("\n=== 测试集评估结果 ===")
        perf_str = 'Test: recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (
            ret['recall'][0], ret['recall'][-1],
            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
            ret['ndcg'][0], ret['ndcg'][-1])
        print(perf_str)
        exit(0)

    # 继续训练模式：在已有预训练参数基础上继续训练
    if args.pretrain == 2:
        if args.pretrain_path and os.path.exists(args.pretrain_path):
            print(f"\n=== 继续训练模式：加载预训练模型 {args.pretrain_path} ===")
            model.load_state_dict(torch.load(args.pretrain_path))
            print("预训练模型加载成功，继续训练...")
        else:
            print(f"Error: pretrain=2 但找不到预训练模型: {args.pretrain_path}")
            exit(1)

    # 正常训练模式
    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    valid_rec_loger = []

    # 检查是否使用验证集
    use_valid = args.valid_flag == 1 and hasattr(data_generator, 'valid_set') and len(data_generator.valid_set) > 0
    if use_valid:
        print("=== 使用验证集做早停和模型选择 ===")
    else:
        print("=== 使用测试集评估 ===")

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()

        # 选择评估数据集：验证集或测试集
        if use_valid:
            users_to_test = list(data_generator.valid_set.keys())
            ret = valid(model, users_to_test, drop_flag=False)
            eval_type = "Valid"
        else:
            users_to_test = list(data_generator.test_set.keys())
            ret = test(model, users_to_test, drop_flag=False)
            eval_type = "Test"

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], %s recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, eval_type,
                        ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        # 早停逻辑
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=args.patience)

        # 保存最佳模型
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the best weights in path: ', args.weights_path + str(epoch) + '.pkl')

        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

    # 如果使用验证集，最后在测试集上评估
    if use_valid:
        print("\n=== 训练结束，使用最佳模型在测试集上评估 ===")
        # 加载最佳模型（rec_loger 每项是数组，需用 recall[0] 找最佳 epoch）
        if args.save_flag == 1:
            recs_arr = np.array(rec_loger)
            best_idx = int(np.argmax(recs_arr[:, 0]))
            # 评估每 10 个 epoch 做一次，首次在 epoch 9，故 epoch = 9 + best_idx * 10
            best_epoch = 9 + best_idx * 10
            best_model_path = args.weights_path + str(best_epoch) + '.pkl'
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from: {best_model_path}")

        # 在测试集上评估
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        print("\n=== 最终测试集结果 ===")
        perf_str = 'Test: recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (
            ret['recall'][0], ret['recall'][-1],
            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
            ret['ndcg'][0], ret['ndcg'][-1])
        print(perf_str)

        # 更新日志用于最终输出
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)