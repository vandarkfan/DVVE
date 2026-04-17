import math
import time
import copy
from common import *
from regularizers import N3, Lambda3
import torch




class Trainer:
    def __init__(self, model, optimizer, scheduler, train_data, valid_data, test_data,
                 batch_size, num_batches_per_epoch, num_epochs, target_dict, device, logger, emb_reg,type_reg
                 negative_samples=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_doubles = train_data
        self.train_data = np.array(train_data)
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_batches = num_batches_per_epoch
        self.num_epochs = num_epochs
        self.target_dict = target_dict
        self.device = device
        self.logger = logger
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.best_model = None
        self.best_result = {'mr': -1, 'mrr': -1, 'hits@1': -1, 'hits@3': -1, 'hits@10': -1, 'epoch': -1}

        self.emb_reg = N3(emb_reg)
        self.time_reg = Lambda3(type_reg)

        self.negative_samples_dict = negative_samples


    def train(self):
        start_time = time.time()
        stop_start_epoch = 1
        early_stop_counter = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            random.shuffle(self.train_doubles)
            losses = []

            for batch_num in range(self.num_batches):
                self.optimizer.zero_grad()
                end_idx = min((batch_num + 1) * self.batch_size, len(self.train_doubles))

                x_batch = np.array(
                    self.train_doubles[batch_num * self.batch_size:end_idx]
                )
                batch_h = x_batch[:, 0].astype(np.int32)
                batch_r = x_batch[:, 1].astype(np.int32)
                batch_t = x_batch[:, 2].astype(np.int32)
                batch_o = x_batch[:, 3].astype(np.int32)
                pred,factors,time_a  = self.model.forward(batch_h, batch_r, batch_t,batch_o)
                loss = self.loss_fn(pred, torch.LongTensor(batch_t).to(self.device))
                l_reg = self.emb_reg.forward(factors)
                l_time = self.time_reg.forward(time_a)
                loss = loss + l_reg + l_time

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach().cpu().numpy())
                torch.cuda.empty_cache()

            avg_loss = np.average(losses)
            self.logger.info(f'[Epoch {epoch}]: Train Loss = {avg_loss:.6f}')

            if epoch % 1 == 0:
                valid_result = self._validate_and_log(epoch)

            # Early stopping check
            if epoch >= stop_start_epoch:
                if epoch % 1 == 0:
                    current_result = valid_result
                else:
                    current_result = self.evaluate(self.valid_data)
                if current_result['mrr'] > self.best_result['mrr'] + 1e-6:
                    self.best_result = current_result
                    self.best_result['epoch'] = epoch
                    self.best_model = copy.deepcopy(self.model)
                    # torch.save(self.best_model.state_dict(), 'checkpoint/MKG-W_old_best_model_weights.pth')
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= 20:
                        self.logger.info("Early stopping triggered.")
                        break
                self.scheduler.step(current_result['mrr'])

        end_time = time.time()
        run_time = end_time - start_time
        self._log_runtime(run_time)
        return self.best_model

    def evaluate(self, x_test):
        self.model.eval()
        batch_num = math.ceil(len(x_test) / self.batch_size)
        tail_scores_all = []
        tail_label = []
        with torch.no_grad():
            for i in range(batch_num):
                batch_data = x_test[self.batch_size * i:self.batch_size * (i + 1)]
                batch_h = batch_data[:, 0].astype(np.int32)
                batch_r = batch_data[:, 1].astype(np.int32)
                batch_t = batch_data[:, 2].astype(np.int32)
                batch_o = batch_data[:, 3].astype(np.int32)
                tail_scores,factors,time_a = self.model.forward(batch_h, batch_r, batch_t,batch_o)
                tail_scores = tail_scores.cpu().detach().numpy()
                tail_scores_all.append(tail_scores)

                tail_label.append(batch_t)
                torch.cuda.empty_cache()

        tail_scores_all = np.concatenate(tail_scores_all, axis=0)

        tail_label = np.concatenate(tail_label, axis=0)

        def cal_result(scores, labels, x_test, target_dict):
            ranks = []
            for i in range(len(labels)):
                arr = scores[i]
                mark = labels[i]
                h, r, t,o = x_test[i]
                mark_value = arr[mark]

                ##filter
                targets = target_dict[(h, r, o)]
                for target in targets:
                    target = int(target)
                    if target != mark:
                        arr[target] = np.finfo(np.float32).min
                rank = np.sum(arr > mark_value)
                rank += 1
                ranks.append(rank)

            mr, mrr, hits1, hits3, hits10 = 0, [], [], [], []
            mr = np.average(ranks)

            for rank in ranks:
                mrr.append(1 / rank)
                if rank == 1:
                    hits1.append(1)
                else:
                    hits1.append(0)
                if rank <= 3:
                    hits3.append(1)
                else:
                    hits3.append(0)
                if rank <= 10:
                    hits10.append(1)
                else:
                    hits10.append(0)
            mrr = np.average(mrr)
            hits1 = np.average(hits1)
            hits3 = np.average(hits3)
            hits10 = np.average(hits10)
            result = {'mr': mr, 'mrr': mrr, 'hits1': hits1, 'hits3': hits3, 'hits10': hits10}
            return result

        tail_result = cal_result(tail_scores_all, tail_label, x_test, self.target_dict)
        return {'mr': tail_result['mr'], 'mrr': tail_result['mrr'],
                'hits1': tail_result['hits1'], 'hits3': tail_result['hits3'], 'hits10': tail_result['hits10']}

    def _validate_and_log(self, epoch):
        train_result = self.evaluate(self.train_data)
        self.logger.info(
            f'[Epoch {epoch} - TRAIN]: MR = {train_result["mr"]:.4f}, MRR = {train_result["mrr"]:.4f}, '
            f'Hits@1 = {train_result["hits1"]:.4f}, Hits@3 = {train_result["hits3"]:.4f}, Hits@10 = {train_result["hits10"]:.4f}'
        )
        valid_result = self.evaluate(self.valid_data)
        self.logger.info(
            f'[Epoch {epoch} - VALID]: MR = {valid_result["mr"]:.4f}, MRR = {valid_result["mrr"]:.4f}, '
            f'Hits@1 = {valid_result["hits1"]:.4f}, Hits@3 = {valid_result["hits3"]:.4f}, Hits@10 = {valid_result["hits10"]:.4f}'
        )

        test_result = self.evaluate(self.test_data)
        self.logger.info(
            f'[Epoch {epoch} - TEST] : MR = {test_result["mr"]:.4f}, MRR = {test_result["mrr"]:.4f}, '
            f'Hits@1 = {test_result["hits1"]:.4f}, Hits@3 = {test_result["hits3"]:.4f}, Hits@10 = {test_result["hits10"]:.4f}'
        )
        return valid_result

    def _log_runtime(self, run_time):
        hours = int(run_time // 3600)
        minutes = int((run_time - hours * 3600) // 60)
        seconds = int(run_time - hours * 3600 - minutes * 60)



