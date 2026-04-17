from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from buildtrain import *
from data_process import *
from model import *
from datetime import datetime


parser = ArgumentParser("main", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--data_path", default="./data/", help="Data sources.")
parser.add_argument("--run_folder", default="./", help="Data sources.")
parser.add_argument("--data_name", default="DB15K", help="Name of the dataset.")

parser.add_argument("--embedding_dim", default=1024, type=int, help="Entity/Relation dimension")
parser.add_argument("--batch_size", default=1024, type=int, help='Batch Size')
parser.add_argument("--device", default='cuda:0', type=str)

parser.add_argument("--opt", default="Adam", type=str)
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=0.0002, type=float)
parser.add_argument("--min_lr", default=1e-8, type=float, help='L2 regularization')
parser.add_argument("--factor", default=0.5, type=float)

parser.add_argument("--patience", default=5, type=int)

parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--emb_reg", default=0.1, type=float)
parser.add_argument("--type_reg", default=0.0001, type=float)
parser.add_argument("--save_name", default='wn18rr.pt')
# Logging parameters
parser.add_argument("--name", default='runlog_' + str(datetime.now().strftime("%Y-%m-%d_%H-%M")), help='Name of the experiment')
parser.add_argument('--logdir', dest="log_dir", default='./log/', help='Log directory')
parser.add_argument('--config', dest="config_dir", default='./config/', help='Config directory')
parser.add_argument('--seed', type=int, dest="seed", default='0', help='random seed')
parser.add_argument('--step_size', type=int, dest="step_size", default=50, help='Step size for CosineAnnealingWarmRestarts')




args = parser.parse_args()
setup_seed(args.seed)
cuda_num = int(args.device[-1])
torch.cuda.set_device(cuda_num)

config_dir = args.config_dir
log_dir = args.log_dir
exp_name = args.name
logger = get_logger(exp_name, log_dir, config_dir, args.epoch)
logger.info(vars(args))

if args.data_name == "FB15k-237" or args.data_name == "fb15k-237":
    train, valid, test, entity_indexes, relation_indexes = build_data(path=args.data_path, name=args.data_name)
elif args.data_name == "WN18RR" or args.data_name == "wn18rr":
    train, valid, test, entity_indexes, relation_indexes = build_data(path=args.data_path, name=args.data_name)
elif args.data_name == "DB15K" or args.data_name == "db15k":
    train, valid, test, entity_indexes, relation_indexes = build_DB15K(path=args.data_path, name=args.data_name)
elif args.data_name == "MKG-W" or args.data_name == "MKG-Y":
    train, valid, test, entity_indexes, relation_indexes = build_MKG(path=args.data_path, name=args.data_name)
else:
    logger.info("No such dataset")


train_doubles, valid_doubles, test_doubles, relation_indexes = get_doubles(train, valid, test, relation_indexes)
x_valid = np.array(valid_doubles).astype(np.int32)
x_test = np.array(test_doubles).astype(np.int32)

target_dict = get_target_dict(train_doubles, x_valid, x_test)


model = DVVE(logger, len(entity_indexes), len(relation_indexes),
                embedding_dim=args.embedding_dim, device=args.device
                ).to(args.device)


device = args.device
model.init()


opt = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, args.step_size, T_mult=2)
num_batches_per_epoch = len(train_doubles) // args.batch_size + 1
trainer = Trainer(model, opt, scheduler, train_doubles, x_valid, x_test, args.batch_size,
                num_batches_per_epoch, args.epoch, target_dict, args.device, logger,emb_reg = args.emb_reg, type_reg = args.type_reg)
trainer.train()

final_result = trainer.evaluate(x_test)
mr_final, mrr_final, hit1_final, hit3_final, hit10_final = final_result['mr'], final_result['mrr'], final_result['hits1'],final_result['hits3'], final_result['hits10']
logger.info('[Final result]: MR = {:.4f}, MRR = {:.4f}, Hits@1 = {:.4f}, Hits@3 = {:.4f},Hits@10 = {:.4f}'.format(mr_final, mrr_final, hit1_final, hit3_final,hit10_final))



