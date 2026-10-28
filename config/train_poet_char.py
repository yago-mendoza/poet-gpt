# character-level "mid" model on a poet's posts, trained on a laptop CPU
# ~4.75M parameters, 3000 steps x 8192 tokens = ~25M tokens (~7.6 epochs)

out_dir = 'out-poet-char'
eval_interval = 250
eval_iters = 50  # the default 200 would add ~25 min of evaluation on a CPU
log_interval = 10
always_save_checkpoint = False  # save only when val loss improves

wandb_log = False
dataset = 'poet_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256  # 256 characters of context

n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.2

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cpu'
compile = False
dtype = 'float32'
