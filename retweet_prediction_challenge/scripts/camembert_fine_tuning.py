import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['WANDB_DISABLED'] = 'true'
from retweet_prediction_challenge.code.dataset import get_custom_dataset
from retweet_prediction_challenge.code.model import CamembertRegressor

dataset = get_custom_dataset()

training_args = dict(output_dir="/shared/personal/vladtom/exp4_bs25_acc4_lr1e4",
                     label_names=['labels'],
                     learning_rate=1e-4,
                     save_total_limit=10,
                     fp16=False,
                     per_device_train_batch_size=20,
                     gradient_accumulation_steps=4,
                     num_train_epochs=6,
                     logging_strategy='steps',
                     evaluation_strategy='steps',
                     eval_steps=100,
                     logging_steps=10,
                     )

model = CamembertRegressor(training_args)
model.train(dataset)
