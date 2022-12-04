import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['WANDB_DISABLED'] = 'true'
from retweet_prediction_challenge.code.dataset import get_custom_dataset
from retweet_prediction_challenge.code.model import CamembertRegressor

dataset = get_custom_dataset()

training_args = dict(output_dir="/shared/personal/vladtom/exp2",
                     label_names=['labels'],
                     learning_rate=1e-4,
                     save_total_limit=20,
                     fp16=True,
                     per_device_train_batch_size=38,
                     gradient_accumulation_steps=4,
                     num_train_epochs=2,
                     logging_strategy='steps',
                     evaluation_strategy='steps',
                     eval_steps=10000,
                     )

model = CamembertRegressor(training_args)
model.train(dataset)
