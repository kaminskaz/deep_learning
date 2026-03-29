model_names = [
    "resnet50",
    "xception",
    "efficientnetb4"
]
batch_sizes = [64, 128, 256]
optimizers = ["sgd", "adam"]
learning_rates = [0.1, 0.01, 0.001]

for model_name in model_names:
    for batch_size in batch_sizes:
        for optimizer in optimizers:
            for learning_rate in learning_rates:
                yaml_schema = f"""
# model
model: "{model_name}"          # resnet50, xception, efficientnetb4
num_classes: 10
dropout_rate: 0.0

# training
num_epochs: 30
batch_size: {batch_size}
patience: 5
seed: 42
device: "cuda"              # cpu, cuda, mps

# optimizer
optimizer: "{optimizer}"           # sgd, adam
learning_rate: {learning_rate}
momentum: 0.9              # only for sgd

# scheduler
scheduler: "cosine"        # cosine

# data
train_dir: "./data/train"
val_dir:   "./data/valid"

# transforms
preset: "none"           # none, light, medium, heavy 
augmentor_config: "None" #"code/configs/augmentor.yaml" # None if no augmentor

# paths
model_path: "./saved_models"
"""
                filename = f"experiment_1_{model_name}_bs{batch_size}_{optimizer}_lr{learning_rate}.yaml"
                with open(filename, 'w') as f:
                    f.write(yaml_schema.strip())
                print(f"Generated {filename}")