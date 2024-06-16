import warnings
import os
warnings.filterwarnings("ignore")
import time

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling

from functools import lru_cache

def contar_caracteres(archivo, max_block_size=128):
    with open(archivo, 'r', encoding='utf-8') as file:
        text = file.read()
    
    word_count = len(text.split())
    block_size = min(word_count, max_block_size)
    return block_size

def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = contar_caracteres(file_path),
        cache_dir="C:/Users/User/TP1/Cache",
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

# Label 
label = 61

model_name = f'Models/Label {label}'
output_dir = f'C:/Users/User/TP1/Models/Label {label}'
overwrite_output_dir = True
per_device_train_batch_size = 1
num_train_epochs = 1.0

path = f"Data/Dato_Bloques/Label {label}"
archivos = [archivo for archivo in os.listdir(path)]

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

data_collator = load_data_collator(tokenizer)

training_args = TrainingArguments(
          output_dir=output_dir,
          logging_dir="C:/Users/User/TP1/logs",
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
          save_steps = 100000,
)

for archivo in archivos:
    print(archivo)

    train_dataset = load_dataset(f"{path}/{archivo}", tokenizer)

    trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
    )
      
    data = trainer.train()
    trainer.save_model()
    # time.sleep(1)

    with open(f'logs/Label {label}/Label {label}.txt', 'a') as aux:
        aux.write(f"{archivo}, {data} \n")

    os.remove(f"{path}/{archivo}")