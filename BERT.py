# Load the necessary libraries
import torch
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW


class BertClassification(object):


    def __init__(self, model_name='bert-base-uncased', num_labels=2, lr=2e-5, eps=1e-8, batch_size=32):

        # Load the pre-trained BERT model and tokenizer
        self._model_name = model_name
        self._num_labels = num_labels
        self._lr = lr
        self._eps = eps
        self._batch_size = batch_size

        self._init_env()

    def _init_env(self):
        self.tokenizer = BertTokenizer.from_pretrained(self._model_name)
        self.model = BertForSequenceClassification.from_pretrained(self._model_name, num_labels=self._num_labels)

        # Define the optimizer and learning rate
        self.optimizer = AdamW(self.model.parameters(), lr=self._lr, eps=self._eps)

    def _prepare_data(self, train_data):
        # Define the DataLoader for the training data
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self._batch_size)

    def train(self, train_data):
        self._prepare_data(train_data)
        # Train the model
        self.model.train()
        for epoch in range(3):
            for batch in self.train_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def save_model(self, output_dir):
        # Save the fine-tuned model
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)