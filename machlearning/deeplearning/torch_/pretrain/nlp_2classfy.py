import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report
# C:\Users\EDY\.cache\huggingface\hub\models--bert-base-uncased

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class TransformerClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 二分类问题

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output  # BERT的[CLS] token的输出
        logits = self.classifier(pooler_output)
        return logits

def train(model, train_loader, val_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {acc * 100:.2f}%")
        model.train()

def evaluate(model, val_loader):
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    print(classification_report(val_labels, val_preds))

if __name__=='__main__':
    data = [
        ("I love programming", 1),
        ("I hate bugs", 0),
        ("This is awesome", 1),
        ("I don't like error messages", 0),
    ]

    texts, labels = zip(*data)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    model = TransformerClassifier()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    train(model, train_loader, val_loader, optimizer, criterion)
    evaluate(model, val_loader)

    # torch.save(model.state_dict(), 'transformer_classifier.pth')
    # model.load_state_dict(torch.load('transformer_classifier.pth'))



