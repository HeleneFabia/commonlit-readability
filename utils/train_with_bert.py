import torch
from torch.utils.data import Dataset
from torch import nn


class BERTDataset(Dataset):
    
    def __init__(self, tokenizer, max_length, texts, targets):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.targets = targets
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        
        bert_text = self.tokenizer.encode_plus(
            text,
            padding="max_length",
            pad_to_max_length=self.max_length,
            truncation=True,
        )
        
        input_ids = torch.tensor(bert_text["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(bert_text["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(bert_text["token_type_ids"], dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "target": target
        }
    
    
def loss_fn(output, target):
    mse_loss = nn.MSELoss()
    return torch.sqrt(mse_loss(output, target))


def training(
    train_dataloader,
    model,
    optimizer,
    scheduler,
    device
): 
    model.train()
    # all_preds = list()
    # all_targets = list()

    for batch in train_dataloader:
        losses = list()
        optimizer.zero_grad()

        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        tokentype = batch["token_type_ids"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        output = model(ids, mask)
        output = output["logits"].squeeze(-1)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss)
        # all_preds.append(output.detach().cpu().numpy())
        # all_targets.append(target.detach().squeeze(-1).cpu().numpy())

        del ids
        del mask
        del tokentype
        del target
        del loss 

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    train_rme_loss = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    print(f"Training RSME: {train_rme_loss:.4f}")
    return train_rme_loss


def validating(
    valid_dataloader,
    model,
    device
): 
    model.eval()
    # all_preds = list()
    # all_targets = list()

    for batch in valid_dataloader:
        losses = list()

        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        tokentype = batch["token_type_ids"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        output = model(ids, mask)
        output = output["logits"].squeeze(-1)
        loss = loss_fn(output, target)

        losses.append(loss)
        # all_preds.append(output.detach().cpu().numpy())
        # all_targets.append(target.detach().squeeze(-1).cpu().numpy())

        del ids
        del mask
        del tokentype
        del target
        del loss 

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    valid_rme_loss = np.sqrt(mean_squared_error(all_targets, all_preds))

    print(f"Validation RSME: {valid_rme_loss:.4f}")
    return valid_rme_loss