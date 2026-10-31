# dataset.py
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def prepare_dataloaders(batch_size=16, max_length=128):
    print("Memuat dan menggabungkan berbagai sumber dataset...")
    
    # 1. Dataset Pertama: GitHub Issues (untuk konteks teknis/bug)
    print("- Memuat dataset GitHub Issues...")
    github_raw = load_dataset("lewtun/github-issues", split="train")
    
    def process_github(example):
        title = str(example["title"]) if example["title"] else ""
        body = str(example["body"]) if example["body"] else ""
        return {"text": f"Technical Bug: {title} - {body}"}
    
    github_dataset = github_raw.map(process_github, remove_columns=github_raw.column_names)
    # Ambil sebagian saja agar seimbang (misal 3000 baris pertama)
    github_dataset = github_dataset.select(range(min(3000, len(github_dataset))))

    # 2. Dataset Kedua: Dataset Percakapan / Umum (Misal: DailyDialog atau Wikitext untuk pengetahuan umum)
    print("- Memuat dataset percakapan/umum...")
    # Menggunakan wikitext subset kecil atau dailydialog untuk percakapan sehari-hari
    try:
        general_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter baris kosong
        general_dataset = general_raw.filter(lambda x: len(x["text"].strip()) > 20)
        general_dataset = general_dataset.select(range(min(3000, len(general_dataset))))
    except Exception:
        # Cadangan jika wikitext gagal diunduh, gunakan duplikasi/subset github lain
        general_dataset = github_dataset

    # 3. Gabungkan Kedua Dataset Menjadi Satu Korpus Pelatihan
    combined_dataset = concatenate_datasets([github_dataset, general_dataset])
    print(f"Total korpus data gabungan: {len(combined_dataset)} baris teks.")

    print("Memuat tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    vocab_size = tokenizer.vocab_size

    # 4. Fungsi Tokenisasi
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    print("Memulai tokenisasi teks gabungan...")
    tokenized_datasets = combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")

    # 5. DataLoader
    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=batch_size)
    print(f"DataLoader siap! Total batch: {len(train_dataloader)}")
    
    return train_dataloader, vocab_size, tokenizer