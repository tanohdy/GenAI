# main.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

# Mengimpor modul yang sudah diperbarui
from dataset import prepare_dataloaders
from model import CustomDecoderLM

def main():
    # 1. Pengecekan dan Otomasi Perangkat GPU Lokal
    print("--- Memeriksa Perangkat Komputasi ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Terdeteksi: {torch.cuda.get_device_name(0)}")
        print("Pelatihan akan berjalan ngebut menggunakan VRAM GPU! 🚀")
        # GPU lokal bisa menampung batch size lebih besar agar proses lebih cepat
        batch_size = 32 
    else:
        device = torch.device("cpu")
        print("GPU tidak terdeteksi. Menggunakan CPU (proses mungkin lebih lambat).")
        batch_size = 16

    # 2. Konfigurasi Parameter Pelatihan
    max_length = 128
    epochs = 10          # Jumlah epoch yang ideal untuk dataset gabungan
    learning_rate = 1e-2 # Learning rate optimal untuk konvergensi stabil

    # 3. Persiapan Data (Dataset Gabungan GitHub Issues + Wikitext)
    train_loader, vocab_size, tokenizer = prepare_dataloaders(batch_size=batch_size, max_length=max_length)

    # 4. Inisialisasi Arsitektur Model
    print("\n--- Membangun Arsitektur Custom Decoder-Only LM ---")
    model = CustomDecoderLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        ff_hidden_dim=512,
        max_seq_len=max_length
    ).to(device)

    # 5. Opsi Melanjutkan Pelatihan (Resume Training) jika file bobot sudah ada
    model_filename = "decoder_bug_generator.pth"
    if os.path.exists(model_filename):
        print(f"Menemukan file bobot lama ({model_filename}). Melanjutkan pelatihan...")
        model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))
    else:
        print("Memulai pelatihan dari awal (scratch)...")

    # 6. Loss, Optimizer, dan Scheduler
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 7. Siklus Pelatihan (Training Loop)
    print("\n--- Memulai Siklus Pelatihan di GPU ---")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Pindahkan data ke GPU
            input_ids = batch["input_ids"].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Shifted targets untuk Next-Token Prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Hitung Loss
            loss = loss_fn(
                shift_logits.view(-1, vocab_size), 
                shift_labels.view(-1)
            )
            
            # Backward pass & Optimasi
            loss.backward()
            
            # Gradient Clipping untuk mencegah ledakan gradien
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        # Update learning rate scheduler per epoch
        scheduler.step()
        
        avg_epoch_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] | Loss Rata-rata: {avg_epoch_loss:.4f} | LR: {current_lr:.6f}")

    # 8. Simpan Model Hasil Pelatihan
    print("\n--- Menyimpan Artefak Model Terbaru ---")
    torch.save(model.state_dict(), model_filename)
    print(f"Model berhasil disimpan sebagai '{model_filename}'")

if __name__ == "__main__":
    main()