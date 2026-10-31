# inference.py
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import os

# Mengimpor arsitektur model dari file model.py
from model import CustomDecoderLM

def generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, device="cpu"):
    model.eval()
    
    # 1. Encode prompt menjadi token IDs dan pindahkan ke device (GPU/CPU)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Batasi panjang input maksimum sesuai kapasitas model (128 token)
            current_input = input_ids[:, -128:]
            
            # 2. Forward pass melalui decoder
            logits = model(current_input) # Ukuran: [Batch, Seq_Len, Vocab_Size]
            
            # Ambil logit dari token terakhir dan sesuaikan dengan temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            # 3. Ubah logit menjadi probabilitas menggunakan Softmax
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 4. Gunakan pengambilan sampel (Multinomial sampling) agar teks lebih bervariasi
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 5. Gabungkan token baru ke urutan input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Berhenti jika model memproduksi token padding
            if next_token.item() == tokenizer.pad_token_id:
                break
                
    # 6. Decode token kembali menjadi string teks
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    # 1. Otomasi Pendeteksian Perangkat (GPU RTX 3050 / CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Menggunakan perangkat untuk inference: CUDA ({torch.cuda.get_device_name(0)}) 🚀")
    else:
        device = torch.device("cpu")
        print("Menggunakan perangkat untuk inference: CPU")

    # 2. Inisialisasi Tokenizer
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        
    vocab_size = tokenizer.vocab_size

    # 3. Inisialisasi Arsitektur Model
    print("Memuat arsitektur model...")
    model = CustomDecoderLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        ff_hidden_dim=512,
        max_seq_len=128
    ).to(device)

    # 4. Muat Bobot Terbaru (.pth) hasil pelatihan
    model_filename = "decoder_bug_generator.pth"
    if os.path.exists(model_filename):
        # Menggunakan weights_only=True sesuai standar keamanan PyTorch modern
        model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))
        print(f"Bobot model berhasil dimuat dari '{model_filename}'!\n")
    else:
        print(f"Peringatan: File '{model_filename}' tidak ditemukan. Model berjalan dengan bobot acak.")
        return

    # 5. Sesi Interaktif Uji Coba Generasi Teks
    print("="*55)
    print("PENGUJIAN GENERATIVE AI (CUSTOM DECODER TRANSFORMER)")
    print("="*55)
    
    while True:
        prompt_user = input("\nMasukkan kalimat awal / prompt (ketik 'exit' untuk keluar): ")
        if prompt_user.lower() == 'exit':
            print("Keluar dari program inference.")
            break
            
        if not prompt_user.strip():
            continue
            
        print("Generating...")
        hasil_generate = generate_text(model, tokenizer, prompt=prompt_user, max_new_tokens=35, temperature=0.7, device=device)
        
        print(f"\nHasil Generasi Model:\n{hasil_generate}")
        print("-" * 55)

if __name__ == "__main__":
    main()