import json
import os
from transformers import GPT2Config, GPT2LMHeadModel

def create_dummy_model():
    """学習していないダミーモデルと語彙を作成"""
    print("Creating dummy model and vocabulary...")
    
    # データから音素を収集して語彙を作成
    data_dir = 'data'
    all_phonemes = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'phonemes' in item:
                        all_phonemes.append(item['phonemes'])
    
    # 語彙を構築
    from collections import Counter
    all_tokens = []
    for sequence in all_phonemes:
        all_tokens.extend(sequence)
    
    token_counts = Counter(all_tokens)
    vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + [token for token, _ in token_counts.most_common()]
    
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {vocab[:20]}")
    
    # モデルを作成（学習なし）
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=512,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    
    model = GPT2LMHeadModel(config)
    
    # 保存
    os.makedirs('./model', exist_ok=True)
    model.save_pretrained('./model')
    
    # 語彙も保存
    with open('./model/vocab.json', 'w') as f:
        vocab_dict = {
            'token_to_id': token_to_id,
            'id_to_token': {str(k): v for k, v in id_to_token.items()}
        }
        json.dump(vocab_dict, f)
    
    print("\nDummy model created successfully!")
    print(f"Model saved to ./model")
    print(f"Vocabulary size: {len(vocab)}")
    print("\nYou can now run: python validate.py")

if __name__ == "__main__":
    create_dummy_model()

