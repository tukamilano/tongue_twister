import json
import os
from collections import Counter
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, TrainingArguments, Trainer
import torch

# (1) データの読み込み
def load_phoneme_data(data_dir):
    """data配下のJSONファイルからphonemesを読み込む"""
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
    
    print(f"Total sequences: {len(all_phonemes)}")
    return all_phonemes

# (2) 音素語彙の構築
def build_vocab(all_phonemes):
    """すべての音素を収集して語彙を作成"""
    all_tokens = []
    for sequence in all_phonemes:
        all_tokens.extend(sequence)
    
    # 出現頻度順にソート
    token_counts = Counter(all_tokens)
    vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + [token for token, _ in token_counts.most_common()]
    
    # トークンからIDへのマッピング
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {vocab[:20]}")
    
    return vocab, token_to_id, id_to_token

# (3) シーケンスをIDに変換
def convert_to_ids(sequences, token_to_id):
    """phonemesのシーケンスをIDのリストに変換"""
    converted = []
    for seq in sequences:
        ids = [token_to_id.get(token, token_to_id['<UNK>']) for token in seq]
        ids = [token_to_id['<BOS>']] + ids + [token_to_id['<EOS>']]
        converted.append(ids)
    return converted

# (4) データセットの準備
def prepare_dataset(all_phonemes, token_to_id):
    """学習用のデータセットを準備"""
    sequences = convert_to_ids(all_phonemes, token_to_id)
    
    # 入力とターゲットを作成
    examples = []
    for seq in sequences:
        if len(seq) > 1:
            input_ids = seq[:-1]
            labels = seq[1:]
            examples.append({
                'input_ids': input_ids,
                'labels': labels
            })
    
    return examples

# (5) カスタムデータコラータ
def collate_fn(batch):
    """バッチ内のシーケンスをパディング"""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    
    for item in batch:
        inp = item['input_ids'] + [0] * (max_len - len(item['input_ids']))
        lab = item['labels'] + [0] * (max_len - len(item['labels']))
        input_ids.append(inp)
        labels.append(lab)
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

# (6) カスタムモデルクラス
class PhonemeDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# メイン実行部分
if __name__ == "__main__":
    # データの読み込み
    data_dir = 'data'
    all_phonemes = load_phoneme_data(data_dir)
    
    # 語彙の構築
    vocab, token_to_id, id_to_token = build_vocab(all_phonemes)
    vocab_size = len(vocab)
    
    # (1) モデルの設定
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    
    # (2) モデルの生成
    model = GPT2LMHeadModel(config)
    
    # (3) データセットの準備
    examples = prepare_dataset(all_phonemes, token_to_id)
    dataset = PhonemeDataset(examples)
    
    # (4) 学習設定
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
    )
    
    # (5) Trainerの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    # (6) 学習開始
    print("Training started...")
    trainer.train()
    
    # (7) モデルの保存
    model.save_pretrained('./model')
    
    # 語彙も保存（キーを文字列に変換）
    with open('./model/vocab.json', 'w') as f:
        vocab_dict = {
            'token_to_id': token_to_id,
            'id_to_token': {str(k): v for k, v in id_to_token.items()}
        }
        json.dump(vocab_dict, f)
    
    print("Training completed!")
    print(f"Model saved to ./model")