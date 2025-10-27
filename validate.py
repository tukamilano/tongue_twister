import json
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# (1) モデルと語彙のロード
def load_model_and_vocab(model_dir='./model'):
    """学習済みモデルと語彙をロード"""
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        print("Please run one of the following:")
        print("  1. python3 create_dummy_model.py  # Create a dummy model for testing")
        print("  2. python3 train.py               # Train a model")
        return None, None, None
    
    vocab_path = os.path.join(model_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' does not exist.")
        print("Please run one of the following:")
        print("  1. python3 create_dummy_model.py  # Create a dummy model for testing")
        print("  2. python3 train.py               # Train a model")
        return None, None, None
    
    # 語彙の読み込み
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        # id_to_tokenのキーが文字列になっているので整数に変換
        id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
    
    # モデルの読み込み
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    
    print(f"Loaded model from {model_dir}")
    print(f"Vocabulary size: {len(token_to_id)}")
    
    return model, token_to_id, id_to_token

# (2) シーケンス生成
def generate_sequence(model, token_to_id, id_to_token, start_tokens, max_length=50, temperature=1.0):
    """音素シーケンスを生成"""
    # 開始トークンをIDに変換
    generated = [token_to_id.get('<BOS>', 0)]
    if start_tokens:
        for token in start_tokens:
            generated.append(token_to_id.get(token, token_to_id.get('<UNK>', 3)))
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - len(generated)):
            # 現在のシーケンス全体を入力として使用
            input_ids = torch.tensor([generated], dtype=torch.long)
            
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :] / temperature
            
            # 次のトークンをサンプリング
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # EOS またはPADが生成されたら終了
            if next_token_id == token_to_id.get('<EOS>', 1) or next_token_id == token_to_id.get('<PAD>', 0):
                break
            
            generated.append(next_token_id)
    
    # IDをトークンに変換
    tokens = [id_to_token.get(idx, '<UNK>') for idx in generated]
    
    # BOSとEOSを除外
    if tokens and tokens[0] == '<BOS>':
        tokens = tokens[1:]
    if tokens and tokens[-1] == '<EOS>':
        tokens = tokens[:-1]
    
    return tokens

# (3) データセットの精度評価
def evaluate_on_dataset(model, token_to_id, id_to_token, data_dir='data', num_samples=100):
    """データセット上でモデルの性能を評価"""
    import random
    
    # データの読み込み
    all_phonemes = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'phonemes' in item:
                        all_phonemes.append(item['phonemes'])
    
    # サンプルをランダムに選択
    sample_phonemes = random.sample(all_phonemes, min(num_samples, len(all_phonemes)))
    
    print(f"\nEvaluating on {len(sample_phonemes)} samples...")
    
    # 各サンプルの一部を与えて、残りを予測
    correct_predictions = 0
    total_tokens = 0
    
    for target_seq in sample_phonemes:
        if len(target_seq) < 3:
            continue
        
        # 前半をコンテキストとして使用
        context_len = len(target_seq) // 2
        context = target_seq[:context_len]
        target = target_seq[context_len:]
        
        # 生成
        generated = generate_sequence(model, token_to_id, id_to_token, context, max_length=len(target) + 10)
        
        # 正確性を計算（生成されたトークンの最初の部分を比較）
        min_len = min(len(generated), len(target))
        if min_len > 0:
            correct = sum(1 for i in range(min_len) if generated[i] == target[i])
            correct_predictions += correct
            total_tokens += len(target)
    
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    print(f"Token-level accuracy: {accuracy:.2%}")
    print(f"Correct: {correct_predictions}/{total_tokens}")
    
    return accuracy

# (4) 対話的な生成
def interactive_generation(model, token_to_id, id_to_token):
    """対話的に音素シーケンスを生成"""
    print("\n=== Interactive Phoneme Generation ===")
    print("Enter phonemes (space-separated) to start generation, or 'q' to quit")
    print("Example: y o r o")
    
    while True:
        user_input = input("\nInput phonemes: ").strip()
        
        if user_input.lower() == 'q':
            break
        
        if not user_input:
            continue
        
        try:
            start_tokens = user_input.split()
            generated = generate_sequence(model, token_to_id, id_to_token, start_tokens)
            print(f"Generated: {' '.join(generated)}")
        except Exception as e:
            print(f"Error: {e}")

# メイン実行部分
if __name__ == "__main__":
    import sys
    
    # モデルのロード
    model, token_to_id, id_to_token = load_model_and_vocab()
    
    # モデルのロードに失敗した場合は終了
    if model is None or token_to_id is None or id_to_token is None:
        sys.exit(1)
    
    # コマンドライン引数で動作を切り替え
    if len(sys.argv) > 1:
        if sys.argv[1] == 'eval':
            # データセットでの評価
            num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            evaluate_on_dataset(model, token_to_id, id_to_token, num_samples=num_samples)
        elif sys.argv[1] == 'interactive':
            # 対話的な生成
            interactive_generation(model, token_to_id, id_to_token)
        elif sys.argv[1] == 'generate':
            # 特定のシーケンスを生成
            start_tokens = sys.argv[2:] if len(sys.argv) > 2 else []
            generated = generate_sequence(model, token_to_id, id_to_token, start_tokens)
            print(f"Generated: {' '.join(generated)}")
    else:
        # デフォルトで対話モード
        interactive_generation(model, token_to_id, id_to_token)

