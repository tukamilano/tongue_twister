import pyopenjtalk

def get_phonemes(text: str) -> list[str]:
    """
    テキストをpyopenjtalkで音素のリストに変換する。
    例: 「東京特許」 -> ['t', 'o', 'o', 'ky', 'o', 'o', 't', 'o', 'Q', 'ky', 'o']
    """
    if not text:
        return []
    
    # g2p (grapheme-to-phoneme) を実行
    phonemes = pyopenjtalk.g2p(text, kana=False)
    
    # 'pau' (ポーズ) や 'sil' (無音) を除外し、音素のみのリストにする
    cleaned_phonemes = [p for p in phonemes.split(' ') if p not in ['sil', 'pau']]
    return cleaned_phonemes