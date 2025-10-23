#!/usr/bin/env python3
"""
Script to extract utterances text from real-persona-chat repository JSON files
and dump them to sentence.json
"""

import json
import os
import tempfile
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Any


def clone_repository(repo_url: str, target_dir: str) -> bool:
    """Clone the repository to a temporary directory"""
    try:
        print(f"Cloning repository: {repo_url}")
        result = subprocess.run(
            ["git", "clone", repo_url, target_dir],
            capture_output=True,
            text=True,
            check=True
        )
        print("Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print(f"stderr: {e.stderr}")
        return False


def extract_utterances_from_json(file_path: str) -> List[str]:
    """Extract utterances text from a single JSON file"""
    utterances = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Navigate through the JSON structure to find utterances
        if isinstance(data, dict):
            # Check if it's a dialogue structure
            if 'utterances' in data:
                for utterance in data['utterances']:
                    if isinstance(utterance, dict) and 'text' in utterance:
                        text = utterance['text'].strip()
                        if text:
                            utterances.append(text)
            
            # Check if it's a list of dialogues
            elif isinstance(data.get('dialogues'), list):
                for dialogue in data['dialogues']:
                    if isinstance(dialogue, dict) and 'utterances' in dialogue:
                        for utterance in dialogue['utterances']:
                            if isinstance(utterance, dict) and 'text' in utterance:
                                text = utterance['text'].strip()
                                if text:
                                    utterances.append(text)
            
            # Check if the data itself is a list of utterances
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        text = item['text'].strip()
                        if text:
                            utterances.append(text)
        
        elif isinstance(data, list):
            # If the root is a list, check each item
            for item in data:
                if isinstance(item, dict):
                    if 'utterances' in item:
                        for utterance in item['utterances']:
                            if isinstance(utterance, dict) and 'text' in utterance:
                                text = utterance['text'].strip()
                                if text:
                                    utterances.append(text)
                    elif 'text' in item:
                        text = item['text'].strip()
                        if text:
                            utterances.append(text)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return utterances


def find_json_files(dialogues_dir: str) -> List[str]:
    """Find all JSON files in the dialogues directory"""
    json_files = []
    
    # Look for JSON files recursively
    for pattern in ['**/*.json', '*.json']:
        json_files.extend(glob.glob(os.path.join(dialogues_dir, pattern), recursive=True))
    
    return json_files


def main():
    """Main function to extract utterances and save to sentence.json"""
    repo_url = "https://github.com/nu-dialogue/real-persona-chat.git"
    dialogues_subdir = "real_persona_chat/dialogues"
    
    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, "real-persona-chat")
        dialogues_dir = os.path.join(repo_dir, dialogues_subdir)
        
        # Clone repository
        if not clone_repository(repo_url, repo_dir):
            print("Failed to clone repository. Exiting.")
            return
        
        # Check if dialogues directory exists
        if not os.path.exists(dialogues_dir):
            print(f"Dialogues directory not found: {dialogues_dir}")
            return
        
        # Find all JSON files
        json_files = find_json_files(dialogues_dir)
        print(f"Found {len(json_files)} JSON files")
        
        if not json_files:
            print("No JSON files found in dialogues directory")
            return
        
        # Extract utterances from all files
        all_utterances = []
        for json_file in json_files:
            print(f"Processing: {os.path.basename(json_file)}")
            utterances = extract_utterances_from_json(json_file)
            all_utterances.extend(utterances)
            print(f"  Found {len(utterances)} utterances")
        
        print(f"\nTotal utterances extracted: {len(all_utterances)}")
        
        # Save to multiple files (10000 utterances per file)
        chunk_size = 10000
        file_count = 0
        
        for i in range(0, len(all_utterances), chunk_size):
            chunk = all_utterances[i:i + chunk_size]
            output_file = f"sentence{file_count}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(chunk)} utterances to {output_file}")
            file_count += 1
        
        print(f"\nTotal files created: {file_count}")
        
        # Show some sample utterances
        if all_utterances:
            print("\nSample utterances:")
            for i, utterance in enumerate(all_utterances[:5]):
                print(f"{i+1}. {utterance[:100]}{'...' if len(utterance) > 100 else ''}")


if __name__ == "__main__":
    main()

