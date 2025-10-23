#!/usr/bin/env python3
"""
Script to extract utterances text from real-persona-chat repository using GitHub API
and dump them to sentence*.json files.
Each GitHub file is processed sequentially, and output files rotate every 10,000 utterances.
"""

import json
import os
import requests
import time
from typing import List, Dict


def get_github_file_content(repo_owner: str, repo_name: str, file_path: str, token: str = None) -> str:
    """Get file content from GitHub using API"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        import base64
        content = base64.b64decode(response.json()["content"]).decode('utf-8')
        return content
    except Exception as e:
        print(f"Error fetching {file_path}: {e}")
        return ""


def get_repository_files(repo_owner: str, repo_name: str, path: str = "", token: str = None) -> List[Dict]:
    """Get list of files in a repository directory"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching directory {path}: {e}")
        return []


def extract_utterances_from_json_content(json_content: str) -> List[str]:
    """Extract utterances text from JSON content"""
    utterances = []
    try:
        data = json.loads(json_content)

        if isinstance(data, dict):
            if "utterances" in data:
                for u in data["utterances"]:
                    if isinstance(u, dict) and "text" in u:
                        t = u["text"].strip()
                        if t:
                            utterances.append(t)
            elif isinstance(data.get("dialogues"), list):
                for dialogue in data["dialogues"]:
                    if isinstance(dialogue, dict) and "utterances" in dialogue:
                        for u in dialogue["utterances"]:
                            if isinstance(u, dict) and "text" in u:
                                t = u["text"].strip()
                                if t:
                                    utterances.append(t)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "text" in item:
                    t = item["text"].strip()
                    if t:
                        utterances.append(t)
    except Exception as e:
        print(f"Error processing JSON content: {e}")

    return utterances


def main():
    repo_owner = "nu-dialogue"
    repo_name = "real-persona-chat"
    dialogues_path = "real_persona_chat/dialogues"

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("⚠️  GITHUB_TOKEN environment variable not set.")
        print("   Please run:")
        print("   export GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXXXXX")
        return

    print(f"Fetching files from {repo_owner}/{repo_name}/{dialogues_path}")
    files = get_repository_files(repo_owner, repo_name, dialogues_path, token)

    if not files:
        print("No files found in dialogues directory")
        return

    json_files = [f for f in files if f.get("name", "").endswith(".json")]
    print(f"Found {len(json_files)} JSON files")

    os.makedirs("utterances_output", exist_ok=True)

    buffer = []
    file_index = 0
    total_utterances = 0

    for i, file_info in enumerate(json_files, start=1):
        file_name = file_info["name"]
        file_path = file_info["path"]

        print(f"Processing: {file_name} ({i}/{len(json_files)})")
        content = get_github_file_content(repo_owner, repo_name, file_path, token)

        if content:
            utterances = extract_utterances_from_json_content(content)
            if utterances:
                buffer.extend(utterances)
                total_utterances += len(utterances)
                print(f"  Found {len(utterances)} utterances (buffer={len(buffer)})")

                # ファイルごとに書き出し判定
                if len(buffer) >= 10000:
                    output_file = os.path.join("utterances_output", f"sentence{file_index}.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(buffer, f, ensure_ascii=False, indent=2)
                    print(f"💾 Saved {len(buffer)} utterances → {output_file}")
                    buffer = []  # バッファをリセット
                    file_index += 1
            else:
                print(f"⚠️ No utterances in {file_name}")
        else:
            print(f"⚠️ Failed to get content for {file_name}")

        time.sleep(0.1)

    # 最後の残りを保存
    if buffer:
        output_file = os.path.join("utterances_output", f"sentence{file_index}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(buffer, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved remaining {len(buffer)} utterances → {output_file}")

    print(f"\n✅ Done. Total utterances: {total_utterances}")
    print(f"   Output files: {file_index + 1} under ./utterances_output/")


if __name__ == "__main__":
    main()
