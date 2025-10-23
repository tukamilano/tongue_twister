#!/usr/bin/env python3
"""
Extract persona text from interlocutors.json and save as a simple JSON file.
This script extracts all Japanese persona sentences and saves them in a flat structure.
"""

import json
import os

def extract_persona_text(input_file, output_file):
    """
    Extract persona text from interlocutors.json and save as a simple JSON file.
    
    Args:
        input_file (str): Path to the input interlocutors.json file
        output_file (str): Path to the output JSON file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all persona text
        persona_texts = []
        
        for interlocutor_id, interlocutor_data in data.items():
            if 'persona' in interlocutor_data and isinstance(interlocutor_data['persona'], list):
                # Add all persona sentences for this interlocutor
                for sentence in interlocutor_data['persona']:
                    if isinstance(sentence, str) and sentence.strip():
                        persona_texts.append(sentence.strip())
        
        # Save the extracted text as a simple JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(persona_texts, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully extracted {len(persona_texts)} persona sentences")
        print(f"Output saved to: {output_file}")
        
        # Show first few examples
        if persona_texts:
            print("\nFirst 5 examples:")
            for i, text in enumerate(persona_texts[:5]):
                print(f"{i+1}. {text}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Set file paths
    input_file = "interlocutors.json"
    output_file = "persona_text.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found in current directory")
        print("Please make sure interlocutors.json is in the same directory as this script")
        exit(1)
    
    # Extract persona text
    extract_persona_text(input_file, output_file)