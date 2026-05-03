"""CUAD Dataset Adapter

This module provides functionality to load and process the CUAD (Contract Understanding Atticus Dataset)
for use with the Legal QA Assistant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset


class CUADAdapter:
    """Adapter for loading and processing CUAD dataset."""
    
    def __init__(self):
        self.dataset = None
        self.qa_pairs = []
    
    def load_from_huggingface(self) -> bool:
        """Load CUAD dataset from Hugging Face."""
        try:
            # Try to load the main CUAD dataset first
            self.dataset = load_dataset("theatticusproject/cuad")
            return True
        except Exception as e:
            print(f"Error loading CUAD from Hugging Face: {e}")
            # Fallback to demo data
            return self._load_demo_data()
    
    def _load_demo_data(self) -> bool:
        """Load demo CUAD-style data for demonstration."""
        demo_data = [
            {
                "id": "demo_1",
                "title": "Sample Contract Agreement",
                "question": "Highlight the parts (if any) of this contract related to 'Confidentiality' that should be reviewed by a lawyer. Details: Confidential information protection",
                "answer": "The receiving party shall keep all confidential information secret for three years after termination.",
                "context": "CONFIDENTIALITY: The receiving party shall keep all confidential information secret for three years after termination. Neither party shall disclose confidential information to any third party without prior written consent.",
                "evidence": "The receiving party shall keep all confidential information secret for three years after termination.",
                "category": "Confidentiality"
            },
            {
                "id": "demo_2", 
                "title": "Sample Service Agreement",
                "question": "Highlight the parts (if any) of this contract related to 'Termination' that should be reviewed by a lawyer. Details: Contract termination conditions",
                "answer": "Either party may terminate this Agreement by giving 30 days written notice.",
                "context": "TERMINATION: Either party may terminate this Agreement by giving 30 days written notice. The Company may terminate the Agreement immediately if the Contractor commits a material breach.",
                "evidence": "Either party may terminate this Agreement by giving 30 days written notice.",
                "category": "Termination"
            },
            {
                "id": "demo_3",
                "title": "Sample Employment Contract", 
                "question": "Highlight the parts (if any) of this contract related to 'Non-Compete' that should be reviewed by a lawyer. Details: Non-competition restrictions",
                "answer": "For six months after termination, the Employee shall not work for a direct competitor in the same city.",
                "context": "NON-COMPETE: For six months after termination, the Employee shall not work for a direct competitor in the same city. This restriction applies to competitors within a 50-mile radius.",
                "evidence": "For six months after termination, the Employee shall not work for a direct competitor in the same city.",
                "category": "Non-Compete"
            },
            {
                "id": "demo_4",
                "title": "Sample License Agreement",
                "question": "Highlight the parts (if any) of this contract related to 'Governing Law' that should be reviewed by a lawyer. Details: Applicable law and jurisdiction",
                "answer": "This Agreement shall be governed by the laws of the Republic of Kazakhstan.",
                "context": "GOVERNING LAW: This Agreement shall be governed by the laws of the Republic of Kazakhstan. Any disputes shall be resolved in the courts of Astana.",
                "evidence": "This Agreement shall be governed by the laws of the Republic of Kazakhstan.",
                "category": "Governing Law"
            },
            {
                "id": "demo_5",
                "title": "Sample Partnership Agreement",
                "question": "Highlight the parts (if any) of this contract related to 'Liability' that should be reviewed by a lawyer. Details: Liability limitations",
                "answer": "Neither party shall be liable for indirect, incidental, or consequential damages.",
                "context": "LIABILITY: Neither party shall be liable for indirect, incidental, or consequential damages. The total liability of either party shall not exceed the total fees paid during the previous six months.",
                "evidence": "Neither party shall be liable for indirect, incidental, or consequential damages.",
                "category": "Liability"
            }
        ]
        
        self.qa_pairs = demo_data
        return True
    
    def load_from_json(self, file_path: Path) -> bool:
        """Load CUAD dataset from local JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, dict) and 'data' in data:
                self.qa_pairs = data['data']
            elif isinstance(data, list):
                self.qa_pairs = data
            else:
                raise ValueError("Unsupported JSON format")
            
            return True
        except Exception as e:
            print(f"Error loading CUAD from JSON: {e}")
            return False
    
    def extract_qa_pairs(self, split: str = "train", limit: Optional[int] = None) -> List[Dict]:
        """Extract QA pairs from the dataset."""
        qa_pairs = []
        
        if self.dataset is not None:
            # Load from Hugging Face dataset (if successful)
            try:
                data = self.dataset[split]
                
                for i, item in enumerate(data):
                    if limit and i >= limit:
                        break
                    
                    # Extract answer text
                    answer_text = item['answers']['text'][0] if item['answers']['text'] else ""
                    
                    qa_pair = {
                        'id': item['id'],
                        'title': item['title'],
                        'question': item['question'],
                        'answer': answer_text,
                        'context': item['context'],
                        'evidence': item['context'][item['answers']['answer_start'][0]:item['answers']['answer_start'][0] + len(answer_text)] if item['answers']['answer_start'] and answer_text else "",
                        'category': self._extract_category_from_question(item['question'])
                    }
                    qa_pairs.append(qa_pair)
            except Exception as e:
                print(f"Error processing Hugging Face dataset: {e}")
                # Fall back to demo data
                return self._extract_demo_qa_pairs(limit)
        
        elif self.qa_pairs:
            # Use loaded QA pairs (demo data)
            return self._extract_demo_qa_pairs(limit)
        
        return qa_pairs
    
    def _extract_demo_qa_pairs(self, limit: Optional[int] = None) -> List[Dict]:
        """Extract QA pairs from demo data."""
        qa_pairs = []
        for i, item in enumerate(self.qa_pairs):
            if limit and i >= limit:
                break
            
            # Map CUAD category to natural question
            natural_question = self._map_category_to_natural_question(item.get('category', ''))
            
            qa_pair = {
                'id': item.get('id', f'item_{i}'),
                'title': item.get('title', ''),
                'question': natural_question,
                'original_question': item.get('question', ''),  # Keep original for reference
                'answer': item.get('answer', ''),
                'context': item.get('context', ''),
                'evidence': item.get('evidence', ''),
                'category': item.get('category', '')
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _map_category_to_natural_question(self, category: str) -> str:
        """Map CUAD category to natural question format."""
        category_mappings = {
            "Confidentiality": "What confidentiality obligations are stated?",
            "Termination": "How can the agreement be terminated?",
            "Non-Compete": "What non-compete restriction is stated?",
            "Governing Law": "Which law governs the agreement?",
            "Liability": "What is the liability limitation or liability cap?",
            "Assignment": "Can the agreement be assigned?",
            "Payment": "What payment obligations are stated?",
            "Intellectual Property": "Who owns the intellectual property?",
            "Data Protection": "What data protection obligation is stated?",
            "Dispute Resolution": "How are disputes resolved?",
            "Force Majeure": "What events are covered by force majeure?",
            "Audit": "What audit rights are granted?",
            "Change of Control": "What happens if there is a change of control?",
            "Compliance": "What compliance requirements are stated?",
            "Delivery": "What delivery terms are specified?",
            "Effective Date": "When does the agreement become effective?",
            "Expiration": "When does the agreement expire?",
            "Governing Law": "Which law governs the agreement?",
            "Indemnification": "What indemnification obligations exist?",
            "Insurance": "What insurance requirements are stated?",
            "Intellectual Property": "Who owns the intellectual property?",
            "Joint Venture": "What joint venture terms are specified?",
            "License": "What license rights are granted?",
            "No Solicit": "What no-solicit restrictions apply?",
            "Non-Compete": "What non-compete restriction is stated?",
            "Non-Disparagement": "What non-disparagement obligations exist?",
            "Non-Disclosure": "What non-disclosure requirements apply?",
            "Payment": "What payment obligations are stated?",
            "Price": "What pricing terms are specified?",
            "Purchase Price": "What is the purchase price?",
            "Renewal": "How can the agreement be renewed?",
            "Revenue Sharing": "What revenue sharing arrangements exist?",
            "Source Code": "What source code obligations apply?",
            "Term": "What is the term of the agreement?",
            "Termination": "How can the agreement be terminated?",
            "Third Party": "What third party rights are granted?",
            "Uncapped Liability": "What uncapped liability provisions exist?",
            "Volume": "What volume commitments are stated?",
            "Warranty": "What warranty provisions apply?"
        }
        
        return category_mappings.get(category, f"What {category.lower()} provisions are stated?")
    
    def _extract_category_from_question(self, question: str) -> str:
        """Extract category from question text."""
        # CUAD questions have format like: "Highlight the parts related to 'Category Name'"
        if "related to" in question:
            start = question.find('"') + 1
            end = question.find('"', start)
            if start > 0 and end > start:
                return question[start:end]
        return "Unknown"
    
    def get_categories(self) -> List[str]:
        """Get all unique categories in the dataset."""
        qa_pairs = self.extract_qa_pairs()
        categories = list(set(pair['category'] for pair in qa_pairs))
        return sorted([cat for cat in categories if cat != "Unknown"])
    
    def get_dataframe(self, split: str = "train", limit: Optional[int] = None) -> pd.DataFrame:
        """Get QA pairs as a pandas DataFrame."""
        qa_pairs = self.extract_qa_pairs(split, limit)
        return pd.DataFrame(qa_pairs)
    
    def get_sample_questions(self, num_samples: int = 10) -> List[str]:
        """Get sample questions from the dataset."""
        qa_pairs = self.extract_qa_pairs(limit=num_samples)
        return [pair['question'] for pair in qa_pairs]
    
    def create_document_from_contexts(self, split: str = "train", limit: Optional[int] = None) -> str:
        """Create a single document from all contexts for processing."""
        qa_pairs = self.extract_qa_pairs(split, limit)
        contexts = []
        
        for i, pair in enumerate(qa_pairs):
            # Add section header
            category = pair['category']
            title = pair['title'][:50] + "..." if len(pair['title']) > 50 else pair['title']
            contexts.append(f"=== Section {i+1}: {category} ({title}) ===")
            contexts.append(pair['context'])
            contexts.append("")  # Add spacing
        
        return "\n".join(contexts)
