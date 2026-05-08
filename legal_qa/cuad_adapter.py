"""CUAD Dataset Adapter

This module provides functionality to load and process the CUAD (Contract Understanding Atticus Dataset)
for use with the Legal QA Assistant.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


CUAD_QA_DATA_REVISION = "53fc9be1de79a35a82ac36f33198a753df949523"


class CUADAdapter:
    """Adapter for loading and processing CUAD dataset."""
    
    def __init__(self):
        self.dataset = None
        self.qa_pairs = []
        self.last_error = ""
    
    def load_from_huggingface(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        use_demo_on_fail: bool = False,
    ) -> bool:
        """Load real CUAD QA dataset from Hugging Face."""
        self.last_error = ""

        loaders = [
            self._load_cuad_qa_dataset_revision,
            self._load_cuad_v1_json,
        ]

        errors = []
        for loader in loaders:
            try:
                if loader(split=split, limit=limit):
                    return True
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")

        self.last_error = " | ".join(errors) or "No CUAD loader returned data."
        print(f"Error loading CUAD from Hugging Face: {self.last_error}")

        if use_demo_on_fail:
            return self._load_demo_data()

        return False

    def _load_cuad_qa_dataset_revision(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> bool:
        """Load the converted CUAD-QA data revision that does not need remote code."""
        self.dataset = load_dataset(
            "theatticusproject/cuad-qa",
            revision=CUAD_QA_DATA_REVISION,
        )
        self.qa_pairs = self.extract_qa_pairs(split=split, limit=limit)

        if not self.qa_pairs:
            raise ValueError("CUAD-QA loaded, but no QA pairs were extracted.")

        return True

    def _load_cuad_v1_json(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> bool:
        """Load the original CUAD SQuAD-style JSON from the stable CUAD repo."""
        file_path = hf_hub_download(
            repo_id="theatticusproject/cuad",
            repo_type="dataset",
            filename="CUAD_v1/CUAD_v1.json",
        )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dataset = None
        self.qa_pairs = self._extract_squad_qa_pairs(data, limit=limit)

        if not self.qa_pairs:
            raise ValueError("CUAD_v1.json loaded, but no QA pairs were extracted.")

        return True

    def load_from_huggingface_legacy(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        use_demo_on_fail: bool = False,
    ) -> bool:
        """Legacy loader kept for reference with old datasets versions."""
        try:
            self.dataset = load_dataset(
                "theatticusproject/cuad-qa",
                trust_remote_code=True,
            )

            self.qa_pairs = self.extract_qa_pairs(split=split, limit=limit)

            if not self.qa_pairs:
                raise ValueError("CUAD-QA loaded, but no QA pairs were extracted.")

            return True

        except Exception as e:
            self.last_error = str(e)
            print(f"Error loading CUAD-QA from Hugging Face: {e}")

            if use_demo_on_fail:
                return self._load_demo_data()

            return False
    
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
                if data['data'] and isinstance(data['data'][0], dict) and 'paragraphs' in data['data'][0]:
                    self.qa_pairs = self._extract_squad_qa_pairs(data)
                else:
                    self.qa_pairs = data['data']
            elif isinstance(data, list):
                self.qa_pairs = data
            else:
                raise ValueError("Unsupported JSON format")
            
            return True
        except Exception as e:
            print(f"Error loading CUAD from JSON: {e}")
            return False

    def _extract_squad_qa_pairs(
        self,
        data: Dict,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Extract QA pairs from SQuAD-style CUAD JSON."""
        qa_pairs = []

        for item in self._iter_squad_items(data):
            if limit is not None and len(qa_pairs) >= limit:
                break
            qa_pairs.append(self._normalize_qa_item(item, fallback_id=f"cuad_{len(qa_pairs)}"))

        return qa_pairs

    def _iter_squad_items(self, data: Dict) -> Iterable[Dict]:
        for document in data.get("data", []):
            title = document.get("title", "")
            for paragraph in document.get("paragraphs", []):
                context = paragraph.get("context", "") or ""
                for qa in paragraph.get("qas", []):
                    yield {
                        "id": qa.get("id", ""),
                        "title": title,
                        "context": context,
                        "question": qa.get("question", "") or "",
                        "answers": qa.get("answers", []) or [],
                    }

    def _normalize_qa_item(self, item: Dict, fallback_id: str) -> Dict:
        answers = item.get("answers", {}) or {}

        if isinstance(answers, list):
            answer_texts = [answer.get("text", "") for answer in answers]
            answer_starts = [answer.get("answer_start", -1) for answer in answers]
        else:
            answer_texts = answers.get("text", [])
            answer_starts = answers.get("answer_start", [])

        if isinstance(answer_texts, str):
            answer_texts = [answer_texts]

        if isinstance(answer_starts, int):
            answer_starts = [answer_starts]

        answer_text = answer_texts[0] if answer_texts else ""
        answer_start = answer_starts[0] if answer_starts else -1
        context = item.get("context", "") or ""

        if answer_text and isinstance(answer_start, int) and answer_start >= 0:
            evidence = context[answer_start:answer_start + len(answer_text)]
        else:
            evidence = ""

        question = item.get("question", "") or ""
        category = self._extract_category_from_question(question)

        return {
            "id": item.get("id", fallback_id),
            "title": item.get("title", ""),
            "question": self._map_category_to_natural_question(category),
            "original_question": question,
            "answer": answer_text,
            "context": context,
            "evidence": evidence,
            "category": category,
        }
    
    def extract_qa_pairs(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        include_unanswerable: bool = True,
    ) -> List[Dict]:
        """Extract QA pairs from CUAD-QA dataset."""
        qa_pairs = []

        if self.dataset is None:
            return self._extract_demo_qa_pairs(limit) if self.qa_pairs else []

        data = self.dataset[split]

        for i, item in enumerate(data):
            if limit is not None and len(qa_pairs) >= limit:
                break

            qa_pair = self._normalize_qa_item(item, fallback_id=f"cuad_{i}")

            if not include_unanswerable and not qa_pair["answer"]:
                continue

            qa_pairs.append(qa_pair)

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
        """Extract CUAD category from question text."""
        if not question:
            return "Unknown"

        match = re.search(r"related to\s+[\"']([^\"']+)[\"']", question)

        if match:
            return match.group(1).strip()

        return "Unknown"
    
    def get_categories(self) -> List[str]:
        """Get all unique categories in the dataset."""
        qa_pairs = self.extract_qa_pairs()
        categories = list(set(pair['category'] for pair in qa_pairs))
        return sorted([cat for cat in categories if cat != "Unknown"])
    
    def get_dataframe(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get QA pairs as a pandas DataFrame."""
        if not self.qa_pairs:
            self.qa_pairs = self.extract_qa_pairs(split=split, limit=limit)

        data = self.qa_pairs[:limit] if limit else self.qa_pairs
        return pd.DataFrame(data)
    
    def get_sample_questions(self, num_samples: int = 10) -> List[str]:
        """Get sample questions from the dataset."""
        qa_pairs = self.extract_qa_pairs(limit=num_samples)
        return [pair['question'] for pair in qa_pairs]
    
    def create_document_from_contexts(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> str:
        """Create a single document from CUAD contexts."""
        if not self.qa_pairs:
            self.qa_pairs = self.extract_qa_pairs(split=split, limit=limit)

        qa_pairs = self.qa_pairs[:limit] if limit else self.qa_pairs

        contexts = []
        seen_contexts = set()

        for i, pair in enumerate(qa_pairs):
            context = pair.get("context", "")

            # Avoid repeating the same long contract context many times
            if not context or context in seen_contexts:
                continue

            seen_contexts.add(context)

            category = pair.get("category", "Unknown")
            title = pair.get("title", "")
            title = title[:80] + "..." if len(title) > 80 else title

            contexts.append(f"=== CUAD Document {i + 1}: {category} ({title}) ===")
            contexts.append(context)
            contexts.append("")

        return "\n".join(contexts)
