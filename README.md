# Medical NLP Pipeline

A production-oriented, **Hybrid (Generative + logic-based)** NLP system for processing doctor-patient conversations. Designed to balance the **fluency** of Large Language Models (LLMs) with the **safety and precision** of deterministic rule-based extraction.

## 1. System Architecture

The pipeline uses a **Parallel Stream** design to ensure clinical safety while providing readable summaries.

1.  **Preprocessor**: Regex-based Speaker Segmentation and Spacy-based Sentence Chunking.
2.  **NER Engine**: Hybrid model using `en_core_sci_md` (SciSpacy) + Regex Rules for critical clinical entities (dosages, timelines, pain levels).
3.  **Sentiment Analyzer**: Transformer-based (DistilBERT/BART) patient-only sentiment and intent detection.
4.  **Dual-Path SOAP Generation**:
    *   **Path A (Deterministic)**: Entities are mapped strictly to SOAP sections (Subjective, Objective, etc.) to guarantee no hallucinations.
    *   **Path B (Generative)**: A T5-based model generates a fluent SOAP narrative.
5.  **Validator**: Consistency checks (e.g., Timeline validation, contradictions) between extracted entities.

## 2. Hallucination Prevention & Safety

- **Deterministic Fallback**: The primary structured data comes from the Rule-Based/NER path, preventing LLM hallucinations from corrupting the medical record.
- **Evidence Linking**: Every rule-extracted entity is linked to its source sentence.
- **Strict Schemas**: Pydantic models enforce output structure for the deterministic path.

## 3. Uncertainty Handling

- **Explicit vs Inferred**: Entities are tagged.
- **Confidence Scores**: Validation layer calculates an overall confidence score based on consistency.
- **Missing Data**: If a section like "Diagnosis" is not found in the deterministic path, it is left empty or marked "None reported".

## 4. Components

| Component | Method | Reason |
| :--- | :--- | :--- |
| **NER** | Hybrid (SciSpacy + Regex) | Models catch general terms; Rules catch specific formats (e.g., "10 sessions"). |
| **Sentiment** | Transformer (DistilBERT/BART) | Context-aware analysis of patient tone without retraining. |
| **SOAP (Safe)** | Deterministic Mapping | Ensures clinical safety; "Symptom" always goes to Subjective/Objective, never Plan. |
| **SOAP (Fluent)** | T5 Transformer (Flan-T5-Base) | Generates a human-readable summary to assist the physician. |

## 5. Known Limitations

- **Coreference Resolution**: "It hurts" - "It" is not resolved to "Neck" in this version.
- **Complex Negation**: "I do not have pain except when I run" might be simplified to "pain" in a basic extraction if not carefully tuned.
- **Model Size**: Uses `google/flan-t5-base` for demonstration; production would use `flan-t5-large` or `flan-t5-xl` for better quality.

## 6. Setup

1.  **Install Dependencies**:
    The system requires specific versions of NumPy and SciSpacy.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will automatically install `en_core_sci_md` from the direct URL.*

2.  **Run the demo**:
    ```bash
    python demo.py
    ```

    **Alternative**:
    Open `PhysicianNotetaker_Emitrr.ipynb` in Google Colab or Jupyter Notebook and run the cells sequentially.

## 7. Sample Failure Cases

- **Ambiguous Speaker**: If the transcript lacks "Doctor:" labels, segmentation fails.
- **Sarcasm**: Sentiment models may miss sarcasms in "Oh, great, another pill."

## 8. Clinical Safety

This system prioritizes **Recall** for symptoms (capture everything) and **Precision** for Diagnosis (don't label guesses). The Validation layer acts as a safety guard.
