# Design Decisions & Answers

## 1. Medical NLP Summarization Questions

### Q: How would you handle ambiguous or missing medical data?
**Answer:**
1.  **Strict Schema w/ Nulls**: As implemented in `models.py`, fields are Optional. If data is missing (e.g., no explicit Diagnosis), output `null`/`None` rather than guessing.
2.  **Certainty Scoring**: Use an enum (`Explicit`, `Inferred`, `Uncertain`).
    *   *Explicit*: "I have a headache."
    *   *Inferred*: "My head feels heavy" (requires reasoning).
    *   *Uncertain*: "I think it might be a migraine?"
3.  **Evidence Linking**: Always attach the source span. If the model is unsure, the human reviewer can check the `evidence` field.
4.  **Confidence Thresholds**: If the extraction confidence is below 0.7, flag it for review or discard it.

### Q: What pre-trained NLP models would you use for medical summarization?
**Answer:**
1.  **NER**: `en_core_sci_scibert` or `en_core_sci_md` (used in this project) for robust entity detection.
2.  **Summarization (Hybrid Approach)**:
    *   **Extractive**: Use deterministic mapping (like this project's `SOAPMapper`) for safety-critical fields.
    *   **Generative**: Use **T5-Base-Clinical** or **Flan-T5** for generating fluent narratives (like the `GenerativeSOAPModel` in the notebook uses `google/flan-t5-base` as a proof of concept).
    *   **BioGPT**: Microsoft's model trained specifically on biomedical literature for more complex queries.

---

## 2. Sentiment & Intent Analysis Questions

### Q: How would you fine-tune BERT for medical sentiment detection?
**Answer:**
1.  **Data Preparation**: Collect medical dialogue datasets (e.g., MedDialog). Annotate specifically for *patient* utterances (masking doctor speech).
2.  **Label Schema**: Adapt labels to: `Anxious` (fear/worry), `Reassured` (relief), `Frustrated` (pain/waiting), `Neutral`.
3.  **Domain Adaptation**: Start with `ClinicalBERT` or `BlueBERT` (already pre-trained on MIMIC notes) rather than generic `bert-base`.
4.  **Loss Function**: Use Weighted Cross-Entropy if classes are imbalanced (e.g., rarely "Reassured").

    > **Note (Notebook Implementation):** In the provided notebook (`PhysicianNotetaker_Emitrr.ipynb`), we use a Zero-Shot approach with **`facebook/bart-large-mnli`** for intent classification and **`distilbert-base-uncased-finetuned-sst-2-english`** for sentiment, allowing for immediate categorization without extensive fine-tuning data.

### Q: What datasets would you use for training a healthcare-specific sentiment model?
**Answer:**
1.  **MIMIC-III / MIMIC-IV**: Clinical notes (though mostly physician written, they contain "Patient complains of..." sections).
2.  **MedDialog**: Large dataset of doctor-patient conversations.
3.  **eHealthForum / WebMD**: User-generated questions (rich in patient sentiment/anxiety).
4.  **ConceptNet**: For augmenting emotional words related to pain/illness.

---

## 3. SOAP Note Generation Questions

### Q: How would you train an NLP model to map transcripts into SOAP format?
**Answer:**
1.  **Architecture**: Encoder-Decoder model (e.g., **LongT5** or **Ledger**) is ideal because transcripts are long.
2.  **Input Representation**: Tag speakers `<DOC> ... <PAT> ...` to help the model distinguish sources (Subjective comes from PAT, Assessment from DOC).
3.  **Objective**: Sequence-to-Sequence training where Input = Transcript, Target = Structured SOAP JSON (linearized) or Sectioned Text.
4.  **Alignment**: Use an auxiliary loss to align sentences to sections (Sentence Classification task + Generation task).

### Q: What rule-based or deep-learning techniques would improve accuracy?
**Answer:**
**Rule-Based (Safety Layer):**
*   **Section Filtering**: Hard-code logic that "Physical Examination" keywords forces content into **Objective**.
*   **Negation Handling**: (Implemented in this project) `NegEx` or dependency parsing to ensure "No tenderness" isn't listed as a symptom.
*   **Ontology Lookup**: Map terms to **SNOMED-CT**. If it's not a valid medical term, don't put it in "Diagnosis".

**Deep Learning (Fluency Layer):**
*   **Pointer-Generator Networks**: Allows the model to *copy* exact dosages/numbers from the transcript (ensures "10mg" doesn't hallucinate to "20mg").
*   **Consistency Discriminator**: A secondary model that checks if the generated Summary entails the input Transcript.
*   **Hybrid Verification**: As demonstrated in this pipeline, run both Generative and Rule-based systems in parallel and flag anything where the Generative model disagrees with the extracted entities.
