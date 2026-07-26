# BIP 2026 Proposal — File Analysis & Landscape Scan

## File Inventory

| File | Type | Size | Summary |
|------|------|------|---------|
| `Propuesta BIP 2026.md` | Markdown | 2,513 B | Proposal for IEEE BIP 2026 paper on cross-modal KL-divergence as uncertainty signal for glaucoma detection using MedGemma and MM-ODIR dataset. Post-hoc, training-free, single-pass. |

## Core Themes from Proposal

1. **Cross-modal representation disagreement**: Using KL divergence between vision and text features as an uncertainty signal.
2. **Medical Vision-Language Models (MedVLMs)**: Target model is MedGemma (pre-trained, no fine-tuning).
3. **Dataset**: MM-ODIR (multi-modal ODIR-5K with text prompts).
4. **Training-free & single-pass**: Post-hoc uncertainty estimation without model modification.
5. **Application**: Glaucoma detection from fundus images.
6. **Metrics**: AUPRC, Brier score, t-test for statistical significance.
7. **Inspiration**: "Between the Layers Lies the Truth" paper — intra-layer local information scores / signature maps.

## Landscape Scan — Key Findings

### Paper Inspirador: "Between the Layers Lies the Truth"
- **Authors**: Zvi Badash et al. (arXiv 2603.22299v1, March 2026).
- **Method**: Views each layer's post-MLP activation as a probability distribution (temperature-scaled softmax over hidden dimension), computes pairwise directed KL divergence between layers, yielding an L×L "signature map". A LightGBM classifier predicts correctness from these maps.
- **Key properties**: Compact (L² << d_hidden), single forward pass, transferable across datasets, robust to 4-bit quantization.
- **Metrics**: AUPRC and Brier score (1 - MSE between predicted correctness probabilities and binary ground truth).
- **Positioning**: Between classic probing (high-dimensional, task-specific) and Information Bottleneck (global mutual information, impractical online).
- **Code availability**: Not yet verified; paper is very recent (March 2026).

### MedGemma
- **Variants**: 4B Multimodal (vision+text), 27B Text-only, 27B Multimodal.
- **Architecture**: Based on Gemma 3. 4B has 32-layer transformer, hidden dim 4096, 32 heads. Vision encoder is MedSigLIP (400M params, SigLIP-400M tuned on 33M medical image-text pairs). Supports 896×896 images.
- **API**: Available via medgemma.org with API keys. Also on Hugging Face.
- **Pretraining**: Includes ophthalmology data among radiology, histopathology, dermatology.
- **Key challenge**: How to extract intermediate vision and text features (not just final output) from a single forward pass.

### MM-ODIR / ODIR-5K Dataset
- **Source**: Beijing Ophthalmology and Optometry Institute, Shanggong Medical Technology Co., Ltd.
- **Size**: 5,000 patients, 7,000 fundus images (left + right eyes). 3,500 training cases released publicly.
- **Labels**: 8 categories — Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), AMD (A), Hypertension (H), Myopia (M), Other (O). Multi-label.
- **Format**: JPG, varied resolutions (Canon, Zeiss, Kowa cameras).
- **Paired data**: Both eyes per patient, share same label at patient level.
- **Class imbalance**: Normal and Diabetes ~2,100 cases; Hypertension ~200.
- **Challenge**: Need to construct or use text prompts (MM-ODIR implies multi-modal with text, but ODIR-5K is primarily image-only with labels; may need synthetic prompts or use doctor's diagnostic keywords if available).

### Cross-Modal Uncertainty / Disagreement
- **Contrastive loss**: MedSigLIP uses sigmoid contrastive loss on image-text pairs — directly relevant to cross-modal similarity.
- **Related concepts**: Cross-modal retrieval, vision-text alignment, multimodal fusion.
- **Gap**: No prior work specifically using KL divergence between vision and text branch distributions as a training-free uncertainty signal in MedVLMs for ophthalmology.

### Training-Free Uncertainty Methods
- **Output-based**: Entropy, margin, token probability heuristics — cheap but brittle.
- **Probing**: Hidden state classifiers (e.g., Azaria & Mitchell 2023) — effective but high-dimensional and task-specific.
- **Bayesian surrogates**: MC-Dropout, deep ensembles — expensive.
- **Signature maps**: The paper inspirador offers a compact alternative for LLMs; our proposal adapts this to cross-modal VLMs.

### BIP 2026
- **Venue**: IEEE BIP (Biomedical Imaging and Processing) — needs confirmation of exact conference name and dates.
- **Deadline**: July 31, 2026 (per proposal).
- **Format**: 6-8 pages IEEE format.

## Gaps Identified

1. **Code for signature maps**: Need to verify if the inspirador paper has released code or if we need to reimplement from scratch.
2. **MedGemma feature extraction**: How to extract intermediate hidden states from both vision and text branches in a single forward pass? Hugging Face `transformers` `output_hidden_states=True` likely works.
3. **MM-ODIR text prompts**: The original ODIR-5K is image-only with labels. Need to determine if MM-ODIR adds text descriptions or if we need to construct synthetic prompts (e.g., "Does this image show glaucoma?").
4. **Cross-modal KL divergence**: How to align vision and text distributions? Vision features are spatial/temporal patches; text features are token sequences. Need a common representation or pooling strategy.
5. **Statistical validation**: What baseline to compare against? Proposal mentions entropy baseline.
6. **Clinical interpretability**: How to explain KL divergence to clinicians? Connection to XAI pillar of thesis.

## Consolidated Theme List (for Dimension Decomposition)

1. Signature maps / intra-layer information scores (paper inspirador)
2. MedGemma architecture & feature extraction APIs
3. Cross-modal disagreement & KL divergence theory
4. MM-ODIR dataset curation & prompt engineering
5. Glaucoma detection & medical evaluation metrics
6. Training-free uncertainty estimation methods
7. Implementation details (Hugging Face, PyTorch, code)
8. BIP 2026 submission requirements & related work
9. Tesis doctoral extension (4 pilares)
10. Baselines & ablation design
