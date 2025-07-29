# Music Genre Transfer Using Deep Generative Architectures

This project explores the task of **music genre transfer**—modifying the stylistic features of an audio clip to match a **target genre**, while preserving its **melody, rhythm, and structure**. We experiment with a range of **deep generative models**, including Transformer, CycleGAN, VAE, StarGAN, and a novel use of **MusicGen**.

---

## Objectives
- Apply deep learning models for **audio-to-audio genre transformation**
- Compare models on **style transfer quality**, **musical coherence**, and **genre fidelity**
- Showcase the creative use of **prompt-guided MusicGen** for real-world genre transfer

---

## Dataset

**GTZAN Dataset**  
- 1,000 songs (30s each) across 10 genres: *blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock*
- All audio converted to **Mel spectrograms** using `librosa`
- Normalized, padded/truncated for uniform shape (~1300×128)

---

## Models Explored

| Model          | Description                                                                                  |
|----------------|----------------------------------------------------------------------------------------------|
| **Transformer** | Sequence-to-sequence architecture using attention; struggled with coherence & stability     |
| **CycleGAN**    | Unpaired genre translation with adversarial & cycle-consistency loss                        |
| **VAE**         | Latent-space reconstruction with KL-regularization; blurry outputs on longer clips          |
| **StarGAN**     | Multi-domain genre transfer using one generator & discriminator; effective but not perfect |
| **MusicGen**    | Text-conditioned transformer used for prompt-based genre transfer on input audio            |

---

## Experimental Highlights

| Model        | Pros                                                                 | Cons                                                                   |
|--------------|----------------------------------------------------------------------|------------------------------------------------------------------------|
| Transformer  | Theoretical interpretability                                         | Fails to capture long-range temporal structure                         |
| CycleGAN     | Decent structure preservation                                        | Weak genre transfer for dissimilar pairs                               |
| VAE          | Smooth reconstruction on short clips                                | Fails with 30s clips; lacks genre clarity                              |
| StarGAN      | Scalable to many genres; learns genre-specific rhythm/instrumentation | Some outputs retain traces of input genre                             |
| MusicGen     | Best results overall; fully automated and melodic                    | Black-box model; limited control over genre blending                   |

---

## Spectrogram Examples

| Input Genre | Target Genre | Output (Model) |
|-------------|--------------|----------------|
| Pop         | Classical     | CycleGAN       |
| Classical   | Jazz          | StarGAN        |
| Hip-Hop     | Jazz          | MusicGen       |

> *Audio samples and visualizations available in the respective notebooks for each approach*

---

## Evaluation Criteria
- **Style Fidelity**: Does the output match the prompted/target genre?

- **Content Preservation**: Is melody/rhythm retained?

- **Spectrogram Coherence**: Are frequency-time structures preserved?

- **Qualitative Listening**: Subjective audio quality through human perception


