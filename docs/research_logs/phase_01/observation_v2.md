# Phase 1 – New Observations

## Observation 1 – Attention sinks emerge progressively with depth

Attention sinks are weak or nearly absent in the earliest layers. Their strength increases rapidly in the early-middle layers and continues strengthening towards the final layers of the network.

This suggests that attention sinks are an emergent property of deep representations rather than an inherent characteristic of the attention mechanism.

---

## Observation 2 – Local attention persists alongside sink attention

Although attention towards the first token becomes increasingly dominant in later layers, local diagonal attention remains visible throughout the network.

This indicates that sink attention complements rather than replaces local contextual attention.

---

## Observation 3 – Similar behaviour across English and Vietnamese prompts

No substantial qualitative differences were observed between English and Vietnamese prompts under the current experimental setting.

Layer-wise sink progression is highly similar across both languages.

Further experiments with larger datasets are required before drawing stronger conclusions.

---

## Observation 4 – Sink strength depends on the metric definition

Different sink-score definitions produce different absolute values and different sensitivities to sequence length.

Among the evaluated metrics, **mean_from_k** appears to provide the most stable measurements while being less influenced by prompt length.

This metric is selected as the primary sink metric for Phase 2.

---

## Observation 5 – Sequence length is a major confounding variable

Vietnamese prompts consistently produce longer token sequences than their English counterparts (approximately 1.7× longer on average).

Several sink metrics exhibit noticeable correlation with sequence length.

Future experiments should therefore control for token count when comparing different languages or prompt categories.

---

## Observation 6 – Sink behaviour is consistent across prompts

Although individual prompts exhibit minor variation, the overall layer progression remains remarkably consistent.

This suggests that the observed sink behaviour reflects model-level characteristics rather than prompt-specific artefacts.

---

## Observation 7 – Layer depth influences sink strength more than language

Variation across transformer layers is substantially larger than variation between languages.

Layer depth therefore appears to be the dominant factor governing sink formation in Qwen3-1.7B.

---

## Limitations

Current observations are based on:

- 30 prompts
- two languages
- one model (Qwen3-1.7B)
- global attention only

These observations should therefore be interpreted as preliminary characterisation rather than definitive conclusions.

Further validation will be performed during Phase 2.
