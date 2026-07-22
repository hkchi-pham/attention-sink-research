## Experiment 2A — Sink Score Characterisation
### Observation

The baseline head profile shows that all 16 attention heads exhibit relatively high sink scores, ranging from approximately 0.57 to 0.68. No head displays a negligible sink score, indicating that attention sink behaviour is not isolated to a small subset of heads. However, sink strength varies consistently across heads, suggesting functional specialization. Heads 10, 11 and 14 exhibit the strongest sink behaviour, while Head 9 shows the weakest average sink score.

The layer profile demonstrates a clear progression of sink behaviour throughout the network. Sink scores remain close to zero in the first two layers, increase sharply around Layer 3, fluctuate through the middle layers, and then steadily increase toward the final layers where the strongest sink behaviour is observed (approximately 0.85–0.88).

### Analysis

This quantitative result confirms the qualitative observations made in Phase 1.

Rather than appearing immediately after embedding, attention sinks emerge during the early transformer blocks and become progressively stronger as representations are refined. The gradual increase across depth suggests that sink behaviour is an emergent property of the computation performed by successive transformer layers, rather than being hard-coded into the embedding or positional encoding alone.

The variation between heads further indicates that although sink behaviour is widespread, different attention heads contribute unequally. Some heads consistently allocate much more attention to the first token than others, motivating the mechanistic analysis in Phase 3 to investigate why particular heads become stronger sink heads.

## Experiment 2B — Position vs Content
### Observation

Five different first-token conditions were evaluated:
- BOS
- Common word
- Function word
- Gibberish token
- Rare token

Across all conditions, the global sink score remains very similar (approximately 0.60–0.62) with overlapping confidence intervals.

The layer progression curves almost completely overlap across all five conditions. Every condition follows the same trajectory: sink behaviour emerges after the first few layers and gradually strengthens toward the deeper layers.

### Analysis

Changing the semantic identity of the first token produces only negligible changes in sink strength.

This suggests that the attention sink is primarily determined by positional information rather than token content.

The result supports the original hypothesis that sink formation is largely position-dependent. Whether the first token is meaningful, meaningless, common or rare, later layers continue allocating comparable attention mass to that position.

## Experiment 2C — Context Length Scaling
### Observation

Global sink score decreases consistently as context length increases:

| Length | Mean sink |
| ------ | --------- |
| 128    | ≈0.523    |
| 512    | ≈0.441    |
| 1024   | ≈0.422    |
| 2048   | ≈0.418    |

The decrease is approximately monotonic.

The layer progression plots show that all context lengths retain the same overall layer-wise shape. However, longer contexts consistently produce lower sink scores across almost every layer.

### Analysis

This result does not support the original hypothesis.

The hypothesis predicted that sink strength would increase with longer contexts.

Instead, the measured sink score decreases as sequence length grows.

Importantly, this does not necessarily imply that attention sinks become weaker.

The sink score used in this study is normalized over many more query positions when the sequence becomes longer. As additional tokens compete for attention, the average attention allocated to the first token may decrease even if the first token still functions as an attention sink.

The preservation of the overall layer-wise shape suggests that the mechanism responsible for sink formation remains consistent across sequence lengths, while only its measured magnitude changes.

This unexpected result motivates further investigation into how sink metrics scale with sequence length and whether alternative normalization methods better capture sink behaviour.

This finding is consistent with previous literature that proposes the first token acts as a stable attention anchor during autoregressive decoding.

### Initial Interpretation
At first glance, these results appear to suggest that longer contexts weaken attention sinks. If interpreted literally, this would imply that the model relies less on the first token as sequence length increases.

### Metric Validation

However, this interpretation is potentially misleading because the sink metric averages over all valid query positions (queries ≥ k). As context length increases, many additional query positions are included in the average. Those newly added late-sequence queries naturally contain more variation and lower attention-to-token-0 values, reducing the global average even if the sink behaviour for earlier queries remains unchanged.

To test whether the observed decrease reflects a genuine weakening of the sink or simply a consequence of averaging over more query positions, a fixed-window validation was performed. Instead of averaging over every valid query, the metric was recomputed using only the first 128 valid query positions for every context length.

The fixed-window metric remains almost perfectly constant across context lengths (approximately 0.53 for all four settings), whereas the original metric decreases steadily.

This comparison indicates that the apparent reduction in sink strength is primarily caused by the evaluation metric rather than by a fundamental change in the attention mechanism.

When the same portion of the sequence is compared across all context lengths, attention to the first token remains essentially unchanged. The layer-wise trajectories also remain nearly identical after normalization, suggesting that the mechanism responsible for attention sinks is stable across context lengths.

Therefore, the original decrease in global sink score should be interpreted as a consequence of averaging over an increasingly large set of later query positions, rather than evidence that longer contexts intrinsically weaken attention sinks

The query-position profile further supports this conclusion. Across all context lengths, the curves overlap almost perfectly within the first 128 query positions, demonstrating that the model allocates nearly identical attention to the first token over this shared region.

Beyond this overlap, longer contexts simply extend the curve to additional query positions. These later queries exhibit greater variability and generally lower attention to the first token, which reduces the overall average when they are included in the metric. Consequently, the observed decrease in global sink score reflects the inclusion of more late-sequence queries rather than a change in sink behaviour itself.

## Experiment 2D — Multilingual Comparison
### Observation

The head profile shows highly similar sink behaviour across English and Vietnamese.

Across almost every attention head, Vietnamese exhibits slightly higher sink scores than English, although the differences are small.

Similarly, the layer profile demonstrates nearly identical progression across network depth. Vietnamese consistently remains marginally above English throughout most layers.

The paired sentence analysis also shows that most matched sentence pairs have higher sink scores in Vietnamese. The average global sink score increases from approximately 0.613 (English) to 0.629 (Vietnamese).

The distribution plots reveal substantial overlap between the two languages despite the higher Vietnamese mean.

### Analysis

The results indicate that attention sink behaviour generalizes across languages.

Although Vietnamese consistently exhibits slightly stronger sink scores, the overall layer-wise evolution and head specialization remain remarkably similar.

This suggests that attention sinks are largely language-independent features of the transformer architecture rather than phenomena tied to a specific language.

The modest increase observed for Vietnamese may arise from differences in tokenization, average sequence length, or token distribution rather than from fundamentally different attention mechanisms.

Further mechanistic analysis is required to determine whether these differences originate from shared Key/Value representations or from language-specific query behaviour.

