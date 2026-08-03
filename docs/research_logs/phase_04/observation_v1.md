# Phase 4 - Functional Perturbation Analysis

## Transfer validation
### Observation
Before conducting any interventions, the frozen sink-head set identified in Phase was evaluated on held-out text to verify that sink specialisation generalised beyond the discovery corpus.

The scatter plot shows a clear positive relationship between Phase 2 sink scores and the held-out sink scores. Although the absolute sink scores are systematically lower on the held-out corpus, the highest-scoring Phase 2 sink heads remain concentrated among the highest-scoring heads in the new corpus

Rank agreement remains high, with Spearman's correlation of p = 0.871,
indicating that the relative ordering of sink-specialised heads in largely preserved.

The accompanying table further shows that the vast majority of frozen sink heads continue to exhibit elevated sink attention on held-out text.

### Interpretation
This confirms that sink specialisation is not an artefact of he benchmark used during discovery.

The reduction in absolute sink scores indicates that sink magnitude depends somewhat on corpus characteristics, but the strong rank correlation demonstrates that sink specialization is an intrinsic property of particular attention heads rather than a dataset-specific phenomenon.

Consequently, the perturbation experiments target a genuinely stable subset of sink-specialized heads rather than heads selected through circular evaluation.

### Conclusion
The frozen sink-head set generalizes successfully to unseen data, validating the experimental design and providing a reliable basis for all subsequent interventions.

## Functional Importance of Sink Heads
### Observation
3 interventions scopes were compared:
- global sink removal
- sink-head removal
- matched non-sink head removal

The KL-divergence curves reveal clear differences

Global perturbation produces the largest distributional change throughout the sequence.

Sink-head perturbation consistently produces larger KL divergence than matched non-sink perturbation, with repeated peaks across the sequence, although both remain substantially below the global intervention

However, next-token prediction changes only slightly. The corresponding NLL table reports approximately

| Intervention |   Mean ΔNLL |
| ------------ | ----------: |
| Global       | ~0.022 nats |
| Non-sink     | ~0.027 nats |
| Sink         | ~0.008 nats |

The bootstrap confidence interval overlap substantially.

The accompanying statistical tests show that none of these pairwise differences reaches conventional statistical significance after multiple-comparison correction.

### Interpretation
The KL analysis demonstrates that sink perturbation alters the internal probability distribution more than perturbing matched non-sink heads.

However, these internal representational changes do not translate into proportionally large increases in language-model loss.

This discrepancy suggests that sink attention contributes primarily to internal computation rather than directly determining next-token predictions.

Moreover, the absence of statistically significant ΔNLL differences indicates that the model possesses sufficient redundancy to compensate for sink-head removal under ordinary full-context inference.

### Conclusion
Sink-specialized heads influence internal attention dynamics, but removing them alone produces only modest functional degradation during standard inference.

## BOS versus Control Positions
### Observation
The BOS token was compared with neighbouring control positions.

Removing BOS produces a consistently larger increase in ΔNLL than removing Position 1.

The mean degradation is approximately
| Intervention      | Mean ΔNLL |
| ----------------- | --------: |
| Remove BOS        |   ~0.0075 |
| Remove Position 1 |    ~0.003 |

The statistical table reports a positive estimated difference favouring BOS, although confidence intervals remain relatively broad.

### Interpretation
Although both interventions have relatively small effects, BOS consistently produces approximately twice the degradation of a nearby token.

This indicates that BOS possesses genuine functional specialization rather than simply benefiting from being located near the beginning of the sequence.

Nevertheless, the absolute magnitude remains small, suggesting that BOS contributes to computation without being individually indispensable.

### Conclusion
The BOS token exhibits measurable functional importance beyond neighbouring positions, but its contribution remains modest during conventional language modelling.

## Partial sink supression
## Observation
Sink attention was progressively weakened using retention factors

γ = 1.0, 0.75, 0.50, 0.25 and 0.0

Several consistent trends emerge.

The KL divergence increases monotonically as retained sink attention decreases.

Top-1 agreement decreases smoothly from almost perfect agreement at γ = 1 toward approximately 95% after complete removal.

In contrast, ΔNLL remains close to zero throughout the entire trajectory.

Only complete removal produces a small positive increase, and even then the confidence interval remains large.

The statistical table confirms that intermediate attenuation levels are not significantly different from the baseline.

### Interpretation
This represents a classic dose-response experiment.

Internal representations become progressively more different as sink attention is weakened, yet predictive accuracy remains remarkably stable.

The gradual rather than abrupt degradation indicates that sink computation is continuously useful rather than functioning as an all-or-none mechanism.

Furthermore, the weak effect on ΔNLL suggests that other computational pathways can compensate for reduced sink attention during ordinary inference.

### Conclusion
Sink attention behaves as a redundant computational resource whose influence accumulates gradually rather than exhibiting a sharp failure threshold.

## Query Projection
### Observation
Instead of directly removing sink attention, this intervention projected query vectors away from the BOS direction.

The resulting ΔNLL increased dramatically to approximately 2.7 nats, exceeding every previous intervention by roughly two orders of magnitude.

The bootstrap confidence interval is comparatively narrow, indicating that this effect is highly stable across evaluation windows.

### Interpretation
Unlike previous interventions, this manipulation alters the query representation before attention weights are computed.

Consequently, it disrupts not only BOS attention but also the broader geometric computation performed by the attention head.

Combined with Phase 3, where sink behaviour was shown to originate from head-specific query projections, this result demonstrates that the query pathway itself constitutes an essential computational component.

Importantly, this experiment should not be interpreted as evidence that BOS attention alone is indispensable.

Rather, it demonstrates that the query representation supporting sink behaviour is deeply integrated into the attention computation.

### Conclusion
The query pathway is fundamentally important for transformer computation, whereas sink attention itself represents only one manifestation of that underlying mechanism.

## Streaming Proxy
### Observation
The streaming-style evaluation produces the clearest functional separation of the entire phase.

Retaining one or four sink tokens yields NLL trajectories that remain close to the full-context baseline throughout the sequence.

In contrast, removing all retained sink tokens causes an immediate and persistent increase in NLL.

After the streaming window becomes active, the model without retained sinks stabilizes around 8–10 nats, whereas retaining sink tokens maintains performance close to approximately 3 nats, similar to full-context inference.

The corresponding summary table reports a large and statistically significant increase in average ΔNLL when sink retention is disabled.

### Interpretation
Unlike previous experiments, this intervention places the model in precisely the computational regime where attention sinks are hypothesized to matter.

When historical context is no longer fully available, retained sink tokens provide stable global reference points that preserve prediction quality.

The dramatic separation between retained and removed conditions strongly supports the hypothesis that sink tokens primarily facilitate long-context computation rather than standard full-context inference.

This result also reconciles the apparent contradiction observed in Experiments 1–3.

Under unconstrained inference, sink removal has only modest consequences because full historical information remains accessible.

Under streaming constraints, however, sink visibility becomes functionally critical.

### Conclusion
Attention sinks play their most important role under memory-constrained attention, supporting lone-context stability rather than ordinary next-token prediction.

## Conclusion
Taken together, the perturbation experiments reveal a consistent hierarchy of functional importance.

Transfer validation demonstrates that sink specialization generalizes beyond the discovery corpus, establishing that the interventions target stable sink heads.

Removing sink attention changes internal probability distributions but produces only modest increases in prediction error during ordinary full-context inference. 
Even progressive weakening causes smooth representational drift with little degradation in language modelling performance, indicating substantial computational redundancy.

However, perturbing the underlying query representation causes catastrophic degradation, confirming the central mechanistic role of the query pathway identified in Phase 3.

Finally, under streaming-style attention constraints, retained sink tokens become essential for maintaining prediction quality. This provides strong evidence that attention sinks function primarily as stability mechanisms that preserve useful computation when historical context is restricted, rather than serving as indispensable contributors to standard full-context language modelling.




