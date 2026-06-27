# Phase 1 - Observation

## Goal

To visually investigate whether attention sink behaviour appears in Qwen3-1.7B and whether it differs across languages or prompt types.

---

## 1. Overall Attention Behaviour
```
layer0_avg_attention
all_layer_avg_attention
```
- Observation

  Many query tokens consistently allocate part of their attention to the first token. Local attention remains clearly visible in Layer 0, as indicated by the strong diagonal attention pattern.

  The first-column attention remains visible after averaging across all layers.
  

---

## 2. Head Specialization
```
layer0_all_head_attention
layer0_head0_attention
```

- Observation

  Different attention heads appear to specialize in different attention patterns.

  For the selected example prompt, head 0, 4, 6, 8, 10, 13 have attention scores that mostly concentrate in the first token.
  While head 1, 3, 5, 7, 9, 12, 14 have a strong diagonal attention score with little focus on the first token.
  And some other heads have long-range attention.

  Layer 0 Head 0 exhibits both local attention and noticeable attention toward the first token.



---

## 3. Layer Progression
```
layer0_avg_attention
all_layer_avg_attention
```

- Observation

  Layer O average attention shows both local attention pattern and a visible focus on the first token
  
  After averaging across all layers, the diagonal local-attention pattern becomes weaker, while attention toward the first token appears relatively stronger

---

## 4. Sink Score Statistics
```
distribution_of_sink_scores.png"
```
Average English sink score: 0.748557

Average Vietnamese sink score: 0.729932
...

The sink-score distribution appears unimodal and concentrated around 0.73–0.75 rather than uniformly distributed, suggesting that sink behaviour is relatively consistent across the sampled prompts.


Highest observed:	english_question_01 | english |	question | 4 | 0.836914

n | file | language | category | seq_len | sink_score |
-----|----------|----------|---------|------------|----|
30 | english_question_01 | english | question |	4 | 0.836914|
36	| english_question_07 | english	|question | 5 | 0.825000|
34	| english_question_05	|english |question | 5 | 0.821875|
31	| english_question_02	|english	|question | 6 |	0.805339|
32	| english_question_03	|english	|question	| 6	| 0.804036|
33	| english_question_04	|english	|question | 7 |	0.794085|
35	| english_question_06	|english	|question	|7	| 0.780692|
3	| english_code_04	|english|	code|	6|	0.779948|
26	| english_list_07	|english|	list|	6|	0.776693|
27	| english_list_08	|english	|list	|6	|0.776693|

The highest sink scores are predominantly observed in short English question prompts (4–7 tokens), suggesting that prompt length may influence the current sink-score metric.

...

Lowest observed: english_story_01 |	english |	story |	21 | 0.671317

n |file|	language|category|	seq_len	|sink_score
---|------|-------|------|------|------|
40	|english_story_01	|english	|story	|21	|0.671317 |
90	|vietnamese_story_01|	vietnamese|	story	|22	|0.685369|
96	|vietnamese_story_07	|vietnamese|	story	|18|	0.687500|
93	|vietnamese_story_04	|vietnamese	|story	|20	|0.691602|
58  |vietnamese_code_09	|vietnamese	|code	|15	|0.692448|
46	|english_story_07	|english	|story	|14	|0.693359|
48	|english_story_09	|english	|story	|14	|0.694475|
99	|vietnamese_story_10|	vietnamese|	story	|16|	0.694580|
41	|english_story_02	|english	|story	|14	|0.694754|
45	|english_story_06	|english	|story	|14	|0.695033|

This list contains equal number of vietnamese and english prompts. The lowest sink scores are mostly associated with longer story prompts (14–22 tokens), indicating that sink score may decrease as sequence length increases.

---

## 5. English vs Vietnamese
```
eng_vie_avg_attention
eng_vie_sink_boxplot
```

- Observation

  Both heatmaps show similar attention pattern, a bright verticle stripe where the the first token is. The primary visible difference is that Vietnamese prompts are generally tokenized into longer sequences, resulting in a longer diagonal attention structure.

  From the boxplot, the median sink score for english is ~0.745 and for vietnamese it's ~0.735, relatively close. Although english's sink scores have a larger variance and IQR, and one outlier. While vietnamese's sink score remain more stable.



---

## 6. Limitation

- - The current visualizations are generated from relatively short prompts. Longer prompts may exhibit different attention behaviours. Further experiments with longer contexts will be conducted in later phases.
- - Visualisation graphs are limited by using 1 prompt for layer0_head0_attention, layer0_avg_attention, while eng_vie_avg_attention take the average of the whole dataset. Move from single-example visualisation to fully dataset-level visualisation for next phases.
- - The current sink score is a simple proxy metric based only on attention allocated to the first token. It does not capture all possible forms of sink behaviour.

## 7. Initial Conclusions

Attention sink appears to be visually observable.

Some attention heads exhibit specialized behaviour.

English and Vietnamese show broadly similar attention patterns.

The observations support the existence of sink-like attention patterns in Qwen3-1.7B under the current experimental setup.

Further quantitative analysis is required before drawing stronger conclusions.

---

## 8. Questions for Phase 2
- Are sink heads consistent across prompts?

- Which layers contribute most to sink behaviour?

- Is the current sink-score metric sensitive to sequence length?

- Can sink score be improved beyond the current proxy metric?

- How does this compare with StreamingLLM?

...
