# neural-attention
Tensorflow implementation of [Reasoning about Entailment with Neural Attention](https://arxiv.org/abs/1509.06664)


## Dataset
[The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
contains 3 datasets `snli_1.0_train.txt`, `snli_1.0_dev.txt`, `snli_1.0_test.txt`

```python
# contradiction: A contradicts B
"A person on a horse jumps over a broken down airplane."
"A person is at a diner, ordering an omelette."

# entailment: A implies B
"A person on a horse jumps over a broken down airplane."
"A person is outdoors, on a horse."

# neutral: A neither proves nor disproves B
"A person on a horse jumps over a broken down airplane."
"A person is training his horse for a competition."
```
