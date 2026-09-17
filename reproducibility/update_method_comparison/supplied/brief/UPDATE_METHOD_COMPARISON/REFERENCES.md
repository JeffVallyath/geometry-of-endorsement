# References and exact scope

Primary comparison paper:
- Bojie Li, Models Take Notes at Prefill: KV Cache Can Be Editable and Composable, arXiv:2606.17107v1.
- https://arxiv.org/abs/2606.17107v1
- https://arxiv.org/html/2606.17107v1
- First read Section 4 and Appendix B. Section 3 supplies mechanism context; it is not a requirement to reproduce the full circuit study.

Released implementation:
- https://github.com/19PINE-AI/programmable-kv
- https://github.com/19PINE-AI/programmable-kv/blob/main/editkv/README.md
- https://github.com/19PINE-AI/programmable-kv/blob/main/e2/README.md
- https://github.com/19PINE-AI/programmable-kv/blob/main/esys/README.md
- https://github.com/19PINE-AI/programmable-kv/blob/main/results/README.md

These links identify the inspected sources, not a pinned runtime. Resolve and record a commit before implementation; preserve the published version and any code/paper differences. Public results are reported findings, not measurements independently rerun for this handoff. The README documents token-length limits and using only the current correction rather than a stacked history. Preserve these qualifications.

Additional background relevant to the comparison, not another assigned reading queue:
- ReFT: https://arxiv.org/abs/2404.03592v3
- RAVEL: https://aclanthology.org/2024.acl-long.470/
- MQuAKE: https://aclanthology.org/2023.emnlp-main.971/
- RippleEdits: https://aclanthology.org/2024.tacl-1.16/
- KVEraser: https://arxiv.org/abs/2606.17034v2

Do not infer that a named method was tested on the supplied task just because it appears in this file.
