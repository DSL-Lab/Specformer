# Specformer
Code of [Specformer: Spectral Graph Neural Networks Meet Transformers](https://openreview.net/forum?id=0pdSt3oyJa1)

# How to run
- For node-level task, e.g., signal regression and node classification, you should first run preprocess_node_data.py to generate .pt files for each dataset.
- For graph-level taks, you can direcly run dgl_main.py.

# Q&A
Any suggestion/question is welcome.

# Reference
If you make advantage of Specformer in your research, please cite the following in your manuscript:

```
@inproceedings{specformer2023,
  author={Deyu Bo and 
          Chuan Shi and
          Lele Wang and
          Renjie Liao},
  title={Specformer: Spectral Graph Neural Networks Meet Transformers},
  booktitle = {{ICLR}},
  publisher = {OpenReview.net},
  year      = {2023}
}
```
