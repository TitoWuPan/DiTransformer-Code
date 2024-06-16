Code4ML: a Large-scale Dataset of annotated Machine Learning Code

Description
This is an enriched version of Code4ML: a Large-scale Dataset of annotated Machine Learning Code, a corpus of Python code snippets, competition summaries, and data descriptions from Kaggle. The initial corpus consists of ≈ 2.5 million snippets of ML code collected from ≈ 100 thousand Jupyter notebooks. A representative fraction of the snippets is annotated by human assessors through a user-friendly interface specially designed for that purpose. 

The data is organized as a set of tables in CSV format. It includes several central entities: raw code blocks collected from Kaggle (code_blocks.csv), kernels (kernels_meta.csv) and competitions meta information (competitions_meta.csv). Manually annotated code blocks are presented as a separate table (murkup_data.csv). As this table contains the numeric id of the code block semantic type, we also provide a mapping from the id to semantic class and subclass (vertices.csv).

Snippets information (code_blocks.csv) can be mapped with kernels meta-data via kernel_id. Kernels metadata is linked to Kaggle competitions information through comp_name. To ensure the quality of the data kernels_meta.csv includes only notebooks with an available Kaggle score.

Automatic classification of code_blocks are stored in data_with_preds.csv. The mapping of this table with code_blocks.csv can be doe through code_blocks_index column, which corresponds to code_blocks indices.

The updated Code4ML 2.0 corpus includes kernels retrieved from Code Kaggle Meta. These kernels correspond to the kaggle competitions launched since 2020. The natural descriptions of the competitions are retrieved with the aim of LLM. 

kernels_meta2.csv may contain kernels without Kaggle score, but with the place in the leader board (rank).

Code4ML 2.0 dataset can be used for various purposes, including training and evaluating models for code generation, code understanding, and natural language processing tasks. 

https://zenodo.org/records/11213783
