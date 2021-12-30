# Imposing Relation Structure in Language-Model Embeddings Using Contrastive Learning
Official code repository of "Imposing Relation Structure in Language-Model Embeddings Using Contrastive Learning" paper 
(PyTorch implementation).
For a description of the models and experiments, see our paper: https://arxiv.org/abs/2109.00840 (published at CoNLL 2021).

## Setup
### Requirements
 - Python 3.5+
 - PyTorch (tested with version 1.3.1)
 - scikit-learn (tested with version 0.24.2)
 - tqdm (tested with version 4.55.1)
 - numpy (tested with version 1.17.4)
 - rapids (tested with version 0.19)
 - seaborn (tested with version 0.11.2)

### Execution steps
- Download the CharacterBERT model \[1\] using the official <a target="_blank" href="https://github.com/helboukkouri/character-bert">repository</a>.
- Run the ```data_preprocessing.py``` script to process the data ```ade_full_spert.json```.
- Run the ```offline_graph_creation.py``` script to extract the graphs for each sentence.
- Train the CLDR and CLNER models by running the ```main.py``` script under the corresponding folder. There are two modes, the tuning mode, and the final run. If you want to execute only the final run, consult the ```final_run_epochs.xlsx``` file to find the number of training epochs per split (cross-validation).
- Run the ```embeddings_RE_NER_jointly.py``` to extract the trained embeddings.
- Solve the NER and RE tasks by running the KNN classifiers under the ```/classification/``` folder. The "final run" mode is implemented.
- Run the ```evaluation.py``` script to extract the evaluation metrics per task. Strict evaluation \[2\] is used. 
- For the tSNE \[3\] analysis, first run the ```dataset_creation.py``` script to extract the dataset in a particular format (.hdf5 file). Then run the ```tSNE_*.py``` scripts to create the plots. <br>

*** For each execution step the corresponding (if any) config file (/configs/) should be updated accordingly. Importantly, change the <i>split_id</i> number. 


## Notes
Please cite our work when using this software.

Theodoropoulos, C., Henderson, J., Coman, A. C., & Moens, M. F. (2021). Imposing Relation Structure in Language-Model Embeddings Using Contrastive Learning. arXiv preprint arXiv:2109.00840.

BibTex:
<pre>
@inproceedings{theodoropoulos2021imposing,
  title={Imposing Relation Structure in Language-Model Embeddings Using Contrastive Learning},
  author={Theodoropoulos, Christos and Henderson, James and Coman, Andrei Catalin and Moens, Marie Francine},
  booktitle={Proceedings of the 25th Conference on Computational Natural Language Learning},
  pages={337--348},
  year={2021}
}
</pre>


## References
```
[1] Hicham El Boukkouri, et al. 2020. "Characterbert: Reconcilingelmo and bert for word-level open-vocabulary rep-resentations  from  characters." In Proceedings of the 28th International Conference on Computational Linguistics, pages 6903–6915.
[2] Bruno Taillé, et al. 2020. "Let’s stop error propagation in the end-to-end relation extraction literature!" In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), pages 3689–3701.
[3] Laurens  Van  der  Maaten  and  Geoffrey  Hinton.  2008. "Visualizing  data  using  t-sne." Journal  of  MachineLearning Research, 9(11).
```