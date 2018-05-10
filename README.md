# Multi-level Skim-RNN

Multi-level Skim-RNN is an extension of [Skim-RNN](https://arxiv.org/abs/1711.02085), which skims in multiple levels by expanding the number of LSTM cells in a skim cell more than 2 cells.

This project contains code for training and running an extractive question answering model on the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/), based on [this project](https://github.com/google/mipsqa).

example running command:
```bash
python train_and_eval.py --output_dir out/squad/with_pretrain_100_20_10_5_0 --big2nested --skim_1 --skim_2 --num_cells 5 --hidden_size 100 --small_hidden_sizes '[20, 10, 5, 0]' --root_data_dir prepro/sort_filter --restore_dir out/squad/baseline --restore_step 5001 --num_train_steps 15000 --glove_dir ~/data/glove
```