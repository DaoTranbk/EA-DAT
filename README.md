# Fuzzy Alignments in Directed Acyclic Graph for Non-autoregressive Machine Translation (FA-DAT)

Implementation for the ICLR 2023 paper "**[``Fuzzy Alignments in Directed Acyclic Graph for Non-autoregressive Machine Translation``](https://openreview.net/forum?id=LSz-gQyd0zE)**".

**Abstract**: We introduce a fuzzy alignment objective between the directed acyclic graph and
reference sentence based on n-gram matching, aiming to handle the training data multi-modality.

**Highlights**: 
* FA-DAT **outperforms DA-Transformer baseline by 1.1 BLEU** on raw WMT17 ZH<->EN dataset.
* FA-DAT **achieves the performance of the autoregressive Transformer** on raw WMT14 EN<->DE & WMT17 ZH-EN dataset with **fully parallel decoding (13× speedup)**.


**Features**:

- We provide **[``numba``](https://github.com/numba/numba) implementations (optional)** for dynamic programming in the calculation of fuzzy alignment to speedup the training. 
- We also provide the functions implemented in **[``PyTorch``](https://github.com/pytorch/pytorch) (by default)**.

**Files**:

- Most codes of the framework are from [``DA-Transformer``](https://github.com/thu-coai/DA-Transformer). We mainly add the following files as plugins.

   ```
   FA-DAT
   └── fs_plugins
       └── criterions
           ├── nat_dag_loss_ngram.py                   # fuzzy alignment loss
           └── pass_prob.py                            # numba implementation of dynamic programming in loss

   ```
- This repo is forked from [``DA-Transformer``](https://github.com/thu-coai/DA-Transformer), which is modified from [``fairseq:5175fd``](https://github.com/pytorch/fairseq/tree/5175fd5c267adceec9445bf067597686e159e7e7). You can refer to above repos for more information.

**Below is a guide to replicate the results reported in the paper. We give an example of experiments on WMT14 En-De dataset.**

## Requirements & Installation
### Requirements
* Python >= 3.7
* Pytorch == 1.10.1 (tested with cuda == 11.3)
* gcc >= 7.0.0 (for compiling cuda operations in NLL pretraining, as recommended in [``DA-Transformer``](https://github.com/thu-coai/DA-Transformer))
* (Optional) numba == 0.56.2

### Installation
* ``git clone --recurse-submodules https://github.com/DaoTranbk/EA-DAT.git``
* ``cd FA-DAT && pip install -e .``


## Data Preprocess
Fairseq provides the preprocessed raw datasets [here](http://dl.fbaipublicfiles.com/nat/original_dataset.zip). Please build the binarized dataset by the following script:

```bash
input_dir=path/to/raw_data        # directory of raw text data
data_dir=path/to/binarized_data   # directory of the generated binarized data
src=en                            # source language id
tgt=de                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.${src}-${tgt} --validpref ${input_dir}/valid.${src}-${tgt} --testpref ${input_dir}/test.${src}-${tgt} \
    --src-dict ${input_dir}/dict.${src}.txt --tgt-dict ${input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 32
```

## Negative Log-likelihood Pre-training

At the pre-training stage, we use a batch size of approximating 64k tokens **(GPU number * max_tokens * update_freq == 64k)**.

Run the following script for pre-training:

```bash
data_dir=/path/to/binarized_data
checkpoint_dir=/path/to/checkpoint_dir

fairseq-train ${data_dir}  \
    --user-dir fs_plugins \
    --task translation_lev_modified  --noise full_mask \
    --arch glat_decomposed_link_base \
    --decoder-learned-pos --encoder-learned-pos \
    --share-all-embeddings --activation-fn gelu \
    --apply-bert-init \
    --links-feature feature:position --decode-strategy lookahead \
    --max-source-positions 128 --max-target-positions 1024 --src-upsample-scale 8.0 \
    --criterion nat_dag_loss \
    --length-loss-factor 0 --max-transition-length 99999 \
    --glat-p 0.5:0.1@200k --glance-strategy number-random \
    --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 \
    --lr-scheduler inverse_sqrt  --warmup-updates 10000   \
    --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    --ddp-backend c10d \
    --max-tokens 4096  --update-freq 4 --grouped-shuffling \
    --max-update 200000 --max-tokens-valid 4096 \
    --save-interval 1  --save-interval-updates 10000  \
    --seed 0 \
    --valid-subset valid \
    --validate-interval 1       --validate-interval-updates 10000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --fixed-validation-seed 7 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric   \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --log-format 'simple' --log-interval 100

```

## Fuzzy Alignment Training

At the fuzzy alignment training stage, we use a batch size of approximating 256k tokens **(GPU number * max_tokens * update_freq == 256k)**.

Run the following script for fuzzy alignment training:

```
ngram_order=2
pretrained_model_dir=/path/to/model
data_dir=/path/to/binarized_data
checkpoint_dir=/path/to/checkpoint_dir


fairseq-train ${data_dir}  \
    --user-dir fs_plugins \
    --task translation_lev_modified  --noise full_mask \
    --arch glat_decomposed_link_base \
    --decoder-learned-pos --encoder-learned-pos \
    --share-all-embeddings --activation-fn gelu \
    --finetune-from-model ${pretrained_model_dir} \
    --links-feature feature:position --decode-strategy lookahead \
    --max-source-positions 128 --max-target-positions 1024 --src-upsample-scale 8.0 \
    --criterion nat_dag_loss_ngram --max-ngram-order ${ngram_order} \
    --length-loss-factor 0 --max-transition-length 99999 \
    --glat-p 0.1:0.1@4k --glance-strategy number-random \
    --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 \
    --lr-scheduler inverse_sqrt  --warmup-updates 500   \
    --clip-norm 0.1 --lr 0.0002 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    --ddp-backend=legacy_ddp \
    --max-tokens 200  --update-freq 328 --grouped-shuffling \
    --max-update 5000 --max-tokens-valid 256 \
    --save-interval 1  --save-interval-updates 500 \
    --seed 0 \
    --valid-subset valid \
    --validate-interval 1       --validate-interval-updates 500 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --fixed-validation-seed 7 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --log-format 'simple' --log-interval 5
    
```
To apply numba implementation for fuzzy alignment loss, add the following command-line argument:
* `` --numba-ngram-loss ``

## Inference

Like DA-Transformer, three fully-parallel decoding strategies can be used:

* **Greedy**:  Fastest, use argmax in token prediction and transition prediction.
* **Lookahead**: Similar speed as Greedy, but higher quality. Consider transition prediction and token prediction together
* **[Joint-Viterbi](https://aclanthology.org/2022.findings-emnlp.322/)**: Slightly slower than Lookahead but higher quality. Support length penalty to control the output length.

**We provide an example of employing the strategy of Joint-Viterbi decoding below.**

### Average Checkpoints

We average the parameters of the best 5 checkpoints, empirically leading to a better performance.

```bash
checkpoint_dir=/path/to/checkpoint_dir
average_checkpoint_path=/path/to/checkpoint_dir/average.pt

python3 ./fs_plugins/scripts/average_checkpoints.py --inputs ${checkpoint_dir} \
                --max-metric --best-checkpoints-metric bleu --num-best-checkpoints-metric 5 --output ${average_checkpoint_path}
```

### Joint-Viterbi Decoding

Joint-Viterbi decoding has been proposed in "**[Viterbi Decoding of Directed Acyclic Transformer for Non-Autoregressive Machine Translation](https://aclanthology.org/2022.findings-emnlp.322/)**". It uses the length penalty parameter ``decode_viterbibeta`` to control the output length. Joint-Viterbi decoding finds the output that maximizes P(A,Y|X) / |Y|^{beta}.

You need to specify ``decode_strategy`` to ``jointviterbi`` to enable the Joint-Viterbi decoding.

```bash
# Viterbi
data_dir=/path/to/binarized/data_dir
average_checkpoint_path=/path/to/checkpoint_dir/average.pt

fairseq-generate ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --max-tokens 4096 --seed 0 \
    --model-overrides "{\"decode_strategy\":\"jointviterbi\",\"decode_viterbibeta\":0.1}" \
    --path ${average_checkpoint_path}
```
For WMT17 En-Zh, please add ``--source-lang en --target-lang zh --tokenizer moses --scoring sacrebleu --sacrebleu-tokenizer zh``.

**Note: ``decode_viterbibeta`` should be tuned for each translation task on the development set. Following the Viterbi-decoding paper, we tune the parameter to obtain a results with similar translation length as lookahead decoding.**

For the convenience of reproduction, we provide the settings of ``decode_viterbibeta`` used in FA-DAT on all tasks below:

| Task |  En-De  |  De-En  | Zh-En  | En-Zh     |
| ---- | ---- | ---- | ---- | ---- |
| ``decode_viterbibeta`` |  0.1  |  3.5  | 3.5  | 3.0    |


## Citing

Please kindly cite us if you find our papers or codes useful.

FA-DAT:

```
@inproceedings{ma2023fuzzy,
   author = {Zhengrui Ma and Chenze Shao and Shangtong Gui and Min Zhang and Yang Feng},
   title = {Fuzzy Alignments in Directed Acyclic Graph for Non-Autoregressive Machine Translation},
   booktitle = {International Conference on Learning Representations},
   year={2023},
   url={https://openreview.net/forum?id=LSz-gQyd0zE}
}
```

DA-Transformer:

```
@inproceedings{huang2022DATransformer,
  author = {Fei Huang and Hao Zhou and Yang Liu and Hang Li and Minlie Huang},
  title = {Directed Acyclic Transformer for Non-Autoregressive Machine Translation},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning, {ICML} 2022},
  year = {2022}
}
```

Viterbi Decoding:

```
@inproceedings{shao2022viterbi,
  author = {Chenze Shao and Zhengrui Ma and Yang Feng},
  title = {Viterbi Decoding of Directed Acyclic Transformer for Non-Autoregressive Machine Translation},
  booktitle = {Findings of EMNLP 2022},
  year = {2022}
}
```
