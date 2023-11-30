data_name=iwslt14
data_dir=data-bin/iwslt14.tokenized.de-en
arch=glat_decomposed_link_small
checkpoint_dir=/cm/archive/daotc2/EA-DAT/checkpoints/$data_name"_"$arch


CUDA_VISIBLE_DEVICES=5 fairseq-train ${data_dir}  \
    --user-dir fs_plugins \
    --task translation_lev_modified  --noise full_mask \
    --arch ${arch} \
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
    --max-tokens 4096  --update-freq 1 --grouped-shuffling \
    --max-update 200000 --max-tokens-valid 4096 \
    --save-interval 1  --save-interval-updates 10000  \
    --seed 0 \
    --valid-subset test \
    --validate-interval 1       --validate-interval-updates 10000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --skip-invalid-size-inputs-valid-test \
    --fixed-validation-seed 7 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric   \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --log-format 'simple' --log-interval 100 \
    --wandb-project EA-DAT