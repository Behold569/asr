import modal

app = modal.App("asr-finetune")

image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install(
    "torch==2.6.0",
    "torchaudio==2.6.0",
    "transformers==4.46.3",
    "datasets==3.1.0",
    "accelerate==1.1.1",
    "tensorboard==2.18.0",
    "jiwer==3.0.5",
    "librosa==0.10.2",
    "soundfile==0.12.1",
)

CPUS = 8
EVAL_STEPS = 50
runs = modal.Volume.from_name("asr-finetune-runs", create_if_missing=True)

STAGES = {
    1: dict(augment=False, lr=1e-5, warmup=100, batch_size=32, eval_batch_size=16, grad_accum=1, max_steps=1500),
    2: dict(augment=True, lr=5e-5, warmup=40, batch_size=8, eval_batch_size=4, grad_accum=4, max_steps=1000),
}


@app.cls(
    image=image,
    gpu="L4",
    cpu=CPUS,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/runs": runs},
    secrets=[modal.Secret.from_name("huggingface")],
)
class Trainer:
    @modal.method()
    def train(
        self,
        dataset: str,
        hf_repo: str,
        stage: int,
        model: str,
        language: str,
        text_column: str,
        train_split: str,
        eval_split: str,
        max_steps: int,
        lr: float,
        batch_size: int,
        grad_accum: int,
    ) -> str:
        import jiwer
        import numpy as np
        import torch
        import torchaudio
        from datasets import Audio, load_dataset
        from huggingface_hub import HfApi
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrainerCallback,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        overrides = dict(max_steps=max_steps, lr=lr, batch_size=batch_size, grad_accum=grad_accum)
        cfg = {**STAGES[stage], **{k: v for k, v in overrides.items() if v}}

        name = hf_repo.replace("/", "_")
        log_path, samples_path = f"/runs/{name}.tsv", f"/runs/{name}_samples.tsv"
        with open(samples_path, "w") as f:
            f.write("step\tref\tpred\n")

        processor = WhisperProcessor.from_pretrained(model, language=language, task="transcribe")
        normalize = BasicTextNormalizer()

        def augment_audio(array, sr):
            audio = array.astype(np.float32)
            if np.random.rand() < 0.5:
                audio = audio + np.random.uniform(0.001, 0.015) * np.random.randn(len(audio))
            if np.random.rand() < 0.5:
                audio = audio * np.random.uniform(0.7, 1.3)
            if np.random.rand() < 0.5:
                speed = np.random.uniform(0.9, 1.1)
                audio = torchaudio.functional.resample(
                    torch.tensor(audio).unsqueeze(0), orig_freq=int(sr * speed), new_freq=sr
                ).squeeze(0).numpy()
            return np.clip(audio, -1.0, 1.0)

        def prepare(example, augment=False):
            audio = example["audio"]
            array = augment_audio(audio["array"], audio["sampling_rate"]) if augment else audio["array"]
            example["input_features"] = processor.feature_extractor(
                array, sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            example["labels"] = processor.tokenizer(example[text_column]).input_ids
            example["duration"] = len(audio["array"]) / audio["sampling_rate"]
            return example

        def process(split, augment):
            ds = load_dataset(dataset, split=split).cast_column("audio", Audio(sampling_rate=16000))
            return (
                ds.shuffle(seed=42)
                .map(
                    prepare,
                    fn_kwargs={"augment": augment},
                    remove_columns=["audio"],
                    writer_batch_size=1,
                    num_proc=1 if cfg["augment"] else CPUS,
                )
                .filter(lambda x: bool(x[text_column]) and x["duration"] <= 30 and len(x["labels"]) <= 448)
            )

        train_ds, eval_ds = process(train_split, cfg["augment"]), process(eval_split, False)

        net = WhisperForConditionalGeneration.from_pretrained(model)
        net.config.apply_spec_augment = net.config.d_model == 1280  # SpecAugment for large models only
        net.config.use_cache = False
        net.config.forced_decoder_ids = None
        net.generation_config.forced_decoder_ids = None
        net.generation_config.language = language
        net.generation_config.task = "transcribe"

        def collate(features):
            batch = processor.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
            )
            labels = processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
            ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
            if (ids[:, 0] == net.config.decoder_start_token_id).all():
                ids = ids[:, 1:]
            batch["labels"] = ids
            return batch

        def compute_metrics(pred):
            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            preds = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            refs = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            with open(samples_path, "a") as f:
                for ref, p in zip(refs[:10], preds[:10]):
                    f.write(f"{trainer.state.global_step}\t{ref.strip()}\t{p.strip()}\n")
            pairs = [(normalize(r), normalize(p)) for r, p in zip(refs, preds)]
            pairs = [(r, p) for r, p in pairs if r.strip()]
            return {
                "wer": np.mean([min(1.0, jiwer.wer(r, p)) for r, p in pairs]),
                "cer": np.mean([min(1.0, jiwer.cer(r, p)) for r, p in pairs]),
            }

        class TsvLogger(TrainerCallback):
            columns = ("epoch", "loss", "eval_loss", "eval_wer", "eval_cer")

            def __init__(self):
                with open(log_path, "w") as f:
                    f.write("step\t" + "\t".join(self.columns) + "\n")

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    row = [state.global_step] + [logs.get(c, "") for c in self.columns]
                    with open(log_path, "a") as f:
                        f.write("\t".join(map(str, row)) + "\n")
                    runs.commit()

        args = Seq2SeqTrainingArguments(
            output_dir="/tmp/out",
            max_steps=cfg["max_steps"],
            learning_rate=cfg["lr"],
            lr_scheduler_type="polynomial",
            lr_scheduler_kwargs={"lr_end": 1e-8, "power": 4},
            warmup_steps=cfg["warmup"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["eval_batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            gradient_checkpointing=True,
            fp16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            eval_on_start=True,
            save_strategy="steps",
            save_steps=EVAL_STEPS,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=128,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=f"/runs/{name}_tb",
        )
        trainer = Seq2SeqTrainer(
            model=net,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collate,
            compute_metrics=compute_metrics,
            processing_class=processor,
            callbacks=[TsvLogger()],
        )
        trainer.train()

        trainer.model.push_to_hub(hf_repo)
        processor.push_to_hub(hf_repo)
        api = HfApi()
        for path, repo_path in ((log_path, "training_log.tsv"), (samples_path, "eval_samples.tsv")):
            api.upload_file(path_or_fileobj=path, path_in_repo=repo_path, repo_id=hf_repo)
        runs.commit()
        return hf_repo


@app.local_entrypoint()
def main(
    dataset: str,
    hf_repo: str,
    stage: int,
    model: str = "",
    language: str = "yo",
    gpu: str = "",
    text_column: str = "transcription",
    train_split: str = "train",
    eval_split: str = "validation",
    max_steps: int = 0,
    lr: float = 0.0,
    batch_size: int = 0,
    grad_accum: int = 0,
):
    options = dict(locals())
    gpu = options.pop("gpu")
    if stage not in STAGES:
        raise ValueError("stage must be 1 or 2")
    if not model:
        if stage == 2:
            raise ValueError("Stage 2 starts from the stage 1 model: pass its HuggingFace repo as --model")
        options["model"] = "openai/whisper-small"
    gpu = gpu or ("L40S" if "large" in options["model"] else "L4")
    print(Trainer.with_options(gpu=gpu)().train.remote(**options))
