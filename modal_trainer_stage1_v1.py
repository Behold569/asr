import modal

app = modal.App("asr-finetune-stage1")

image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install(
    "torch==2.6.0",
    "torchaudio==2.6.0",
    "transformers==4.46.3",
    "datasets==3.1.0",
    "accelerate==1.1.1",
    "peft==0.13.2",
    "jiwer==3.0.5",
    "librosa==0.10.2",
    "soundfile==0.12.1",
    "tensorboard==2.18.0",
)

CPUS = 8
runs = modal.Volume.from_name("asr-finetune-runs", create_if_missing=True)


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
        model: str = "openai/whisper-small",
        language: str = "yo",
        text_column: str = "transcription",
        train_split: str = "train",
        eval_split: str = "validation",
        max_steps: int = 1500,
        lr: float = 1e-5,
        eval_steps: int = 50,
        batch_size: int = 32,
        grad_accum: int = 1,
        spec_augment: bool = False,
    ) -> str:
        from dataclasses import dataclass

        import jiwer
        import numpy as np
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

        name = hf_repo.replace("/", "_")
        log_path, samples_path = f"/runs/{name}.tsv", f"/runs/{name}_samples.tsv"
        with open(samples_path, "w") as f:
            f.write("step\tref\tpred\n")

        processor = WhisperProcessor.from_pretrained(model, language=language, task="transcribe")
        normalize = BasicTextNormalizer()

        def prepare(example):
            audio = example["audio"]
            example["input_features"] = processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            example["labels"] = processor.tokenizer(example[text_column]).input_ids
            example["duration"] = len(audio["array"]) / audio["sampling_rate"]
            return example

        def process(split):
            ds = load_dataset(dataset, split=split).cast_column("audio", Audio(sampling_rate=16000))
            return (
                ds.shuffle(seed=42)
                .map(prepare, remove_columns=["audio"], writer_batch_size=1, num_proc=CPUS)
                .filter(lambda x: bool(x[text_column]) and x["duration"] <= 30 and len(x["labels"]) <= 448)
            )

        train_ds, eval_ds = process(train_split), process(eval_split)

        net = WhisperForConditionalGeneration.from_pretrained(model)
        net.config.apply_spec_augment = spec_augment
        net.config.use_cache = False
        net.config.forced_decoder_ids = None
        net.generation_config.forced_decoder_ids = None
        net.generation_config.language = language
        net.generation_config.task = "transcribe"

        @dataclass
        class Collator:
            def __call__(self, features):
                batch = processor.feature_extractor.pad(
                    [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
                )
                labels = processor.tokenizer.pad(
                    [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
                )
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
            max_steps=max_steps,
            learning_rate=lr,
            lr_scheduler_type="polynomial",
            lr_scheduler_kwargs={"lr_end": 1e-8, "power": 4},
            warmup_steps=100,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            fp16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=eval_steps,
            eval_on_start=True,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=128,
            remove_unused_columns=False,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=net,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=Collator(),
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
    model: str = "openai/whisper-small",
    language: str = "yo",
    gpu: str = "L4",
    text_column: str = "transcription",
    train_split: str = "train",
    eval_split: str = "validation",
    max_steps: int = 1500,
    lr: float = 1e-5,
    batch_size: int = 32,
    grad_accum: int = 1,
    spec_augment: bool = False,
):
    print(
        Trainer.with_options(gpu=gpu)().train.remote(
            dataset=dataset,
            hf_repo=hf_repo,
            model=model,
            language=language,
            text_column=text_column,
            train_split=train_split,
            eval_split=eval_split,
            max_steps=max_steps,
            lr=lr,
            batch_size=batch_size,
            grad_accum=grad_accum,
            spec_augment=spec_augment,
        )
    )
