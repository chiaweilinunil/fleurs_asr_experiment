"""CLI: fine-tune an ASR model (added only after zero-shot is reproducible).

Must support a tiny smoke-test mode (--max-train-samples, --max-eval-samples,
--num-train-epochs). Train on train split, select on dev, test only at the end.
"""
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
import argparse
import torch
from fleurs_asr.data import decode_audio
from fleurs_asr.models import load_asr_model
from fleurs_asr.data import load_fleurs_split
from fleurs_asr.evaluation import compute_corpus_metrics

def prepare_example(example, processor):
    array, sr = decode_audio(example)
    feats = processor.feature_extractor(array, sampling_rate=sr).input_features[0]
    labels = processor.tokenizer(example["raw_transcription"]).input_ids
    
    return  {"input_features": feats, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    
    def __call__(self, features): 
        
        # pad and tranform audio features into a batch tensor
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # pad and transform the label sequences into a batch tensor 
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding token ids with -100
        labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
        )
        
        # drop the 
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch


def run_training(language, model_name="openai/whisper-small", output_dir="results/checkpoints", max_train_samples=None, max_eval_samples=None, num_train_epochs=1, batch_size=8, learning_rate=1e-5,seed=42):
    
    # load model and config
    model, processor = load_asr_model(model_name)
    # Generation settings moved from model.config to model.generation_config in
    # recent transformers; setting them on model.config now errors at generate().
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    # load and preprocess train and eval data
    train = load_fleurs_split(language, "train", max_samples=max_train_samples)
    eval_ = load_fleurs_split(language, "validation", max_samples=max_eval_samples)
    train = train.map(lambda ex: prepare_example(ex, processor), remove_columns=train.column_names)
    eval_ = eval_.map(lambda ex: prepare_example(ex, processor), remove_columns=eval_.column_names)
    
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id   # undo the -100 masking
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        metrics = compute_corpus_metrics(label_str, pred_str)
        return {"wer": metrics["wer"], "cer": metrics["cer"]}

    use_cuda = torch.cuda.is_available()
    args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    fp16=use_cuda,                   
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,      
    generation_max_length=225,
    logging_steps=10,
    seed=seed,
    report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=eval_,
    data_collator=collator,
    compute_metrics=compute_metrics,
    processing_class=processor,        # newer transformers; older used tokenizer=processor.feature_extractor
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a FLEURS language.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="results/checkpoints")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_training(args.language, args.model, args.output_dir,
                 args.max_train_samples, args.max_eval_samples,
                 args.num_train_epochs, args.batch_size)

if __name__ == "__main__":
    main()