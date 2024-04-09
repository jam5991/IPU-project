from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_model():
    model_name = "distilbert-base-uncased"
    dataset_name = "imdb"
    num_train_epochs = 3
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Load and preprocess the dataset
    dataset = load_dataset(dataset_name)
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512), batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels") # Ensure the label column is named 'labels'
    tokenized_dataset.set_format('torch')
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./model',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).astype(float).mean()}
    )
    
    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

if __name__ == "__main__":
    train_model()
