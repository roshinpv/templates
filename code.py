from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_local_gpt2(save_dir="./local-gpt2"):
    """
    Loads a locally saved GPT-2 model and tokenizer.

    Args:
        save_dir (str): Directory containing the saved model and tokenizer files.

    Returns:
        model: The GPT-2 model.
        tokenizer: The GPT-2 tokenizer.
    """
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(save_dir)
    
    print(f"GPT-2 model and tokenizer loaded successfully from '{save_dir}'.")
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    # Specify the directory where the model is saved
    model_directory = "./local-gpt2"
    
    # Load the locally saved model and tokenizer
    model, tokenizer = load_local_gpt2(save_dir=model_directory)

    # Test with a simple input
    from transformers import pipeline
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = text_generator("Once upon a time", max_length=50, num_return_sequences=1)

    # Display the generated text
    print("\nGenerated Text:")
    print(output[0]['generated_text'])
