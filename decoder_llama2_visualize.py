import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 
def generate_response(model, tokenizer, prompt, system_prompt, layers):
    formatted_prompt = (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n"
        f"{prompt} [/INST]"
    )
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        input_ids = input_ids.cuda()
    attention_mask = input_ids.new_ones(input_ids.shape)
    with torch.no_grad():
        outputs = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
        hidden_states = [outputs.hidden_states[layer] for layer in layers]
        layer_probs = [torch.softmax(model.lm_head(hidden_state), dim=-1) for hidden_state in hidden_states]
        k = 10
        top_probs_and_indices = [torch.topk(probs[0, -1], k=k) for probs in layer_probs]
        top_tokens = [[indices.tolist() for indices in top_indices] for _, top_indices in top_probs_and_indices]
        top_tokens_decoded = [[tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in layer_tokens] for layer_tokens in top_tokens]
        top_probs = [[probs.tolist() for probs in top_probs] for top_probs, _ in top_probs_and_indices]
        response = tokenizer.decode(top_tokens[0][0], skip_special_tokens=True)
    return response.strip(), top_tokens_decoded, top_probs
 
def main():
    # Load the model and tokenizer with Hugging Face API token
    tokenizer = AutoTokenizer.from_pretrained("../../models/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("../../models/Llama-2-7b-chat-hf")
 
    system_prompt = "You are a helpful AI assistant, who answers very concisely. If you can give the answer in one word, you should do so."
    prompts = [
        "What is the capital of France?",
        "What is the opposite of hot?",
        "What is the capital of Texas?",
        "What is the largest planet in our solar system?",
        "What is the currency of Japan?",
        "What is the tallest mammal?",
        "What is the most populous country in the world?",
        "What is the smallest continent?",
        "What is the fastest land animal?",
        "What is the chemical symbol for gold?",
    ]
   
    layers = [8, 16, 24, 32]
 
    # Create the "figures" folder if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")
 
    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}")
        response, top_tokens_decoded, top_probs = generate_response(model, tokenizer, prompt, system_prompt, layers)
        print(f"Response: {response}\n")
 
        # Plot the probabilities of the top 10 tokens for each layer
        num_layers = len(layers)
        fig, axs = plt.subplots(num_layers, 1, figsize=(10, 10 * num_layers), dpi=120)
        plt.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots
       
        for j, layer in enumerate(layers):
            axs[j].bar(range(len(top_tokens_decoded[j])), top_probs[j])
            axs[j].set_xlabel("Top 10 Tokens")
            axs[j].set_ylabel("Probability")
            axs[j].set_title(f"Probabilities of Top 10 Tokens (Layer {layer})")
            axs[j].set_xticks(range(len(top_tokens_decoded[j])))
            axs[j].set_xticklabels(top_tokens_decoded[j], rotation=45)
            axs[j].set_ylim(top=axs[j].get_ylim()[1] * 1.1)  # Add padding to the top
            axs[j].set_ylim(bottom=axs[j].get_ylim()[0] * 0.9)  # Add padding to the bottom
 
        # Save the figure to the "figures" folder
        plt.savefig(f"figures/prompt_{i+1}.png")
        plt.close(fig)
 
if __name__ == "__main__":
    main()