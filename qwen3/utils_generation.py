import os 
import re
import torch 

def func_generate_text(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)
            out = out[-1, :]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            token_ids = torch.cat([token_ids, next_token], dim=0)
    return token_ids
            

def token_ids_to_text(token_ids, tokenizer, batch_dim=False):
    if batch_dim:
        flat = token_ids.squeeze(0)  # remove batch dimension
    else:
        flat = token_ids
    return tokenizer.decode(flat.tolist())
            
            
def get_clean_generated_text(generated_text, tokenizer):
    output_text = token_ids_to_text(generated_text, tokenizer)
    
    # Post-processing to remove incomplete special tokens at the end
    incomplete_special_token_pattern = re.compile(r"<\|[^>]*?$")
    output_text = re.sub(incomplete_special_token_pattern, "", output_text)
    output_text = output_text.strip()
    
    return output_text