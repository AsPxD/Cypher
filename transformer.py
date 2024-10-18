import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_name='gpt2'):
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def calculate_perplexity(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')

def classify_code(model, tokenizer, code_sample, threshold=50):
    perplexity = calculate_perplexity(model, tokenizer, code_sample)
    print(f"Perplexity: {perplexity}")
    
    if perplexity < threshold:
        return "Likely AI-generated"
    else:
        return "Likely Human-written"

def main():
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("Failed to load model and tokenizer. Exiting.")
        return

    code_sample = """
    def fourSum(nums, target):
        nums.sort()
        result = []
        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                left, right = j + 1, len(nums) - 1
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if current_sum < target:
                        left += 1
                    elif current_sum > target:
                        right -= 1
                    else:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
        return result
    """
    
    result = classify_code(model, tokenizer, code_sample)
    print(result)

if __name__ == "__main__":
    main()