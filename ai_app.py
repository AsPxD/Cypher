import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_name='gpt2'):
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Calculate perplexity
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
        st.error(f"Error calculating perplexity: {e}")
        return float('inf')

# Classify code based on perplexity
def classify_code(model, tokenizer, code_sample, threshold=50):
    perplexity = calculate_perplexity(model, tokenizer, code_sample)
    st.write(f"Perplexity: {perplexity:.4f}")
    
    if perplexity < threshold:
        return "Likely AI-generated"
    else:
        return "Likely Human-written"

# Streamlit App
def main():
    st.title("Code Classification using GPT-2 Perplexity")
    
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        st.error("Failed to load model and tokenizer. Exiting.")
        return
    
    st.write("Enter a code sample to classify whether it is AI-generated or Human-written.")
    
    # Text input for code sample
    code_sample = st.text_area("Code Sample", height=300, value="""
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
    """)
    
    # Add a button to classify the code
    if st.button("Classify Code"):
        if code_sample.strip():
            classification_result = classify_code(model, tokenizer, code_sample)
            st.write(f"Classification Result: **{classification_result}**")
        else:
            st.warning("Please input some code to classify.")

if __name__ == "__main__":
    main()
