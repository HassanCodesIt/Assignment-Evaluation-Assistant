import os
import base64
import re
from dotenv import load_dotenv
from openai import OpenAI

# LOAD .env VARIABLES
load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_ocr_client():
    """Initialize OpenAI client for Hugging Face router."""
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

def perform_ocr(image_path, client=None):
    """
    Performs OCR on the given image using Qwen2.5-VL-32B-Instruct via Hugging Face.
    """
    if client is None:
        client = get_ocr_client()
        
    image_base64 = image_to_base64(image_path)

    # ---------- OCR REQUEST ----------
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract ALL text from this image exactly as written. Do not summarize or infer. Act strictly as an OCR engine.Don't add any extra text or characters."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()

def is_new_question(text):
    """
    Detects whether the text starts a new question.
    """
    # Look for Q1, Question 1, etc. at the start of the text
    return bool(re.match(r"^\s*(Q\d+\.|Question\s+\d+)", text, re.IGNORECASE))

def stitch_text(previous_text, current_text):
    """
    Merge OCR text while preserving continuity across pages.
    """
    if not previous_text:
        return current_text

    if is_new_question(current_text):
        return previous_text + "\n\n" + current_text
    else:
        # Continuation of previous content
        # Check if we need to add a space or if it's already there
        prev = previous_text.rstrip()
        curr = current_text.lstrip()
        
        # Simple heuristic: if previous ends with hyphen, don't add space
        if prev.endswith("-"):
            return prev[:-1] + curr
        
        return prev + " " + curr

if __name__ == "__main__":
    # Test OCR
    TEST_IMAGE = "numbers_handwritten.png"
    if os.path.exists(TEST_IMAGE):
        print("OCR RESULT:\n")
        print(perform_ocr(TEST_IMAGE))
    else:
        print(f"Test image {TEST_IMAGE} not found.")



