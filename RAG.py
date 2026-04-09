import os
import re
from groq import Groq
import chromadb
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def extraction(file_path):
    text = extract_text(file_path)
    return text


def sanitize_collection_name(name):
    """Sanitizes names to comply with ChromaDB's 3-63 char, alphanumeric requirements."""
    # Replace non-alphanumeric with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9]', '-', name.strip().lower())
    # Ensure it starts and ends with alphanumeric
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    # Ensure length (3-63)
    if len(sanitized) < 3:
        sanitized = (sanitized + "db-")[:3]
    return sanitized[:63]

def vectordbadd(storage, sub):
    sub = sanitize_collection_name(sub)
    print(f"DEBUG: Sanitized collection name: {sub}")
    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="./vector_db")
    collection = chroma_client.get_or_create_collection(name=sub) 

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(storage)

    # Initialize Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings for each chunk
    embeddings = embedding_model.embed_documents(chunks)

    # Generate IDs
    current_count = len(collection.get()["ids"])
    ids = [f"id{current_count + i}" for i in range(len(chunks))]

    # Add to gallery
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    return ids


def vectordbget(sub, query, top_chunks=3):
    sub = sanitize_collection_name(sub)
    chroma_client = chromadb.PersistentClient(path="./vector_db")
    collection = chroma_client.get_or_create_collection(name=sub)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_chunks
    )
    
    # Extract documents from results
    documents = results["documents"][0]
    return documents    


def llm(prompt, context):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Format context for the prompt
    context_str = "\n\n".join(context) if isinstance(context, list) else context
    
    system_prompt = f"You are an educational assistant. Use the provided context to answer the user request.\n\nContext:\n{context_str}\n\nTask: {prompt}\n\nNotes:\n- If the information requested is entirely missing from the context, state that clearly.\n- Provide reference locations (e.g., Para 5 or Line 6) in brackets when possible.\n- Avoid using markdown formatting.\n\nResponse:"
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": system_prompt
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )
    
    full_answer = ""
    for chunk in completion:
        full_answer += chunk.choices[0].delta.content or ""
    return full_answer

def evaluate_answer(subject, question, student_answer, instructions=None, max_marks=10):
    # 1. Retrieve relevant context from vector DB
    context = vectordbget(subject, question)
    
    # 2. Use LLM to evaluate student answer against context
    instruction_text = f"\n\nAdditional Evaluation Instructions from Teacher: {instructions}" if instructions else ""
    
    eval_prompt = (
        f"You are a lenient grader. Evaluate the student's answer based on the provided context.\n\n"
        f"!!! CRITICAL: TEACHER'S SPECIAL INSTRUCTIONS !!!\n"
        f"{instruction_text if instruction_text else 'No special instructions.'}\n"
        f"-> If the teacher says 'let it slide' or 'ignore mistakes', you MUST give {max_marks}/{max_marks} regardless of minor errors.\n\n"
        f"GENERAL GRADING RUBRIC:\n"
        f"1. BASELINE {max_marks}/{max_marks}: If the student explains the CORE concepts correctly, start with {max_marks}/{max_marks}. Only deduct for major scientific misconceptions.\n"
        f"2. IGNORE TYPOS/OCR: Handwritten answers often have OCR errors or minor typos (e.g., 'frictionaly' instead of 'friction'). DO NOT deduct for these.\n"
        f"3. OUT OF CONTEXT INFO: If the student adds extra info not in the textbook, IGNORE IT. Do not penalize for 'extra' knowledge.\n"
        f"4. DEPTH VS BREVITY: As long as the answer is more than 2 lines and hits the main points, it is sufficient for full marks.\n"
        f"5. POSITIVE FEEDBACK: Focus 90% of your feedback on what the student got right.\n\n"
        f"Question: {question}\n\n"
        f"Student Answer: {student_answer}\n\n"
        f"Format the FIRST line as: '[[SCORE: X/{max_marks}]]' (where X is the points awarded).\n"
        f"Then provide your encouraging feedback below."
    )
    
    evaluation_text = llm(eval_prompt, context)
    
    # Robust score extraction
    # 1. Try to find [[SCORE: X/MAX]] or [[SCORE: X]]
    score_match = re.search(r'\[+SCORE:\s*(\d+(?:\.\d+)?)(?:/\d+(?:\.\d+)?)?\s*\]+', evaluation_text, re.IGNORECASE)
    
    if not score_match:
        # 2. Try to find any "X/max_marks" in the text
        score_match = re.search(rf'(\d+(?:\.\d+)?)\s*/\s*{max_marks}', evaluation_text)
        
    if not score_match:
        # 3. Try to find "Score is X" or "Score: X"
        score_match = re.search(r'Score\s*(?:is|:)\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)

    if score_match:
        try:
            score = float(score_match.group(1))
            # Convert to int if it's a whole number
            if score.is_integer():
                score = int(score)
        except:
            score = 0
    else:
        score = 0
    
    # Clean up the text for display
    # Remove the score tag and any redundant grading phrases
    clean_text = re.sub(r'\[+SCORE:.*?\]+', '', evaluation_text, flags=re.IGNORECASE).strip()
    clean_text = re.sub(r'(?i)The score for this answer (would be|is):?\s*\d+/\d+', '', clean_text).strip()
    clean_text = re.sub(r'(?i)Score:\s*\d+/\d+', '', clean_text).strip()
    
    return {
        "text": clean_text,
        "score": score,
        "context": context
    }
