from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional, List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from RAG import extraction, vectordbadd, vectordbget, llm, evaluate_answer
from ocr import perform_ocr, get_ocr_client, stitch_text
import os
import json

app = FastAPI()

# Ensure static and templates directories exist, although user only mentioned static
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("./savepdf", exist_ok=True)
os.makedirs("./vector_db", exist_ok=True)

ASSIGNMENTS_FILE = "assignments.json"

def save_assignment(data):
    assignments = []
    if os.path.exists(ASSIGNMENTS_FILE):
        with open(ASSIGNMENTS_FILE, "r") as f:
            assignments = json.load(f)
    assignments.append(data)
    with open(ASSIGNMENTS_FILE, "w") as f:
        json.dump(assignments, f, indent=4)

def get_assignments():
    if os.path.exists(ASSIGNMENTS_FILE):
        with open(ASSIGNMENTS_FILE, "r") as f:
            return json.load(f)
    return []


@app.get("/")
def home():
    return FileResponse("templates/index.html")


@app.post("/upload")
async def upload(subject: str = Form(...), files: List[UploadFile] = File(...)):
    processed_files = []
    for file in files:
        file_path = f"./savepdf/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = extraction(file_path)
        vectordbadd(text, subject)
        processed_files.append(file.filename)
    
    return {"message": "Uploaded & processed", "filenames": processed_files, "subject": subject}


@app.get("/query")
def query_page():
    return FileResponse("templates/query.html")


@app.post("/query")
def query(user_query: str = Form(...), subject: str = Form(...)):
    chunks = vectordbget(subject, user_query)
    answer = llm(user_query, chunks)  # directly send list to llm
    return {"response": answer}


@app.get("/evaluate")
def evaluate_page():
    return FileResponse("templates/evaluate.html")


@app.post("/evaluate")
async def evaluate(
    subject: str = Form(...), 
    questions: List[str] = Form(...), 
    total_marks: int = Form(...),
    instructions: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    client = get_ocr_client()
    full_extracted_text = ""
    
    for file in files:
        file_path = f"./savepdf/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 1. OCR (Single page)
        page_text = perform_ocr(file_path, client)
        
        # 2. Stitch
        full_extracted_text = stitch_text(full_extracted_text, page_text)
    
    # 3. Evaluate each question. We divide total_marks among questions for internal scoring.
    # However, for the user, we focus on the total.
    num_questions = len(questions)
    mark_per_q = total_marks / num_questions if num_questions > 0 else 0
    
    results = []
    total_score = 0
    combined_context = []
    
    for q_text in questions:
        eval_data = evaluate_answer(subject, q_text, full_extracted_text, instructions, mark_per_q)
        results.append({
            "question": q_text,
            "max": mark_per_q,
            "score": eval_data["score"],
            "feedback": eval_data["text"]
        })
        total_score += eval_data["score"]
        if isinstance(eval_data["context"], list):
            combined_context.extend(eval_data["context"])
        else:
            combined_context.append(eval_data["context"])
    
    # Deduplicate context strings
    unique_context = list(dict.fromkeys(combined_context))
    
    # Final total score adjustment (ensure it doesn't exceed total_marks due to rounding)
    total_score = min(total_score, total_marks)
        
    return {
        "extracted_text": full_extracted_text,
        "results": results,
        "total_score": round(total_score, 1),
        "total_max": total_marks,
        "context": unique_context
    }


@app.get("/ocr")
def ocr_page():
    return FileResponse("templates/ocr.html")


@app.post("/ocr_detect")
async def ocr_detect(files: List[UploadFile] = File(...)):
    client = get_ocr_client()
    full_extracted_text = ""
    
    for file in files:
        file_path = f"./savepdf/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Perform OCR
        page_text = perform_ocr(file_path, client)
        
        # Stitch
        full_extracted_text = stitch_text(full_extracted_text, page_text)
    
    return {
        "extracted_text": full_extracted_text
    }


@app.get("/assignment")
def assignment_page():
    return FileResponse("templates/assignment.html")


@app.post("/create_assignment")
async def create_assignment(
    academic_year: str = Form(...),
    class_section: str = Form(...),
    subject: str = Form(...),
    assignment_title: str = Form(...),
    chapter: List[str] = Form(...),
    assignment_topic: List[str] = Form(...),
    textbook_reference: List[UploadFile] = File(...),
    questions: List[str] = Form(...),
    total_marks: int = Form(...),
    submission_date: str = Form(...),
    is_active: Optional[bool] = Form(False)
):
    combined_text = ""
    for ref_file in textbook_reference:
        # Save the textbook reference
        ref_file_path = f"./savepdf/ref_{ref_file.filename}"
        with open(ref_file_path, "wb") as f:
            f.write(await ref_file.read())

        # Extract text and append to combined context
        text = extraction(ref_file_path)
        combined_text += "\n" + text
    
    # Store in a separate collection in the vector database
    collection_name = f"{academic_year}_{class_section}_{subject}_{assignment_title}"
    
    try:
        vectordbadd(combined_text, collection_name)
    except Exception as e:
        print(f"ERROR: vectordbadd failed: {str(e)}")
        raise e
    
    # Save assignment metadata
    assignment_data = {
        "id": f"{academic_year}_{class_section}_{subject}_{assignment_title}".replace(" ", "_").replace("-", "_").lower()[:60],
        "academic_year": academic_year,
        "class_section": class_section,
        "subject": subject,
        "assignment_title": assignment_title,
        "chapter": chapter,
        "topic": assignment_topic,
        "collection_name": collection_name,
        "questions": questions,
        "total_marks": total_marks,
        "submission_date": submission_date,
        "is_active": is_active
    }
    save_assignment(assignment_data)
    
    return {
        "message": "Assignment created and textbook reference indexed.",
        "collection": collection_name,
        "details": assignment_data
    }


@app.get("/list_assignments")
def list_assignments():
    return get_assignments()

