from fastapi import FastAPI, Form, Request, Response, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_pipeline

app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates=Jinja2Templates(directory="templates")


# Root route
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle file upload and processing
@app.post("/upload")
async def upload_file(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    # Save the uploaded PDF file
    base_folder = "static/docs/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    pdf_file_name=os.path.join(base_folder, filename)

    # Save the uploaded file asynchronously, asynchronously means it will not block the main thread while writing the file
    async with aiofiles.open(pdf_file_name, 'wb') as out_file:
        await out_file.write(pdf_file)

    return {"msg": "File uploaded successfully", "pdf_filename": pdf_file_name}

# Route to generate Q&A and return CSV file
def get_csv(file_path, limit=5):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = os.path.join(base_folder, "QA.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        # Normalize and limit questions to the requested count
        limited_questions = [q.strip() for q in ques_list if q and q.strip()]
        limited_questions = limited_questions[:limit]

        for question in limited_questions:
            print("Question: ", question)
            # Use invoke (expects input key 'query') and normalize the returned shape
            try:
                result = answer_generation_chain.invoke({"query": question})
            except Exception as e:
                print("Error invoking answer chain:", e)
                answer_text = str(e)
            else:
                if isinstance(result, dict):
                    answer_text = result.get("result") or result.get("answer") or result.get("output") or result.get("text") or ""
                else:
                    answer_text = str(result)

            print("Answer: ", answer_text)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer_text])
    return output_file


@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...), limit: int = Form(5)):
    try:
        # Validate the file exists
        if not os.path.exists(pdf_filename):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file not found")

        csv_file_path = get_csv(pdf_filename, limit=limit)
        return {"msg": "Q&A generated successfully", "output_file": csv_file_path}
    except HTTPException:
        # Re-raise HTTP exceptions so FastAPI sends the proper status
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"An error occurred during analysis: {str(e)}")
    

if __name__=="__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)