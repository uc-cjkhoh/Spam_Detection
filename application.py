from fastapi import FastAPI, File, UploadFile, Request  # Add Request here
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tqdm import tqdm

import os
import sys
import joblib
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from loader.logger_loader import logging
from loader.config_loader import cfg
from src.llm import text_embedding 

# Create templates and static folders if they don't exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

model = joblib.load(f'models/{sys.argv[1]}')
app = FastAPI(title='Spam Detection Module')

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ModelData(BaseModel):
    message: str

@app.get("/")
async def home(request: Request):  # Add type hint
    return templates.TemplateResponse("index.html", {"request": request})
 
@app.post("/predict_file")
async def predict_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):  
    temp_path = None
    output_path = None
    try:
        # Create temp file to store upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Read Excel file
        df = pd.read_excel(temp_path)
        is_spam_list = []
        confidence_score_list = []
            
        # Process each message
        for message in tqdm(df[cfg.data.target_column]):
            try:
                message_embedding = text_embedding(pd.Series(message))
                prediction = model.predict(message_embedding)
                confidence_score = model.predict_proba(message_embedding)
                
                is_spam_list.append(prediction[0])
                confidence_score_list.append(float(confidence_score.max(axis=1)[0]))
                
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}", exc_info=True)
            
        # Add results to DataFrame
        df[f'{cfg.data.target_column}_label'] = np.array(is_spam_list)
        df[f'{cfg.data.target_column}_score'] = np.array(confidence_score_list)

        # Save results to new Excel file in a temp directory
        output_path = os.path.join(tempfile.gettempdir(), "spam_detection_results.xlsx")
        df.to_excel(output_path, index=False)
        
        # Add cleanup task to background tasks
        background_tasks.add_task(os.unlink, output_path)
        
        # Return file response
        return FileResponse(
            path=output_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="spam_detection_results.xlsx"
        )

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up only the input temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)