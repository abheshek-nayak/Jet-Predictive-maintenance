from fastapi import FastAPI, File, UploadFile, Request,HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import FileResponse
from io import BytesIO
import pandas as pd
import numpy as np
import uvicorn
import keras
from packages.clean import BASIC_CLEAN
from packages.train import SETUP,FINAL_TOUCHES
from packages import constants


scaler = MinMaxScaler()
app = FastAPI(title='Remaining Uselful Life of Aero Turbine Engines')
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
model = keras.models.load_model('predictive_model.hdf5')

@app.get("/",response_class=HTMLResponse)
def func(request: Request):
       
    return templates.TemplateResponse('home.html', {'request': request})

@app.post("/fd1")
def upload(request: Request,file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        buffer = BytesIO(contents)
        test1 = pd.read_csv(buffer,sep=" ",header=None)
    except:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        buffer.close()
        file.file.close()

    test_file = BASIC_CLEAN(test1)
    feats = test_file.columns.difference(['unit', 'time_cycle','RUL'])
    scaler = MinMaxScaler(feature_range=(-1,1))
    test_file[feats] = scaler.fit_transform(test_file[feats])
    x_test=np.concatenate(list(list(SETUP(test_file[test_file['unit']==unit],constants.SEQUENCE_LENGHT, feats,constants.MASKED_VALUE)) for unit in test_file['unit'].unique()))

    predictions = model.predict(x_test)

    df = FINAL_TOUCHES(predictions)


    return templates.TemplateResponse('home.html', context={'request': request, 'data': df.to_html()})




