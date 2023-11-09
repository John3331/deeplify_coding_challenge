import io
import base64
from typing import Annotated
from PIL import Image, ImageDraw, ImageFont
import torch

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse

from src.train import create_efficientnet, create_transforms


MODEL_PATH = r'models\effnet,lr=0.00245,EP=25,BS=64,Nov09_00-18-40\epoch24.pth'
INV_CLASS_MAP = {
    0: 'NORMAL',
    1: 'PNEUMONIA'
}


app = FastAPI()
templates = Jinja2Templates(directory="src/templates")


model = create_efficientnet() # for some reason this is needed
model = torch.load(MODEL_PATH)
model.eval()

_, val_test_transform = create_transforms() 


def add_label_to_image(pil_image, label):
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    image_width, image_height = pil_image.size

    # Define the font color
    font_color = 255  # White color
    font_size=image_height//9

    # Define the font (you can specify a font file)
    font = ImageFont.truetype(r"src/open-sans\OpenSans-Regular.ttf", font_size)

    # Calculate the position to center the label
    text_width, text_height = draw.textsize(label, font)
    
    x = (image_width - text_width) / 2
    y = image_height - text_height - 10  

    # Add the label to the image
    draw.text((x, y), label, font=font, fill=font_color)

    return pil_image


@app.post("/files/")
async def classify_image(files: Annotated[list[bytes], File()], request:Request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # preprocess the image
    pil_image = Image.open(io.BytesIO(files[0]))
    image = val_test_transform(pil_image)
    image = image[None, :, :, :].to(device)

    # evaluate the image
    output = torch.sigmoid(model(image)).reshape(-1).cpu().item()
    bin_output = round(output)
    confidence = round(((bin_output - 0.5) * 2) * (output - 0.5) * 2, 4)

    # add the label to the image and encode it
    labeled_image = add_label_to_image(pil_image, INV_CLASS_MAP[bin_output])
    buffered = io.BytesIO()
    labeled_image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    result = {
        "Classification": INV_CLASS_MAP[bin_output],
        "confidence": confidence,
        "image": image_base64
    }

    return  JSONResponse(content=result) 
    

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("template.html", {"request": request})
    