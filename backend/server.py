from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import uvicorn
import argparse
from model import MyYOLO

arg_parser = argparse.ArgumentParser(description="Evaluate Mamba-YOLO on COCO dataset")
arg_parser.add_argument("--weights", type=str, default="output_dir/mscoco260606/mambayolo_n/weights/best.pt", help="Path to the trained weights")
arg_parser.add_argument("--model_yaml", type=str, default="ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml", help="Path to the model YAML configuration")
arg_parser.add_argument("--device", type=str, default='0', help="Device to use for evaluation (e.g., '0' for GPU or 'cpu' for CPU)")

app = FastAPI()

# 必须配置 CORS，否则前端 fetch 会报错
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL = None  # 全局变量，存储模型实例
args = None

@app.get("/")
async def index():
    return FileResponse("./backend/index.html")

@app.post("/start_model")
async def start_model():
    """
    启动模型，加载到内存中
    """
    global MODEL, args
    if MODEL is not None:
        return {"status": "模型已加载"}
    try:
        MODEL = MyYOLO(args)
        return {"status": "模型加载成功"}
    except Exception as e:
        MODEL = None
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

@app.post("/stop_model")
async def stop_model():
    """
    卸载模型，释放显存
    """
    global MODEL
    if MODEL is None:
        return {"status": "模型未加载"}
    try:
        # 删除模型引用
        del MODEL
        MODEL = None
        return {"status": "模型已释放"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型释放失败: {str(e)}")

@app.get("/model_status")
async def model_status():
    global MODEL
    return {
        "loaded": MODEL is not None
    }

@app.post("/process")
async def process_image(image: UploadFile = File(...)):  # 修改：参数名改为 'image' 匹配前端
    """
    接收前端上传的图片，进行AI处理（示例：风格化滤镜），返回处理后的图片
    """
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=400, detail="模型未加载，请先启动模型")
    try:
        # 1. 验证文件类型
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传有效的图片文件")

        # 2. 读取上传的文件内容
        contents = await image.read()

        # 限制文件大小（例如 10MB）
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片大小不能超过10MB")

        input_image = Image.open(BytesIO(contents)).convert("RGB")

        img = MODEL.predict(input_image)

        # 转回 PIL 保存到内存
        output_image = Image.fromarray(img)
        img_byte_arr = BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")


    except HTTPException:
        raise
    except Exception as e:
        print(f"处理错误: {str(e)}")  # 服务端日志记录
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")

@app.post("/process_test")
async def process_test(image: UploadFile = File(...)):  # 修改：参数名改为 'image' 匹配前端
    """
    接收前端上传的图片，进行AI处理（示例：风格化滤镜），返回处理后的图片
    """
    try:
        # 1. 验证文件类型
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传有效的图片文件")

        # 2. 读取上传的文件内容
        contents = await image.read()

        # 限制文件大小（例如 10MB）
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片大小不能超过10MB")

        # 3. 使用 PIL 打开图片
        input_image = Image.open(BytesIO(contents)).convert("RGB")

        # 4. 在这里运行你的 Torch 模型
        # ========== AI 处理示例 ==========
        # 示例1：增强对比度 + 边缘检测风格（模拟AI艺术效果）
        enhancer = ImageEnhance.Contrast(input_image)
        enhanced = enhancer.enhance(1.5)  # 增强对比度

        # 添加轻微的锐化效果
        sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # 可选：添加暖色调滤镜 (模拟AI调色)
        r, g, b = sharpened.split()
        # 增强红色通道，略微降低蓝色，营造暖色氛围
        r = r.point(lambda i: i * 1.1)
        b = b.point(lambda i: i * 0.9)
        output_image = Image.merge('RGB', (r, g, b))

        # 如果你有真正的 PyTorch 模型，可以这样加载：
        # model = load_model()  # 在启动时加载一次
        # output_tensor = model(preprocess(input_image))
        # output_image = postprocess(output_tensor)
        # =================================

        # 5. 将处理后的图片保存到内存（保持高质量）
        img_byte_arr = BytesIO()
        # 保存为 PNG 格式（无损，确保质量）
        output_image.save(img_byte_arr, format='PNG', optimize=True)
        img_byte_arr = img_byte_arr.getvalue()

        # 6. 返回二进制图片流
        return Response(
            content=img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=processed.png"}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"处理错误: {str(e)}")  # 服务端日志记录
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)