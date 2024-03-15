import gradio as gr
import os
import time
import torch
import clip
from PIL import Image

from pymilvus import (
    connections,
    Collection
)

root_path = "val2017"

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
model, preprocess = clip.load("ViT-B/32")
client = connections.connect(host="172.18.68.230", port="19530")
collection = Collection("HOIR")
collection.load()


def search_milvus(collection, query_features, top_k):
    start = time.time()
    results = collection.search(query_features, "embeddings", param={"nprobe": 16}, limit=top_k,
                                output_fields=["id", "path"])
    print(search_latency_fmt.format(time.time() - start))
    return results


# 输入文本处理程序
def retrieval(image):
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    query_features = image_features.numpy().tolist()
    results = search_milvus(collection, query_features, 5)
    path = results[0][0].entity.get('path')
    print(path)
    return Image.open(os.path.join(root_path, path))


demo = gr.Interface(fn=retrieval, inputs="image", outputs="image")
demo.launch(share=True)
