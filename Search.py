import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import os
from tqdm import tqdm
import math
import re
import numpy as np
from langchain_community.vectorstores import Chroma
#from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import warnings

warnings.filterwarnings('ignore')
#__import__('pysqlite3') 
import sys 
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def recipes_embedding(recipes,model,processor,tokenizer):


     #取得品名列表
    candidate_list = list(recipes.keys())

    results = []
    for name in candidate_list:
        image_file = f"{name}.jpg"
        image_path = os.path.join("./images/", image_file)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_preproc = processor(images=image, return_tensors="pt").to("cpu")
        else:
            print(f"警告: 找不到 {name}.jpg 文件")
    
        with torch.no_grad():
            image_embeddings = model.get_image_features(image_preproc.pixel_values)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings.cpu().numpy()

        with torch.no_grad():
            tokenized_text = tokenizer(name, return_tensors="pt", padding=True)
            text_embedding = model.get_text_features(tokenized_text["input_ids"].to("cpu"))
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            text_embeddings = text_embedding.cpu().numpy()

        


        prediction = {
            "品名": name,
            "品名embedding": text_embeddings,
            "食材": recipes[name]["食材"],
            "步驟": recipes[name]["步驟"],
            "圖片embedding" : image_embeddings
            }
        # 添加到結果列表
    results.append(prediction)


    return results



#透過一段話來找到對應的食譜
def Search_by_Word(recipes, clip_or_langchain, openai_api_key, text_query, model, processor, tokenizer, top_k=5):

    candidate_list = list(recipes.keys())

    if clip_or_langchain == 'CLIP':
        with torch.no_grad():
            tokenized_text = tokenizer(text_query, return_tensors="pt", padding=True)
            text_embedding = model.get_text_features(tokenized_text["input_ids"].to("cpu"))
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            text_embeddings = text_embedding.cpu().numpy()

        # 取得品名列表       

        images_files = []
        for name in candidate_list:
            image_file = f"{name}.jpg"
            image_path = os.path.join("./images/", image_file)
            if os.path.exists(image_path):
                images_files.append(image_path)
            else:
                print(f"警告: 找不到 {name}.jpg 文件")
        
        image_embeddings = get_image_embedding(model, processor, images_files)

        similarities = (image_embeddings @ text_embeddings.T).squeeze(1)
        best_match_image_idx = (-similarities).argsort()

        results = []

        for idx in best_match_image_idx[:top_k]:
            image_path = images_files[idx]
            name = candidate_list[idx]
            similarity = similarities[idx]
            recipe = recipes[name]
            results.append({
                "品名": name,
                "相似度": similarity,
                "圖片路徑": image_path,
                "步驟": recipe["步驟"]
            })

    elif clip_or_langchain == 'Langchain(包含食材)':
        
        results = []
        os.environ['OPENAI_API_KEY'] = openai_api_key

        embeddings = OpenAIEmbeddings()
        db_name = 'chroma_db'
                
        db = Chroma(persist_directory=db_name, embedding_function=embeddings)

        scoreRes = db.similarity_search_with_score(text_query, k=5)

        for idx in scoreRes:
            name = idx[0].metadata['source'].split('/')[-1][:-5]
            image_path = os.path.join("./images/", f"{name}.jpg")
            similarity = idx[-1]
            recipe = idx[0].page_content.split('步驟: ')[-1]
            
            results.append({
                "品名": name,
                "相似度": similarity,
                "圖片路徑": image_path,
                "步驟": recipe
            })

    return results




#透過食材圖片來找到食譜
def Search_by_Image(recipes, image, model, processor, top_k=5):
    """
    透過圖片搜尋食譜，並返回包含品名、機率、步驟與圖片路徑的結果。
    """
    # 取得品名列表
    candidate_list = list(recipes.keys())

    # 跑模型
    inputs = processor(text=candidate_list, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # 計算相似度
    logits_per_image = outputs.logits_per_image  # 圖片與文字的相似度分數
    probs = logits_per_image.softmax(dim=1)  # 將分數轉為機率

    # 找出前 k 名
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)

    results = []
    for i in range(top_k):
        label = candidate_list[top_indices[0, i]]  # 獲取對應的品名
        probability = top_probs[0, i].item()  # 獲取對應的機率
        image_path = f"./images/{label}.jpg"  # 假設圖片名稱與品名一致
        recipe = recipes[label]
        results.append({
            "品名": label,
            "相似度": round(probability, 4),
            "圖片路徑": image_path if os.path.exists(image_path) else None,
            "食材": recipe["食材"],
            "步驟": recipe["步驟"]
        })

    return results



def get_image_embedding(model, processor, images_files):
   
    batch_size = 32
    image_embeddings = []

 
    for i in tqdm(range(math.ceil(len(images_files) / batch_size)), desc="Processing Images"):
        batch_files = images_files[batch_size * i : batch_size * (i + 1)]


       # 加载图片
        batch_images = []
        for path in batch_files:
            if path is not None:
                try:
                    batch_images.append(Image.open(path))
                except Exception as e:
                    print(f"警告: 無法載入 {path}, 錯誤: {e}")

        if not batch_images:
            continue  # 

        # 使用 processor 
        image_preproc = processor(images=batch_images, return_tensors="pt").to("cpu")

        with torch.no_grad():
       
            batch_embeddings = model.get_image_features(image_preproc.pixel_values)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            batch_embeddings = batch_embeddings.cpu().numpy()

        image_embeddings.append(batch_embeddings)

    return np.vstack(image_embeddings)


#透過食材圖片來找到食譜
def Search_by_ImageAndText(recipes, image, text, model, processor, tokenizer, top_k=5):
    candidate_list = list(recipes.keys())

    # 將候選文字與描述合併進行 Tokenize
    text_inputs = [text] + candidate_list

    with torch.no_grad():
        tokenized_text = tokenizer(text_inputs, return_tensors="pt", padding=True)
        text_embedding = model.get_text_features(tokenized_text["input_ids"].to("cpu"))
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embeddings = text_embedding.cpu().numpy()

        # 使用 processor 進行圖片預處理
        image_preproc = processor(images=image, return_tensors="pt").to("cpu")
        image_embeddings = model.get_image_features(image_preproc.pixel_values)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings.cpu().numpy()

    # 計算相似度
    text_similarities = (text_embeddings @ text_embeddings[0].T).flatten()  # 與主描述文字相似度
    image_similarities = (image_embeddings @ text_embeddings.T).flatten()  # 與圖片相似度

    # 平衡相似度權重
    combined_similarities = 0.8 * text_similarities + 0.2 * image_similarities
    candidate_similarities = combined_similarities[1:]  # 排除描述自身的相似度

    # 找出最佳匹配的前 K 名
    top_indices = (-candidate_similarities).argsort()[:top_k]
    results = []
    for idx in top_indices:
        name = candidate_list[idx]
        similarity = candidate_similarities[idx]
        recipe = recipes[name]
        image_path = f"./images/{name}.jpg"  # 假設圖片名稱與食譜名稱一致
        results.append({
            "品名": name,
            "相似度": round(similarity, 4),
            "食材": recipe["食材"],
            "步驟": recipe["步驟"],
            "圖片路徑": image_path if os.path.exists(image_path) else None
        })
    return results

