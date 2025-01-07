import pandas as pd
import torch
from PIL import Image
from Read import Read_Model, Read_Data
from Search import Search_by_Image, Search_by_Word , recipes_embedding ,Search_by_ImageAndText


if __name__ == '__main__':

    # 模型與處理器
    model, processor, tokenizer  = Read_Model()

    recipes = Read_Data()
    
    #results  = recipes_embedding(recipes,model,processor,tokenizer)

    
    #以食材圖片找出食譜
    image_path = "images/豆腐.jpg"  
    image = Image.open(image_path)
    results = Search_by_Image(recipes , image , model, processor,top_k=5)

    #輸出結果
    for result in results:
        print(result)
        print("*"*50)



    #以文字敘述找出食譜
    search_query = "今天想吃一個清淡的食物，有青菜"
    # = "豬肉做成的料理"
    Search_by_Word(recipes,search_query, model, processor,tokenizer, top_k=3)



    #以文字敘述和照片找出食譜
    image_path = "images/雞肉.jpg" 
    image = Image.open(image_path)
    search_query = "今天想吃一個清淡的食物，有青菜"
    Search_by_ImageAndText(recipes, image,search_query , model,processor,tokenizer,top_k=5)
    #目前看起來受照片影響比較多