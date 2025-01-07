import streamlit as st
from PIL import Image
from Read import initialize_model, Read_Data
# 初始化模型和數據
@st.cache_resource
def initialize_data():
    return Read_Data()

def main():
    recipes = initialize_data()

    # 應用程式標題
    st.title("智慧家常食譜")
    #st.write("使用 CLIP 模型進行圖片與文字的食譜搜尋")

    # 使用模型選擇
    model, processor, tokenizer = initialize_model(fine_tuned=False)

    # 功能選擇
    st.sidebar.header("功能選擇")
    option = st.sidebar.selectbox(
        "請選擇功能",
        ["透過圖片搜尋食譜", "透過文字搜尋食譜", "透過圖片與文字搜尋食譜"]
    )

    # 功能 1：透過圖片搜尋食譜
    if option == "透過圖片搜尋食譜":
        st.header("透過圖片搜尋食譜")
        uploaded_image = st.file_uploader("請上傳食材圖片", type=["jpg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="上傳的圖片", use_container_width=True)
            
                    
    # 功能 2：透過文字搜尋食譜
    elif option == "透過文字搜尋食譜":
        st.header("透過文字搜尋食譜")  
        clip_or_langchain = st.radio("請選擇一個搜尋方式：", ('CLIP', 'Langchain(包含食材)'))
    
        if clip_or_langchain == 'CLIP':
            openai_api_key = None

        elif clip_or_langchain == 'Langchain(包含食材)':        
            openai_api_key = st.text_input('key:', type='password')

        search_query = st.text_input("請輸入您的描述，例如：清淡的蔬菜料理")
        


    # 功能 3：透過圖片與文字搜尋食譜
    elif option == "透過圖片與文字搜尋食譜":
        st.header("透過圖片與文字搜尋食譜")

        uploaded_image = st.file_uploader("請上傳食材圖片", type=["jpg", "png"])
        search_query = st.text_input("請輸入您的描述，例如：雞肉與青菜的搭配料理")
        
        if uploaded_image and search_query:
            image = Image.open(uploaded_image)
            st.image(image, caption="上傳的圖片", use_container_width=True)
        elif not uploaded_image:
            st.warning("請上傳圖片後再進行搜尋")
        elif not search_query:
            st.warning("請輸入描述後再進行搜尋")

if __name__ == '__main__':
    main()
