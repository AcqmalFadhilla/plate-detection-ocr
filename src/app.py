import streamlit as st

def main():
    image_page = st.Page("./src/pages/image.py", title="image")
    video_page = st.Page("./src/pages/video.py", title="video")

    pg = st.navigation([image_page, video_page])
    pg.run()

if __name__ == "__main__":
    main()



    


    


