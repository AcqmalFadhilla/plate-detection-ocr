import streamlit as st

def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    image_page = st.Page("pages/image.py", title="image")
    video_page = st.Page("pages/video.py", title="video")

    pg = st.navigation([image_page, video_page])
    pg.run()

if __name__ == "__main__":
    main()



    


    


