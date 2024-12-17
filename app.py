import streamlit as st
from utils import get_recommendations, data, similarity_matrix

def fetch_data(curr=0):
    for i in range(curr, curr+5):
        similar = get_recommendations(data, i, similarity_matrix)
        similar = similar.reset_index()
        tile = st.container()
        tile.title(f":blue[{data['job_title'][i]}]")
        description = data['description'][i]
        short_description = description[:1000] + "..." if len(description) > 1000 else description
        tile.write(short_description)
        if len(description) > 1000:
            with tile.popover("Read more"):
                st.write(description)
        # visit the job link button
        st.page_link(data['link'][i], label="job link", icon="🌎")
        st.write("experience level:", data['experience_level'][i])
        st.write("job type:", data['job_type'][i])
        if data['job_type'][i] == 'Hourly':
            st.write("hourly rate:", data['lower_range'][i], "-", data['higher_range'][i])
            st.write("Duration:", data['duration'][i])
        else:
            st.write("budget:", data['budget'][i])
        with tile.expander("view similar jobs"):
            st.write("similar jobs")
            tabs = st.tabs([similar['job_title'][j] for j in range(len(similar))])
            for idx, tab in enumerate(tabs):
                with tab:
                    description = similar['description'][idx]
                    short_description = description[:1000] + "..." if len(description) > 1000 else description
                    st.write(short_description)
                    if len(description) > 1000:
                        with st.popover("Read more"):
                            st.write(description)
                    # visit the job link button
                    st.page_link(similar['link'][i], label="job link", icon="🌎")
                    st.write("experience level:", similar['experience_level'][i])
                    st.write("job type:", similar['job_type'][i])
                    if similar['job_type'][i] == 'Hourly':
                        st.write("hourly rate:", similar['lower_range'][i], "-", similar['higher_range'][i])
                        st.write("Duration:", similar['duration'][i])
                    else:
                        st.write("budget:", similar['budget'][i])
        st.divider()
    
    if st.button("Load more", key=f"load_more_{curr}"):
        st.session_state['curr'] = curr + min(5, len(data) - curr)
        fetch_data(st.session_state['curr'])

def main():
    st.title("Job Recommendation System")
    if 'curr' not in st.session_state:
        st.session_state['curr'] = 0
    fetch_data(st.session_state['curr'])

if __name__ == "__main__":
    main()
