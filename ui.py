import streamlit as st
import requests
import time

st.title("Blog Generator")

# Input for the keyword
keyword = st.text_input("Enter keyword for blog generation")

# Initialize session state variables
if 'job_id' not in st.session_state:
    st.session_state['job_id'] = None

if 'status' not in st.session_state:
    st.session_state['status'] = ''

if 'blog_post' not in st.session_state:
    st.session_state['blog_post'] = ''

# Function to start blog generation
def start_blog_generation():
    if not keyword:
        st.warning("Please enter a keyword")
        return
    st.session_state['status'] = 'Starting blog generation...'
    st.session_state['job_id'] = None  # Reset job_id
    st.session_state['blog_post'] = ''  # Reset blog content

    # Make POST request to generate-blog endpoint
    try:
        response = requests.post('http://localhost:8000/generate-blog', json={'keyword': keyword})
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'processing':
                st.session_state['status'] = data.get('message', 'Processing')
                st.session_state['job_id'] = data.get('job_id')
            else:
                st.session_state['status'] = 'Error: ' + data.get('error', 'Unknown error')
        else:
            st.session_state['status'] = f'Error: {response.status_code} {response.reason}'
    except Exception as e:
        st.session_state['status'] = 'Error: ' + str(e)

# Button to initiate blog generation
if st.button("Generate Blog"):
    start_blog_generation()

# Display status messages
status_placeholder = st.empty()
if st.session_state.get('status'):
    status_placeholder.info(st.session_state['status'])

# Function to check the blog generation status
def check_blog_status():
    try:
        job_id = st.session_state['job_id']
        response = requests.get(f'http://localhost:8000/blog-status/{job_id}')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'completed':
                st.session_state['status'] = 'Blog generated!'
                st.session_state['blog_post'] = data.get('blog_post', '')
                status_placeholder.success(st.session_state['status'])
                # Removed st.write(st.session_state['blog_post'])
                st.session_state['job_id'] = None  # Reset job_id
            elif data.get('status') == 'failed':
                st.session_state['status'] = 'Error: ' + data.get('error', 'Unknown error')
                status_placeholder.error(st.session_state['status'])
                st.session_state['job_id'] = None  # Reset job_id
            else:
                progress = data.get('progress', 'Processing')
                st.session_state['status'] = f'In Progress: {progress}'
                status_placeholder.info(st.session_state['status'])
                time.sleep(5)
                check_blog_status()  # Recursive call to check the status again
        else:
            st.session_state['status'] = f'Error: {response.status_code} {response.reason}'
            status_placeholder.error(st.session_state['status'])
            st.session_state['job_id'] = None
    except Exception as e:
        st.session_state['status'] = 'Error: ' + str(e)
        status_placeholder.error(st.session_state['status'])
        st.session_state['job_id'] = None

# If a job is in progress, check its status
if st.session_state.get('job_id'):
    check_blog_status()

# Display the generated blog post
if st.session_state.get('blog_post'):
    st.markdown(st.session_state['blog_post'])
