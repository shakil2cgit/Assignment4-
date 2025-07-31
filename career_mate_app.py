import streamlit as st
import asyncio
import uuid
from datetime import datetime

# Import the agent and models from your assignment file
from Assinment4_agent import (
    conversation_agent,
    CareerContext,
    SkillGapAnalysis,
    JobListing,
    CourseRecommendation,
)
from agents import Runner

# Page configuration
st.set_page_config(
    page_title="CareerMate Advisor",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #2196F3;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
        border-left: 5px solid #4CAF50;
    }
    .chat-message .content {
        display: flex;
        margin-top: 0.5rem;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .message {
        flex: 1;
        color: #000000;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_context" not in st.session_state:
    st.session_state.user_context = CareerContext(
        user_id=str(uuid.uuid4()),
        current_skills=[]
    )

if "processing_message" not in st.session_state:
    st.session_state.processing_message = None

# Function to format agent responses
def format_agent_response(output):
    if hasattr(output, "model_dump"):
        output = output.model_dump()

    if isinstance(output, dict) and "missing_skills" in output:
        analysis = SkillGapAnalysis.model_validate(output)
        html = f"<h4>Skill Gap Analysis for: {analysis.target_job}</h4>"
        html += f"<p><strong>Your Skills:</strong> {', '.join(analysis.user_skills)}</p>"
        html += f"<p><strong>Required Skills:</strong> {', '.join(analysis.required_skills)}</p>"
        if analysis.missing_skills:
            html += f"<p><strong>Missing Skills:</strong> <span style='color: red;'>{', '.join(analysis.missing_skills)}</span></p>"
        else:
            html += f"<p><strong>Missing Skills:</strong> None! You're all set.</p>"
        html += f"<p><em>{analysis.notes}</em></p>"
        return html

    elif isinstance(output, list) and output and "job_title" in output[0]:
        html = "<h4>Found some job opportunities for you:</h4><ul>"
        for job_data in output:
            job = JobListing.model_validate(job_data)
            html += f"<li><strong>{job.job_title}</strong> at {job.company} ({job.location})</li>"
            html += f"<ul><li>Skills: {', '.join(job.required_skills)}</li></ul>"
        html += "</ul>"
        return html

    elif isinstance(output, list) and output and "course_title" in output[0]:
        html = "<h4>Here are some courses to help you learn:</h4><ul>"
        for course_data in output:
            course = CourseRecommendation.model_validate(course_data)
            html += f"<li>To learn <strong>{course.skill_to_learn}</strong>: '{course.course_title}' on {course.platform}</li>"
        html += "</ul>"
        return html
    
    return str(output)

# Function to handle user input
def handle_user_message(user_input: str):
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": timestamp})
    st.session_state.processing_message = user_input

# Sidebar for user context
with st.sidebar:
    st.title("🧑‍💼 Your Profile")
    st.subheader("Your Current Skills")
    
    skills_text = st.text_area(
        "Enter your skills, one per line.",
        value="\n".join(st.session_state.user_context.current_skills),
        height=150
    )
    
    if st.button("Save Skills"):
        st.session_state.user_context.current_skills = [skill.strip() for skill in skills_text.split("\n") if skill.strip()]
        st.success("Skills saved!")
        st.rerun()

    st.divider()
    
    if st.button("Start New Conversation"):
        st.session_state.chat_history = []
        st.success("New conversation started!")
        st.rerun()

# Main chat interface
st.title("CareerMate Advisor")
st.caption("Your personal AI-powered career guidance assistant.")

# Display chat messages
for message in st.session_state.chat_history:
    role = message["role"]
    avatar_seed = st.session_state.user_context.user_id if role == "user" else "CareerMate"
    avatar_style = "avataaars" if role == "user" else "bottts"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <div class="content">
                <img src="https://api.dicebear.com/7.x/{avatar_style}/svg?seed={avatar_seed}" class="avatar" />
                <div class="message">
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# User input
user_input = st.chat_input("Ask for career advice...")
if user_input:
    handle_user_message(user_input)
    st.rerun()

# Process message if needed
if st.session_state.processing_message:
    user_input = st.session_state.processing_message
    st.session_state.processing_message = None
    
    with st.spinner("CareerMate is thinking..."):
        try:
            # Running asyncio.run() in a running loop (like Streamlit's) can cause errors.
            # It's safer to get the existing loop or create a new one.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(Runner.run(conversation_agent, user_input, context=st.session_state.user_context))
            response_content = format_agent_response(result.final_output)
            st.session_state.chat_history.append({"role": "assistant", "content": response_content, "timestamp": datetime.now().strftime("%I:%M %p")})
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message, "timestamp": datetime.now().strftime("%I:%M %p")})
        
        st.rerun()

# Footer
st.divider()
st.caption("Powered by Multi-Agent AI | Built with Streamlit")