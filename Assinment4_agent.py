import os
import json
import asyncio
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled, RunContextWrapper

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME in your .env file.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

# --- Dummy Data ---
JOB_SKILLS: Dict[str, List[str]] = {
    "data scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas", "Scikit-learn"],
    "software engineer": ["Python", "Java", "Data Structures", "Algorithms", "Git", "Docker"],
    "product manager": ["Product Strategy", "User Research", "Agile Methodologies", "Roadmapping"],
    "ux designer": ["Figma", "User Research", "Wireframing", "Prototyping", "Interaction Design"],
}

JOB_LISTINGS: List[Dict] = [
    {"title": "Senior Data Scientist", "company": "Innovate AI", "location": "Remote", "skills": ["Python", "Machine Learning", "TensorFlow"]},
    {"title": "Backend Software Engineer", "company": "Tech Solutions Inc.", "location": "New York, NY", "skills": ["Python", "Java", "Docker", "Kubernetes"]},
    {"title": "Junior Data Scientist", "company": "Data Insights Co.", "location": "San Francisco, CA", "skills": ["Python", "SQL", "Pandas"]},
    {"title": "Product Manager, Growth", "company": "ConnectApp", "location": "Remote", "skills": ["Product Strategy", "Agile Methodologies", "A/B Testing"]},
]

COURSE_CATALOG: Dict[str, List[Dict]] = {
    "sql": [{"title": "Complete SQL Bootcamp", "platform": "Udemy", "link": "udemy.com/course/sql-bootcamp"}],
    "statistics": [{"title": "Statistics with Python", "platform": "Coursera", "link": "coursera.org/specializations/statistics-with-python"}],
    "pandas": [{"title": "Data Analysis with Pandas", "platform": "DataCamp", "link": "datacamp.com/courses/data-manipulation-with-pandas"}],
    "python": [{"title": "Python for Everybody", "platform": "Coursera", "link": "coursera.org/specializations/python"}],
}

# --- Pydantic Models for Structured Outputs ---
class SkillGapAnalysis(BaseModel):
    target_job: str = Field(description="The job title being analyzed.")
    user_skills: List[str] = Field(description="The skills the user already possesses.")
    required_skills: List[str] = Field(description="All skills required for the target job.")
    missing_skills: List[str] = Field(description="The skills the user needs to acquire for the job.")
    notes: str = Field(description="A brief summary or encouragement for the user.")

class JobListing(BaseModel):
    job_title: str
    company: str
    location: str
    required_skills: List[str]
    link_to_apply: str = Field(description="A dummy link to the job application page.")

class CourseRecommendation(BaseModel):
    skill_to_learn: str
    course_title: str
    platform: str
    link: str

# --- User Context ---
@dataclass
class CareerContext:
    user_id: str
    current_skills: List[str] = field(default_factory=list)

# --- Tools ---
@function_tool
async def get_missing_skills(wrapper: RunContextWrapper[CareerContext], target_job: str) -> str:
    """Identifies the skills a user is missing for a target job role."""
    target_job_lower = target_job.lower()
    user_skills = wrapper.context.current_skills if wrapper and wrapper.context else []
    
    required_skills = JOB_SKILLS.get(target_job_lower)
    if not required_skills:
        return json.dumps({"error": f"Sorry, I don't have skill information for '{target_job}'. Please try another role."})

    missing_skills = [skill for skill in required_skills if skill not in user_skills]
    
    analysis = SkillGapAnalysis(
        target_job=target_job,
        user_skills=user_skills,
        required_skills=required_skills,
        missing_skills=missing_skills,
        notes=f"You have a good foundation! Focusing on these {len(missing_skills)} skills will be a great next step." if missing_skills else "You have all the required skills for this role!"
    )
    return analysis.model_dump_json()

@function_tool
async def find_jobs(wrapper: RunContextWrapper[CareerContext], location: Optional[str] = None) -> str:
    """Finds job openings based on the user's skills and optional location."""
    user_skills = wrapper.context.current_skills if wrapper and wrapper.context else []
    if not user_skills:
        return json.dumps({"error": "I need to know your skills to find jobs. Please tell me what you're good at first."})

    matching_jobs = []
    for job in JOB_LISTINGS:
        if location and location.lower() not in job["location"].lower():
            continue
        
        if any(skill in job["skills"] for skill in user_skills):
            matching_jobs.append(JobListing(
                job_title=job["title"],
                company=job["company"],
                location=job["location"],
                required_skills=job["skills"],
                link_to_apply=f"https://example.com/jobs/{job['title'].replace(' ', '-').lower()}"
            ).model_dump())
            
    return json.dumps(matching_jobs)

@function_tool
async def recommend_courses(missing_skills: List[str]) -> str:
    """Recommends online courses for a list of skills."""
    recommendations = []
    for skill in missing_skills:
        courses = COURSE_CATALOG.get(skill.lower())
        if courses:
            for course in courses:
                recommendations.append(CourseRecommendation(
                    skill_to_learn=skill,
                    course_title=course["title"],
                    platform=course["platform"],
                    link=course["link"]
                ).model_dump())

    # The agent's output_type is List[CourseRecommendation].
    # Returning a message dict here would cause a validation error if no courses are found.
    # Instead, return an empty list and let the LLM generate a friendly message.
    return json.dumps(recommendations)

# --- Specialist Agents ---
skill_gap_agent = Agent[CareerContext](
    name="Skill Gap Agent",
    handoff_description="Helps users identify the skills required for a job and what they are missing.",
    instructions="You are a Skill Gap Analyzer. Your goal is to help the user understand the skills needed for a specific job role by comparing it to their current skills. Use the get_missing_skills tool.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_missing_skills],
    output_type=SkillGapAnalysis
)

job_finder_agent = Agent[CareerContext](
    name="Job Finder Agent",
    handoff_description="Searches for and suggests job openings based on the user's skills.",
    instructions="You are a Job Finder. Use the user's skills from the context to find relevant job openings using the find_jobs tool. Present the findings clearly.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[find_jobs],
    output_type=List[JobListing]
)

course_recommender_agent = Agent[CareerContext](
    name="Course Recommender Agent",
    handoff_description="Recommends online courses to help users learn new skills.",
    instructions="You are a Course Recommender. Given a list of skills a user wants to learn, use the recommend_courses tool to find and suggest relevant online courses.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[recommend_courses],
    output_type=List[CourseRecommendation]
)

# --- Main Controller Agent ---
conversation_agent = Agent[CareerContext](
    name="CareerMate Conversation Agent",
    instructions="You are CareerMate, a friendly and helpful career advisor. Your job is to greet the user, understand their career-related needs, and route their request to the correct specialist agent. The user's current skills are stored in the context. If you don't know their skills, you can ask.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[skill_gap_agent, job_finder_agent, course_recommender_agent]
)

# --- Main Execution Logic ---
async def main():
    print("--- CareerMate Initialized ---")
    
    career_context = CareerContext(
        user_id="test_user_01",
        current_skills=["Python", "Git", "Data Structures"]
    )
    print(f"User Context: Skills - {career_context.current_skills}\n")

    queries = [
        "I want to become a data scientist. What skills am I missing?",
        "Can you help me find a job with my current skills?",
        "How can I learn SQL and Pandas?",
        "Hi there! What can you do?"
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"USER QUERY: {query}")
        print("="*50)

        result = await Runner.run(conversation_agent, query, context=career_context)
        
        print(f"\nHANDLED BY: {result._last_agent.name}\n")
        print("FINAL RESPONSE:")

        if isinstance(result.final_output, SkillGapAnalysis):
            analysis = result.final_output
            print(f"Skill Gap Analysis for: {analysis.target_job}")
            print(f"  Your Skills: {', '.join(analysis.user_skills)}")
            print(f"  Required Skills: {', '.join(analysis.required_skills)}")
            print(f"  >> Missing Skills: {', '.join(analysis.missing_skills) if analysis.missing_skills else 'None!'}")
            print(f"  Note: {analysis.notes}")
        elif isinstance(result.final_output, list) and result.final_output and isinstance(result.final_output[0], JobListing):
            print("Found some job opportunities for you:")
            for job in result.final_output:
                print(f"  - {job.job_title} at {job.company} ({job.location})")
                print(f"    Skills: {', '.join(job.required_skills)}")
        elif isinstance(result.final_output, list) and result.final_output and isinstance(result.final_output[0], CourseRecommendation):
            print("Here are some courses to help you learn:")
            for course in result.final_output:
                print(f"  - To learn {course.skill_to_learn}: '{course.course_title}' on {course.platform}")
        else:
            print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
