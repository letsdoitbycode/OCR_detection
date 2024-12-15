import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain_groq import ChatGroq
import base64
import io
from time import sleep
import mysql.connector
from mysql.connector import Error
import pandas as pd


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Function to encode an image to base64
def encode_image_pil(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to split an image into horizontal stripes
def split_image_into_horizontal_stripes(image: Image.Image, stripe_count: int = 5, overlap: float = 0.1):
    width, height = image.size
    stripe_height = height // stripe_count
    overlap_height = int(stripe_height * overlap)

    stripes = []
    for i in range(stripe_count):
        upper = max(i * stripe_height - overlap_height, 0)
        lower = min((i + 1) * stripe_height + overlap_height, height)
        stripe = image.crop((0, upper, width, lower))
        stripes.append(stripe)
    return stripes

# Function for OCR using Groq
def ocr(image: Image.Image, model: str = "llama-3.2-90b-vision-preview") -> str:
    groq_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model,
        temperature=0
    )

    image_data_url = f"data:image/jpeg;base64,{encode_image_pil(image)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "The uploaded image contains both printed text and handwritten notes. "
                    "Your task is to carefully extract all textual content, including handwritten elements."
                )},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        }
    ]

    response = groq_llm.invoke(messages)
    return response.content.strip()

# Function to consolidate markdown into tabular format
def format_to_table(markdown_runs: list, model: str = "llama-3.3-70b-versatile") -> str:
    groq_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model,
        temperature=0
    )

    combined_markdown = "\n\n".join(markdown_runs)

    messages = [
        {
            "role": "user",
            "content": (
                "You are provided with multiple markdown outputs extracted from overlapping sections of an image."
                "Some sections may contain duplicate or conflicting information due to overlaps. "
                "Your task is to:"
                "\n\n1. Identify and consolidate rows of data that are related, ensuring that the most complete version of the information is retained."
                "\n2. For rows with conflicting information (e.g., different values for a field), prioritize the more detailed entry."
                "\n3. If a field is missing in one row but present in another, combine the information into a single row."
                "\n4. Output the consolidated data in a clean tabular format using Markdown syntax, suitable for direct rendering."
                "\n5. Output Only Markdown: Return solely the Markdown content without any additional explanations or comments."
                "\n\nHere is the data to process:\n\n"
                + combined_markdown
            )
        }
    ]

    response = groq_llm.invoke(messages)
    return response.content.strip()

# Function to parse the consolidated markdown into a dictionary for database insertion
def parse_markdown_to_dict(markdown_table: str) -> list:
    rows = markdown_table.split("\n")[2:]  # Skip the header row
    records = []
    for row in rows:
        fields = [field.strip() for field in row.split("|")[1:-1]]
        if len(fields) >= 9:
            record = {
                "name": fields[0],
                "email": fields[1],
                "address": fields[2],
                "dob": fields[3],
                "age": int(fields[4]) if fields[4].isdigit() else None,
                "gender": fields[5],
                "mobile": fields[6],
                "education": fields[7],
                "profile": fields[8],
            }
            records.append(record)
    return records

# Function to insert records into MySQL database
def insert_records_to_db(records: list):
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = connection.cursor()

        query = """
        INSERT INTO Record (name, email, address, dob, age, gender, mobile, education, profile)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for record in records:
            cursor.execute(query, (
                record["name"], record["email"], record["address"], record["dob"],
                record["age"], record["gender"], record["mobile"],
                record["education"], record["profile"]
            ))
        connection.commit()
        st.success("Records have been successfully inserted into the database!")
    except Error as e:
        st.error(f"Error connecting to the database: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to fetch all records from the database
def fetch_all_records():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Record")
        records = cursor.fetchall()
        return records
    except Error as e:
        st.error(f"Error fetching data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to search records by name or email
def search_records(search_term):
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT * FROM Record
        WHERE name LIKE %s OR email LIKE %s
        """
        cursor.execute(query, (f"%{search_term}%", f"%{search_term}%"))
        records = cursor.fetchall()
        return records
    except Error as e:
        st.error(f"Error searching data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# Streamlit Application
st.title("OCR to Tabular Data with MySQL Integration")
st.markdown("Convert uploaded image content into a structured table format and store it in a database.")

# Sidebar for Upload and Display
with st.sidebar:
    st.markdown("#### Upload Image")
    uploaded_file = st.file_uploader("Upload an image (JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Resize image for better display
        width, height = image.size
        new_width, new_height = int(width * 1.2), int(height * 1.2)
        image = image.resize((new_width, new_height))
        
        # Display the image in the sidebar
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image
        stripes = split_image_into_horizontal_stripes(image)

# Main Section for Processing and Results
if uploaded_file is not None:
    st.markdown("#### OCR and Results")
    progress_bar = st.progress(0)
    n = 1
    markdown_runs = []
    total_steps = len(stripes) * n
    step = 0

    # Dynamic status box
    status_box = st.empty()

    for run in range(1, n + 1):
        for i, stripe in enumerate(stripes, start=1):
            step += 1
            progress = step / total_steps
            progress_bar.progress(progress)
            status_box.markdown(f"**Processing Stripe {i}, Run {run} ({int(progress * 100)}%)...**")
            sleep(0.1)  # Simulating processing time

            stripe_markdown = ocr(stripe, model="llama-3.2-90b-vision-preview")
            markdown_runs.append(stripe_markdown)

    progress_bar.progress(1.0)
    status_box.markdown("**Processing complete.**")

    # Consolidate results into a table
    table_output = format_to_table(markdown_runs, model="llama-3.3-70b-versatile")
    st.markdown("### Consolidated Table")
    st.markdown(table_output, unsafe_allow_html=True)

    # Parse table to dictionary
    records = parse_markdown_to_dict(table_output)

    # Insert records into the database
    st.markdown("#### Save Results to Database")
    if st.button("Insert Records into Database"):
        insert_records_to_db(records)
else:
    st.markdown("Output will be displayed here after uploading an image.")

# View Records Section
st.markdown("### View and Search Records")

if st.button("View All Records"):
    all_records = fetch_all_records()
    if all_records:
        df = pd.DataFrame(all_records)  # Convert list of dicts to DataFrame
        st.dataframe(df)  # Display as an interactive table
    else:
        st.write("No records found.")

# Search Records Section
search_term = st.text_input("Search by Name or Email")
if st.button("Search Records"):
    search_results = search_records(search_term)
    if search_results:
        df = pd.DataFrame(search_results)  # Convert search results to DataFrame
        st.dataframe(df)  # Display as an interactive table
    else:
        st.write("No matching records found.")
