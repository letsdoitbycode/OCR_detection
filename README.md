# OCR to Tabular Data with Llama3.2 Vision Model

## Overview
This project leverages the powerful capabilities of the Llama3.2 Vision Model to extract textual data from images, specifically tailored for **Keeping Records of Employees**. Designed with approach to keep the task simple in mind, this tool converts Records into structured SQL format which enables seamless integration for keeping records of the employees.

## Features

- **Highly Accurate OCR**: Optimized for extracting text from Records, including handwritten notes.
- **Table Formatting**: Automatically formats extracted data into structured tables as per SQL format provided.
- **Interactive Interface**: Simple and intuitive Streamlit-based web application with a sidebar for image uploads.
- **Progress Updates**: Real-time progress updates during OCR processing.
- **Supports Overlapping Image Stripes**: Splits images into overlapping sections for improved accuracy on large records.


## How It Works
1. **Upload Invoice**: Users upload an invoice (JPEG or PNG format) via the sidebar.
2. **Image Preprocessing**: The tool splits the image into overlapping horizontal stripes for better accuracy.
3. **Text Extraction**: Leverages Llama3.2 Vision Model to process each stripe and extract text.
4. **Tabular Conversion**: Aggregates and formats the extracted text into a structured table.
5. **Downloadable Output**: Users can upload the data further into database for seamless integration.

   
## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- An active API key for the [Llama3.2 Vision Model] from Groq(https://console.groq.com/docs/overview)

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/letsdoitbycode/OCR_detection.git
   cd OCR_detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Key**:
   Create a `.env` file in the root directory and add your GROQ API key and SQL details:
   ```env
   GROQ_API_KEY=your_api_key_here
   DB_HOST = HOST_NAME
   DB_USER = DB_USER
   DB_PASSWORD = DB_PASSWORD
   DB_NAME = DB_NAME
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage
1. **Upload Record**:
   - Use the sidebar to upload an image of a record (JPEG or PNG).
   - The uploaded image will be displayed in the sidebar for review.

2. **OCR Processing**:
   - The main screen will show the processing progress with real-time updates for each section of the record.

3. **Database Uploading**:
   - Once processing is complete, the extracted data will be displayed as a table.
   - Use the "Insert Records to database" to upload the data in database.

4. **View or Search records**:
   - You can view all records that are uploaded to database using "View all Records" button.
   - You can also search records in the database using the 'Search by name or email' option where you have to type name or email of the particular employee and click on the "search record" button to view that particular record from database.  


## Testing
To test the application:
1. Use sample record images, including those with:
   - Printed text
   - Handwritten notes
   - Complex layouts (e.g., multi-column or detailed itemizations)
2. Verify that the output table matches the original record details.
3. Test the Database Uploading and View or Search records functionality.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.
