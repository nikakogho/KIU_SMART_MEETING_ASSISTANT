# Smart Meeting Assistant

## 📜 Project Description

This project, "Smart Meeting Assistant," is the final assignment for Building AI Powered Apps.

KIU Consulting is currently losing an estimated 25,000 GEL per employee each year due to inefficient meetings. This AI-powered system aims to solve this problem by automatically processing meeting recordings to extract valuable, actionable insights and build a searchable knowledge base for the entire organization.

The core of this project is a web application that leverages a suite of OpenAI APIs to offer a seamless solution for meeting analysis and knowledge management.

## ✨ Core Features

This application integrates four key OpenAI APIs to deliver its primary functionalities:

### 🔊 1. Audio Processing with Whisper API

* **Transcription**: Transcribes audio recordings of meetings from formats like `.mp3`, `.wav`, and `.m4a`.
* **Speaker Identification**: Capable of handling audio files ranging from 20 to 30 minutes and distinguishing between different speakers in the conversation.

### 🧠 2. Content Analysis with GPT-4 and Function Calling

* **Meeting Summaries**: Generates concise summaries that highlight the key discussion points and outcomes.
* **Action Item Extraction**: Identifies and extracts actionable tasks, and assigns ownership of those tasks to the responsible individuals.
* **Calendar and Task Integration (Bonus)**: Connects with calendar and task management APIs to automatically create events and to-do items from the meeting's content.

### 🔍 3. Semantic Search with Embeddings API

* **Knowledge Base Creation**: Develops a searchable repository of information from all processed meetings.
* **Similarity-Based Recommendations**: Suggests relevant past meetings or documents based on the content of the current meeting.

### 🎨 4. Visual Concept Synthesis with DALL-E 3

* **Visual Summaries**: Creates engaging visual representations of meeting outcomes for stakeholders who were not in attendance.
* **Automated Presentation Assets**: Generates images and diagrams that can be used in follow-up presentations and reports.

---

## 🚀 Advanced Features

* **Real-Time Processing**: Live meeting transcription and analysis using WebSocket technology.
* **Predictive Analytics**: Predicts the effectiveness of a meeting based on various data points.

---

## 🛠️ Technical Requirements

The successful implementation of this project relies on the seamless integration of the following OpenAI APIs:

* **Whisper API** for audio transcription.
* **GPT-4** with **Function Calling** for content analysis and task extraction.
* **Embeddings API** for creating a semantic search engine.
* **DALL-E 3** for generating visual summaries.

---

## 📦 Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nikakogho/KIU_SMART_MEETING_ASSISTANT.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd KIU_SMART_MEETING_ASSISTANT
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    * Create a `.env` file in the root directory.
    * Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY='your_api_key_here'
        ```

5.  **Run the application:**
    ```bash
    python app.py
    ```

---

## Usage

Once the application is running, open your web browser and navigate to `http://127.0.0.1:5000`. From the web interface, you can upload your meeting recordings and access the generated summaries, action items, and visual assets.

---

##  deliverables

### 🎥 5-Minute Video Demo

[Link to the video demo will be here.]

### プレゼンテーション 5-7 Minute Technical Presentation

[Link to the presentation slides or recording will be here.]

---

## ✅ Test Cases

The project includes a suite of pre-written test cases to ensure the functionality and reliability of the system. These tests cover all the core features and can be found in the `/tests` directory.
