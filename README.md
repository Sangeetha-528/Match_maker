# Match_maker

# 💘 Match_maker

## VFE Matchmaker – Soulmate Finder

This is a **Streamlit-based matchmaking app** that uses **OpenAI embeddings** to find your top 5 soulmate matches based on your thoughts, emotions, and beliefs.

---

### 🚀 Features

- Embeds user profile data using OpenAI's `text-embedding-ada-002` model
- Matches users based on semantic similarity using cosine similarity
- Clean, intuitive form interface for entering profile information
- Displays top 5 most compatible profiles from the dataset

---

### 🧪 How It Works

1. Users fill out a form describing their views on life, family, relationships, memories, etc.
2. The app converts these responses into embeddings using OpenAI API.
3. It compares them to a database of existing profiles.
4. The top 5 matches are displayed with similarity scores and full profile details.

---

### 🖼️ Sample Data

The sample dataset (`extended_dating_app_user_profiles.csv`) contains diverse profiles with attributes like:
- Family meaning
- Favorite holiday destination
- Marriage opinions
- Relationship values
- Regrets, memories, and more

---

### 📦 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt

---

###📁 File Structure

Match_maker/
│
├── extended_dating_app_user_profiles.csv   # Sample dataset
├── main.py                                 # Main Streamlit app script
├── requirements.txt                        # Python dependencies
├── README.md                               # Project overview
└── .env                                    # (Add manually) OpenAI API Key

