
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# a Dummy dataset
data = {
    'title': ['Introduction to Algorithms', 'Artificial Intelligence: A Modern Approach', 'Deep Learning', 
              'Computer Networks', 'Clean Code', 'Design Patterns', 'Data Science from Scratch'],
    'author': ['Thomas H. Cormen', 'Stuart Russell', 'Ian Goodfellow', 
               'Andrew S. Tanenbaum', 'Robert C. Martin', 'Erich Gamma', 'Joel Grus'],
    'genre': ['Algorithms', 'AI', 'Machine Learning', 'Networking', 'Programming', 'Software Design', 'Data Science'],
    'description': [
        'A comprehensive textbook on algorithms and their analysis.',
        'A leading book on AI with theory and practical applications.',
        'The bible of deep learning methods and techniques.',
        'Covers the basics and advanced topics of computer networks.',
        'A guide to writing readable, maintainable, and clean code.',
        'Classic patterns for designing reusable object-oriented software.',
        'A beginner-friendly introduction to data science concepts.'
    ]
}
df_books = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

df_books['combined_features'] = df_books['title'] + ' ' + df_books['author'] + ' ' + df_books['genre'] + ' ' + df_books['description']
df_books['processed_features'] = df_books['combined_features'].apply(preprocess_text)

# Recommendation function
def get_recommendations(user_input, df):
    user_input_processed = preprocess_text(user_input)
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(df['processed_features'])
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, vectors)
    df['similarity'] = similarity[0]
    recommendations = df.sort_values(by='similarity', ascending=False).head(5)
    return recommendations[['title', 'author', 'genre']]

# Streamlit App
def main():
    # Adding the  sidebar
    st.sidebar.header("Customize Your Search")
    selected_genre = st.sidebar.selectbox("Select a genre:", options=["All"] + df_books['genre'].unique().tolist())
    selected_difficulty = st.sidebar.radio("Difficulty Level:", ["Beginner", "Intermediate", "Advanced"])
    st.sidebar.write("Use the main interface to input more details.")

    st.title("📚 Computing  learning Recommendation System")
    st.markdown("Describe the topics you're interested in learning about:")

    #  the Input section
    user_input = st.text_area("Your interests:", placeholder="e.g., deep learning, algorithms, clean code")
    
    # Button to get recommendations
    if st.button("Get Recommendations"):
        if not user_input.strip():
            st.warning("Please enter some text in the input box to get recommendations!")
        else:
            filtered_df = df_books if selected_genre == "All" else df_books[df_books['genre'] == selected_genre]
            recommendations = get_recommendations(user_input, filtered_df)
            if recommendations.empty:
                st.info("No recommendations found. Try broadening your search!")
            else:
                st.success("Here are some book recommendations for you:")
                for _, row in recommendations.iterrows():
                    st.write(f"**{row['title']}** by {row['author']} ({row['genre']})")

    # Sidebar feature for top recommendations
    st.sidebar.subheader("Top Recommendations")
    top_books = df_books.sample(3)  # Show 3 random books
    for _, row in top_books.iterrows():
        st.sidebar.write(f"- **{row['title']}** ({row['genre']})")

    # Footer section
    st.markdown("---")
    st.markdown("🔖 *Enhance your learning journey with curated book recommendations!*")

if __name__ == "__main__":
    main()
