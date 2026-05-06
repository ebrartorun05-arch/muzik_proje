import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Müzik Öneri Sistemi", layout="wide")
st.title("🎵 Akıllı Müzik Öneri Sistemi")

@st.cache_data
def load_data():
    # encoding='utf-8' yaparak Türkçe karakter desteği sağlıyoruz
    df = pd.read_csv("dataset.csv", encoding='utf-8')
    return df

try:
    df = load_data()
    feature_cols = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=feature_cols)

    # Veriyi ölçeklendir
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])
    df_scaled_df = pd.DataFrame(df_scaled, columns=feature_cols, index=df.index)

    song_list = sorted(df['track_name'].astype(str).unique())
    selected_song = st.selectbox("Hangi şarkıyı çok seviyorsun?", song_list)

    if st.button('Benzer Şarkıları Bul'):
        # Hafıza dostu yöntem: Sadece seçilen şarkının benzerliğini hesapla
        idx = df[df['track_name'] == selected_song].index[0]
        selected_song_vector = df_scaled_df.loc[idx].values.reshape(1, -1)
        
        # Tüm matrisi hesaplamak yerine tek tek benzerlik alıyoruz
        similarities = cosine_similarity(selected_song_vector, df_scaled_df)
        
        # Sonuçları sırala
        df['similarity'] = similarities[0]
        recommendations = df[df['track_name'] != selected_song].sort_values('similarity', ascending=False).head(5)
        
        st.subheader(f"'{selected_song}' sevenler bunları da dinledi:")
        cols = st.columns(5)
        for i, (index, row) in enumerate(recommendations.iterrows()):
            with cols[i]:
                st.success(f"**{row['track_name']}**")
                st.caption(f"Sanatçı: {row['artists']}")

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")