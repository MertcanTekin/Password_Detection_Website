import joblib
import streamlit as st

#word fonksiyonunu geri çağırıyoruz
def word(password):
    character = []
    for i in password:
        character.append(i)
    return character
#kaydedilmiş modelleri geri yüklüyoruz
model_for_tdif=joblib.load("C:/Users/user/Desktop/tdif.pkl")
model_from_disc=joblib.load("C:/Users/user/Desktop/passworddetection.pkl")



#Streamlit Uygulaması
page_bg_img="""
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://media.wired.com/photos/641e1a1b43ffd37beea02cdf/master/w_1600,c_limit/Best%20Password%20Managers%20Gear%20GettyImages-1408198405.png");
background-size: cover;
}
<style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("Password Strength Checker")
st.write("Welcome to the Password Strength Checker! Enter a password to see how strong it is.")
user_input = st.text_input("Enter your password:", type="password")

if st.button("Check Strength"):
    if len(user_input) < 1:
        st.warning("Please enter a password.")
    else:
        sample = user_input
        data = model_for_tdif.transform([sample]).toarray()
        a = model_from_disc.predict(data)[0]

        if a == 'Weak':
            color = "red"
            explanation = "This is a weak password. Please use a stronger one."
        elif a == 'Medium':
            color = "blue"
            explanation = "This is a medium-strength password. You can make it stronger by adding numbers and symbols."
        elif a == 'Strong':
            color = "green"
            explanation = "Congratulations! This is a strong password."

        styled_text = f'<span style="color: {color}; background-color: lightgray; padding: 10px; border-radius: 10px; font-size: 18px;">{a}</span>'
        st.markdown(styled_text, unsafe_allow_html=True)
        
        # explanation için de lightgray arkaplan eklenmiş hali
        explanation_with_bg = f'<div style="background-color: lightgray; padding: 10px; border-radius: 10px;">{explanation}</div>'
        st.markdown(explanation_with_bg, unsafe_allow_html=True)
