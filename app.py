import streamlit as st
import pandas as pd
import os
import pickle
from io import BytesIO
import joblib
import base64
import matplotlib.pyplot as plt
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE

from ydata_profiling import ProfileReport


from pycaret.classification import *

# with st.sidebar:
#     st.title("Judul Skripsi")
#     choice = st.radio("Navigation", ["Dataset", "Overview", "SMOTE", "Meta Model", "Prediksi dan Akurasi"])


def home():
    judul = "Optimasi Algoritma Random Forest menggunakan K-Nearest Neighbor dan SMOTE pada Penyakit Diabetes"

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("image/unnes.png")
    st.markdown(
    "<h1 style='text-align: center'>Skripsi</h1>", unsafe_allow_html=True
    )
    st.markdown(
        f"<h1 style='text-align: center'>{judul}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align: center'>Disusun oleh: Syuja Zhafran Rakha Krishandhie</h2>",
        unsafe_allow_html=True,
    )


def flowchart():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<h4 style='text-align: center'>Flowchart Langkah Penelitian</h1>", unsafe_allow_html=True)
        st.image("image/KNNRF ENSEMBLE.png", use_column_width=True)


def upload_data():
    st.title("Upload Dataset")
    st.write("Upload Dataset anda berupa CSV")

    upload_file = st.file_uploader("Upload Dataset", type="csv")
    df = None
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write("Uploaded Diabetes Dataset:")
        st.dataframe(df)

    st.session_state.df = df



def overview():
    st.title("Diabetes Dataset Overview")
    st.write("Akan dilakukan Exploratory Data Analysis (EDA) untuk Dataset, agar mengetahui informasi dari Dataset yang akan digunakan.")

    if "df" not in st.session_state:
        st.session_state.df = None

    df = st.session_state.df

    with st.container():
        placeholder_ov = st.button("Lakukan overview pada Dataset")
        if placeholder_ov:
            profile = df.profile_report()
            st.write("Overview dari Pima Indians Diabetes Dataset:")
            st_profile_report(profile)


def missing():
    st.title("Pengisian Missing Value menggunakan Mean")
    st.write("Di tahap ini dataset yang memiliki nilai 0 atau missing value akan dihitung menggunakan mean.n\
        nilai dari missing value tersebut")
    df_clean = None

    if "df" not in st.session_state:
        st.session_state.df = None

    df = st.session_state.df

    
    with st.container():
        placeholder_miss = st.button("Mulai menghilangkan Missing Value")
        if placeholder_miss:
            with st.spinner("Menghilangkan Missing Value..."):
                means = df.mean()
                df.fillna(means, inplace=True)
                st.write("Dataset yang telah bersih:")
                st.dataframe(df)

                csv = df.to_csv(index=False)
                
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_dataset.csv">Click here to download</a>'
                st.markdown(href, unsafe_allow_html=True)

    st.session_state.df_clean = df


def split_data():
    if "df_clean" not in st.session_state:
        st.session_state.df = None

    df_clean = st.session_state.df_clean

    if df_clean is not None:
        X_train, X_test, y_train, y_test = train_test_split(df_clean.drop('Outcome', axis=1), df_clean['Outcome'], test_size=0.2, random_state=42)

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

def model():
    st.title("Setup Dataset untuk dilakukan Klasifikasi")
    st.write("Pada Tahapan ini, akan mempersiapkan dataset untuk diklasifikasi. Dataset akan di preprocess n\
        menggunakan SMOTE untuk menghilangkan ketidakseimbangan pada kolom categorical.")

    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None

    df_clean = st.session_state.df_clean
    with st.container():
        placeholder_md = st.button("Setup Klasifikasi")
        if placeholder_md:
            with st.spinner("Melakukan Setup pada Dataset..."):
                setup(data=df_clean, train_size=0.7, target="Outcome", fix_imbalance=True, fix_imbalance_method='smote')
                setup_df = pull()
                st.write("Setup Klasifikasi pada Dataset yang akan digunakan:")
                st.dataframe(setup_df, use_container_width=True)
                train_transformed = get_config('train_transformed')
                test_transformed = get_config('test_transformed')

                st.write("Transformed Traning Data shape:", train_transformed.shape)
                st.dataframe(train_transformed, use_container_width=True)

                # st.write("Transformed Test Data shape:", test_transformed.shape)
                # st.dataframe(test_transformed, use_container_width=True)

                st.write("Training Data yang telah bersih dari ketidakseimbangan data:")
                pd.value_counts(train_transformed['Outcome']).plot.bar()
                plt.title('Outcome')
                plt.xlabel('Kelas')
                plt.ylabel('Jumlah')
                train_transformed['Outcome'].value_counts()
                st.pyplot(plt)

                # for column in train_transformed.columns:
                #     value_counts = train_transformed['Outcome'].value_counts()
                #     value_counts_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})
                #     fig = px.bar(value_counts_df, x='Value', y='Count', title=f'Value Count Bar Chart of' ['Outcome'])
                #     st.plotly_chart(fig)


def best_rf():
    st.title("Hyperparameter Tuning untuk Algoritma Random Forest")
    st.write("Pada Tahapan ini, akan membuat sebuah Model Random Forest yang lalu akan dilakukan Hyperparameter Tuning untuk menemukan Parameter terbaik.")
    best_rf = None
    with st.container():
        placeholder_rf = st.button("Tuning RF")
        if placeholder_rf:
            with st.spinner("Melakukan Parameter Tuning untuk RF..."):
                best_rf = create_model('rf')
                rf_best = pull()
                st.info("Parameter Terbaik untuk Random Forest")
                st.dataframe(rf_best[['Accuracy']], use_container_width=True)


    st.session_state.best_rf = best_rf

def best_knn():
    st.title("Hyperparameter Tuning untuk Algoritma K-Nearest Neighbor")
    st.write("Pada Tahapan ini, akan membuat sebuah Model K-Nearest Neighbor yang lalu akan dilakukan Hyperparameter Tuning untuk menemukan Parameter terbaik.")
    best_knn = None
    with st.container():
        placeholder_knn = st.button("Tuning KNN")
        if placeholder_knn:
            with st.spinner("Melakukan Parameter Tuning untuk KNN..."):
                best_knn = create_model('knn')
                knn_best = pull()
                st.info("Parameter Terbaik untuk K-Nearest Neighbor")
                st.dataframe(knn_best[['Accuracy']], use_container_width=True)

    st.session_state.best_knn = best_knn
     
def stacking_model():
    st.title("Stacking Model untuk Algoritma Random Forest dan K-Nearest Neighbor")
    st.write("Pada Tahapan ini, akan melakukan inisiasi untuk Stacking Ensemble model, dimana akan menggabungkan base classifier yaitu RF dan KNN n\
        dan menggunakan RF sebagai Meta Classifier menjadi sebuah model.")


    if "best_rf" not in st.session_state:
        st.session_state.best_rf = False
    if "best_knn" not in st.session_state:
        st.session_state.best_knn = False

    best_rf = st.session_state.best_rf
    best_knn = st.session_state.best_knn
    stacked_models = None

    with st.container():
        placeholder_stack = st.button("Generate Stacking Model")
        if placeholder_stack:
            with st.spinner("Melakukan Building untuk Model Stacking..."):
                meta_model = RandomForestClassifier(random_state=42)
                stacked_models = stack_models(estimator_list=[best_rf, best_knn], meta_model= meta_model)
                stacked = pull()
                st.info("Hasil dari Stacking Model menggunakan Random Forest dan K-Nearest Neighbor")
                st.dataframe(stacked[['Accuracy']], use_container_width=True)
    st.session_state.stacked_models = stacked_models

def prediction():
    st.title("Prediksi Penyakit Diabetes menggunakan Algoritma Random Forest dan K-Nearest Neighbor")
    st.write("Pada Tahapan ini, akan melakukan testing kepada model, yang akan mencari berapa akurasi dari model setelah dilakukan testing.")

    if "stacked_models" not in st.session_state:
        st.session_state.stacked_models = False
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    stacked_models = st.session_state.stacked_models
    accuracy = None

    with st.container():
        placeholder_predict = st.button("Prediksi")
        if placeholder_predict:
            with st.spinner("Melakukan Prediksi menggunakan Random Forest dan K-Nearest Neighbor..."):
                stacked_model = finalize_model(stacked_models)
                base_models = [model for model in stacked_models.estimators_]
                predictions = pd.DataFrame()
                for model in base_models:
                    preds = model.predict(X_test)
                    predictions[model.__class__.__name__] = preds
                predicted_labels = predictions.mode(axis=1)[0]
                accuracy = accuracy_score(y_test, predicted_labels)
                # print("Accuracy of the stacking ensemble model:", accuracy)
                st.write("Akurasi prediksi Model:", accuracy)
    
    st.session_state.accuracy = accuracy

def download_model():

    if "stacked_models" not in st.session_state:
        st.session_state.stacked_models = False
    if "accuracy" not in st.session_state:
        st.session_state.accuracy = None

    stacked_models = st.session_state.stacked_models
    accuracy = st.session_state.accuracy
    model = stacked_models

    if accuracy is not None:
        st.download_button("Download Model",data=pickle.dumps(model),file_name="model.pkl")

def predict_model(model, data):
    return model.predict(data)

def form():
    st.title("Form Prediksi pada Penyakit Diabetes")
    st.write("Prediksi pada Penyakit Diabetes.")


    if "stacked_models" not in st.session_state:
        st.session_state.stacked_models = False

    file_upload = st.file_uploader("Choose PKL file", type="pkl")

    if file_upload is not None:
        model = pickle.load(file_upload)

        Pregnancies = st.text_input('Masukkan berapa kali pasien hamil')
        Glucose = st.text_input('Masukkan kadar Glukosa pasien')
        BloodPressure = st.text_input('Masukkan tekanan darah pasien')
        SkinThickness = st.text_input('Masukkan ketebalan kulit pasien')
        Insulin = st.text_input('Masukkan jumlah insulin pasien')
        BMI = st.text_input('Masukkan berat badan pasien')
        DiabetesPedigreeFunction = st.text_input('Masukkan faktor keturunan diabetes pasien')
        Age = st.text_input('Masukkan umur pasien')

        if st.button("Prediksi"):
            if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
                st.warning("Mohon lengkapi semua data input.")
            else:
                input_data = pd.DataFrame(
                    [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                )
            
                prediction = predict_model(model, data=input_data)
                st.success("Pasien Terkena Diabetes" if prediction[0] == 1 else "Pasien Tidak Terkena Diabetes")

        # if st.button("Prediksi"):
        #     input_data = pd.DataFrame(
        #         [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
        #         columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        #     )
        
        #     prediction = predict_model(model, data=input_data)
        #     st.success("Pasien Terkena Diabetes" if prediction[0] == 1 else "Pasien Tidak Terkena Diabetes")


# def form():
#     st.title("Form Prediksi pada Penyakit Diabetes")

#     if "df_clean" not in st.session_state:
#         st.session_state.df_clean = None

#     if "stacked_models" not in st.session_state:
#         st.session_state.stacked_models = False

#     df_clean = st.session_state.df_clean
#     stacked_models = st.session_state.stacked_models

#     file_upload = st.file_uploader("Choose PKL file", type="pkl")

#     model = pickle.load(file_upload, "rb")

#     Pregnancies = st.text_input('Masukkan berapa kali pasien hamil')
#     Glucose = st.text_input('Masukkan kadar Glukosa pasien')
#     BloodPressure = st.text_input('Masukkan tekanan darah pasien')
#     SkinThickness = st.text_input('Masukkan ketebalan kulit pasien')
#     Insulin = st.text_input('Masukkan jumlah insulin pasien')
#     BMI = st.text_input('Masukkan berat badan pasien')
#     DiabetesPedigreeFunction = st.text_input('Masukkan faktor keturunan diabetes pasien')
#     Age = st.text_input('Masukkan umur pasien')

#     diagnosis = ''

#     with st.container():
#         placeholder_form = st.button("Prediksi")
#         if placeholder_form:
#             input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
#             columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
#             prediction = predict_model(model, data=input_data)
#             st.write("Pasien Terkena Diabetes" if prediction['prediction_label'].iloc[0] == 1 else "Pasien Tidak Terkena Diabetes")
#             st.write("Akurasi Prediksi :" ,prediction['prediction_score'].iloc[0])

def related_works():
    related_works_dict = {
    "Judul": [
        "Classification of Pima Indian Diabetes Dataset using Ensemble of Decision Tree, Logistic Regression and Neural Network",
        "An ensemble approach for classification and prediction of diabetes mellitus using soft voting classifier",
        "Diabot: A Predictive Medical Chatbot using Ensemble Learning",
        "Prediksi Pima Indians Diabetes Database Dengan Ensemble Adaboost Dan Bagging",
        "Diabetes Prediction using Machine Learning Techniques",
        "Proposed Method",
    ],
    "Penulis": [
        "Abedini et al.",
        "Kumari et al.",
        "Bali et al.",
        "Rais et al.",
        "Soni & Varma",
        "Syuja Zhafran R.K",
    ],
    "Tahun": ["2020", "2021", "2019", "2021", "2020", "2023"],
    "Algoritma": [
        "Decision Tree, Logistic Regression, dan Neural Network ensemble",
        "Ensemble soft voting",
        "Ensemble Classifier",
        "SVM ensemble bagging",
        "SVM, KNN, RF, Decision Tree, Logistic Regression, dan Gradient Boosting classifiers",
        "Random Forest dan K-Nearest Neighbor denga SMOTE",
    ],
    "Dataset": [
        "Pima Indian Diabetes Dataset",
        "Pima Indian Diabetes Dataset",
        "Pima Indian Diabetes Dataset",
        "Pima Indian Diabetes Dataset",
        "Pima Indian Diabetes Dataset",
        "Pima Indian Diabetes Dataset",
    ],
    "Akurasi": ["83.03 %", "79.08 %", "84.2 %", "77.7 %","77 %", "-"],
    }
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Daftar Penelitian Terkait</h1>",
        unsafe_allow_html=True,
    )
    st.write('Berikut adalah daftar penelitian terkait yang digunakan penulis sebagai acuan dan perbandingan dalam pembuatan penelitian ini.')
    related_works = pd.DataFrame(related_works_dict)
    related_works.index += 1
    st.dataframe(related_works)


def main():
    with st.sidebar:
        st.title("Aplikasi Prediksi Penyakit Diabetes")
        choice = st.radio("Navigation", ["Home", "Flowchart Penelitian", "Dataset", "Overview", "Missing Value", "Setup", "RF Hyperparameter Tuning", 
        "KNN Hyperparameter Tuning", "Stacking Model", "Prediction", "Form Prediksi","Penelitian Terkait"])

    if choice == "Home":
        home()

    if choice == "Flowchart Penelitian":
        flowchart()

    if choice == "Dataset":
        upload_data()

    if choice == "Overview":
        overview()

    if choice == "Missing Value":
        missing()
        split_data()

    if choice == "Setup":
        model()

    if choice == "RF Hyperparameter Tuning":
        best_rf()

    if choice == "KNN Hyperparameter Tuning":
        best_knn()

    if choice == "Stacking Model":
        stacking_model()

    if choice == "Prediction":
        prediction()
        download_model()

    if choice == "Form Prediksi":
        form()

    if choice == "Penelitian Terkait":
        related_works()
        





if __name__ == "__main__":
    main()


# stacked_model = finalize_model(stacked_models)

# base_models = [model for model in stacked_models.estimators_]

# predictions = pd.DataFrame()
# for model in base_models:
#     preds = model.predict(X_test)
#     predictions[model.__class__.__name__] = preds
    
# predicted_labels = predictions.mode(axis=1)[0]

# accuracy = accuracy_score(y_test, predicted_labels)
# print("Accuracy of the stacking ensemble model:", accuracy)