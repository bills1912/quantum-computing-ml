import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Sampler

# Set page config
st.set_page_config(
    page_title="Quantum ML Insurance Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: inherit;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .info-box {
            background-color: rgba(102, 126, 234, 0.15);
            border-left-color: #8b9bff;
        }
        .sub-header {
            color: #8b9bff;
        }
    }
    
    /* Streamlit dark theme class support */
    [data-theme="dark"] .info-box,
    .stApp[data-theme="dark"] .info-box {
        background-color: rgba(102, 126, 234, 0.15);
        border-left-color: #8b9bff;
    }
    
    [data-theme="dark"] .sub-header,
    .stApp[data-theme="dark"] .sub-header {
        color: #8b9bff;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 Quantum Machine Learning Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Analisis Biaya Asuransi Kesehatan dengan Variational Quantum Classifier</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/Qiskit-Logo.svg", width=200)
    st.markdown("### ⚙️ Konfigurasi Model")
    
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=['csv'])
    
    st.markdown("---")
    st.markdown("### 🎛️ Parameter Quantum Model")
    
    feature_dim = st.slider("Dimensi Fitur", 2, 6, 4, help="Jumlah fitur untuk quantum encoding")
    reps = st.slider("Repetitions (Ansatz)", 1, 5, 2, help="Kedalaman quantum circuit")
    max_iter = st.slider("Max Iterations", 10, 100, 50, help="Iterasi optimisasi")
    
    st.markdown("---")
    st.markdown("### 📊 Opsi Visualisasi")
    show_eda = st.checkbox("Exploratory Data Analysis", value=True)
    show_quantum = st.checkbox("Quantum Circuit", value=True)
    show_results = st.checkbox("Model Performance", value=True)

# Load and process data
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Fallback: create sample data jika file tidak diupload
        st.warning("⚠️ Dataset belum diupload. Menggunakan data sampel untuk demonstrasi.")
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, 1000),
            'sex': np.random.choice(['male', 'female'], 1000),
            'bmi': np.random.normal(30, 6, 1000),
            'children': np.random.randint(0, 5, 1000),
            'smoker': np.random.choice(['yes', 'no'], 1000),
            'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], 1000),
            'charges': np.random.exponential(13000, 1000)
        })
    return df

df = load_data(uploaded_file)

# Main content
if df is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 EDA", "⚛️ Quantum Model", "📈 Results"])
    
    with tab1:
        st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>Total Records</h3><h2>{}</h2></div>'.format(len(df)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Features</h3><h2>{}</h2></div>'.format(len(df.columns)), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Avg Charges</h3><h2>${:,.0f}</h2></div>'.format(df['charges'].mean()), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>Missing Values</h3><h2>{}</h2></div>'.format(df.isnull().sum().sum()), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📋 Data Sample")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("##### 🔢 Data Types")
        st.dataframe(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes, 'Non-Null': df.count()}), use_container_width=True)
    
    with tab2:
        if show_eda:
            st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
            
            # Distribusi Charges
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='charges', nbins=50, 
                                 title='Distribusi Biaya Asuransi',
                                 labels={'charges': 'Biaya ($)', 'count': 'Frekuensi'},
                                 color_discrete_sequence=['#667eea'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y='charges', 
                           title='Box Plot Biaya Asuransi',
                           labels={'charges': 'Biaya ($)'},
                           color_discrete_sequence=['#764ba2'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Analisis kategorikal
            col1, col2 = st.columns(2)
            
            with col1:
                if 'smoker' in df.columns:
                    fig = px.box(df, x='smoker', y='charges', 
                               title='Biaya berdasarkan Status Merokok',
                               labels={'smoker': 'Perokok', 'charges': 'Biaya ($)'},
                               color='smoker',
                               color_discrete_sequence=['#667eea', '#f093fb'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'sex' in df.columns:
                    fig = px.violin(df, x='sex', y='charges', 
                                  title='Biaya berdasarkan Jenis Kelamin',
                                  labels={'sex': 'Jenis Kelamin', 'charges': 'Biaya ($)'},
                                  color='sex',
                                  color_discrete_sequence=['#667eea', '#764ba2'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.markdown("##### 🔥 Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(title='Correlation Matrix', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            if 'age' in df.columns and 'bmi' in df.columns:
                fig = px.scatter(df, x='age', y='bmi', color='charges',
                               title='Hubungan Age, BMI, dan Charges',
                               labels={'age': 'Umur', 'bmi': 'BMI', 'charges': 'Biaya ($)'},
                               color_continuous_scale='Viridis',
                               size='charges', size_max=15)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Quantum Machine Learning Model</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">ℹ️ <b>Variational Quantum Classifier (VQC)</b> menggunakan quantum circuit variational untuk klasifikasi. Model ini memanfaatkan prinsip superposisi dan entanglement kuantum untuk pembelajaran.</div>', unsafe_allow_html=True)
        
        # Prepare data untuk klasifikasi
        with st.spinner('🔄 Memproses data untuk Quantum Model...'):
            # Buat target klasifikasi (high vs low cost)
            threshold = df['charges'].median()
            df['cost_class'] = (df['charges'] > threshold).astype(int)
            
            # Encode categorical variables
            le_dict = {}
            df_encoded = df.copy()
            for col in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                le_dict[col] = le
            
            # Select features
            feature_cols = [col for col in df_encoded.columns if col not in ['charges', 'cost_class']]
            feature_cols = feature_cols[:feature_dim]  # Limit to feature_dim
            
            X = df_encoded[feature_cols].values
            y = df_encoded['cost_class'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### 📊 Data Preparation")
            st.write(f"**Training samples:** {len(X_train)}")
            st.write(f"**Testing samples:** {len(X_test)}")
            st.write(f"**Features used:** {feature_cols}")
            st.write(f"**Target classes:** 0 (Low Cost), 1 (High Cost)")
            st.write(f"**Threshold:** ${threshold:,.2f}")
            
            # Class distribution
            fig = go.Figure(data=[
                go.Bar(x=['Low Cost', 'High Cost'], 
                      y=[sum(y==0), sum(y==1)],
                      marker_color=['#667eea', '#764ba2'])
            ])
            fig.update_layout(title='Class Distribution', 
                            xaxis_title='Class', 
                            yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if show_quantum:
                st.markdown("##### ⚛️ Quantum Circuit Architecture")
                
                # Create feature map
                feature_map = ZZFeatureMap(feature_dimension=len(feature_cols), reps=1)
                
                # Create ansatz
                ansatz = RealAmplitudes(num_qubits=len(feature_cols), reps=reps)
                
                # Draw circuit
                circuit = QuantumCircuit(len(feature_cols))
                circuit.compose(feature_map, inplace=True)
                circuit.compose(ansatz, inplace=True)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                circuit.draw('mpl', ax=ax, style='iqp')
                st.pyplot(fig)
                
                st.markdown(f"""
                **Circuit Details:**
                - Qubits: {len(feature_cols)}
                - Feature Map: ZZFeatureMap
                - Ansatz: RealAmplitudes
                - Repetitions: {reps}
                - Total Gates: {circuit.size()}
                """)
        
        # Train model button
        st.markdown("---")
        
        if st.button("🚀 Train Quantum Model", type="primary", use_container_width=True):
            with st.spinner('⚛️ Training Variational Quantum Classifier... (ini mungkin memakan waktu)'):
                try:
                    # Set seed
                    algorithm_globals.random_seed = 42
                    
                    # Create VQC
                    feature_map = ZZFeatureMap(feature_dimension=len(feature_cols), reps=1)
                    ansatz = RealAmplitudes(num_qubits=len(feature_cols), reps=reps)
                    
                    # Use COBYLA optimizer
                    optimizer = COBYLA(maxiter=max_iter)
                    
                    # Create sampler
                    sampler = Sampler()
                    
                    # Create VQC
                    vqc = VQC(
                        sampler=sampler,
                        feature_map=feature_map,
                        ansatz=ansatz,
                        optimizer=optimizer,
                    )
                    
                    # Reduce training data untuk demo (quantum simulasi lambat)
                    X_train_small = X_train_scaled[:100]
                    y_train_small = y_train[:100]
                    
                    # Train
                    vqc.fit(X_train_small, y_train_small)
                    
                    # Predict
                    y_pred_train = vqc.predict(X_train_small)
                    y_pred_test = vqc.predict(X_test_scaled[:50])  # Test pada subset
                    y_test_small = y_test[:50]
                    
                    # Store results in session state
                    st.session_state['vqc_trained'] = True
                    st.session_state['y_pred_train'] = y_pred_train
                    st.session_state['y_pred_test'] = y_pred_test
                    st.session_state['y_train_small'] = y_train_small
                    st.session_state['y_test_small'] = y_test_small
                    st.session_state['train_acc'] = accuracy_score(y_train_small, y_pred_train)
                    st.session_state['test_acc'] = accuracy_score(y_test_small, y_pred_test)
                    
                    st.success("✅ Model berhasil dilatih!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error saat training: {str(e)}")
                    st.info("💡 Tips: Coba kurangi dimensi fitur atau max iterations")
    
    with tab4:
        if show_results:
            st.markdown('<div class="sub-header">Model Performance & Results</div>', unsafe_allow_html=True)
            
            if 'vqc_trained' in st.session_state and st.session_state['vqc_trained']:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>Train Accuracy</h3><h2>{st.session_state["train_acc"]:.2%}</h2></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>Test Accuracy</h3><h2>{st.session_state["test_acc"]:.2%}</h2></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>Train Samples</h3><h2>{len(st.session_state["y_train_small"])}</h2></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-card"><h3>Test Samples</h3><h2>{len(st.session_state["y_test_small"])}</h2></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 🎯 Confusion Matrix - Test Set")
                    cm = confusion_matrix(st.session_state['y_test_small'], st.session_state['y_pred_test'])
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Low', 'Predicted High'],
                        y=['Actual Low', 'Actual High'],
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16},
                        colorscale='Purples'
                    ))
                    fig.update_layout(title='Confusion Matrix', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### 📊 Classification Report")
                    report = classification_report(
                        st.session_state['y_test_small'], 
                        st.session_state['y_pred_test'],
                        target_names=['Low Cost', 'High Cost'],
                        output_dict=True
                    )
                    
                    df_report = pd.DataFrame(report).transpose()
                    st.dataframe(df_report.round(3), use_container_width=True)
                
                # Accuracy comparison
                st.markdown("##### 📈 Model Performance Comparison")
                
                fig = go.Figure(data=[
                    go.Bar(name='Train', x=['Accuracy'], y=[st.session_state['train_acc']], marker_color='#667eea'),
                    go.Bar(name='Test', x=['Accuracy'], y=[st.session_state['test_acc']], marker_color='#764ba2')
                ])
                fig.update_layout(barmode='group', yaxis_title='Accuracy', yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure(data=[
                        go.Pie(labels=['Low Cost', 'High Cost'], 
                              values=[sum(st.session_state['y_pred_test']==0), sum(st.session_state['y_pred_test']==1)],
                              marker_colors=['#667eea', '#764ba2'])
                    ])
                    fig.update_layout(title='Predicted Class Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure(data=[
                        go.Pie(labels=['Low Cost', 'High Cost'], 
                              values=[sum(st.session_state['y_test_small']==0), sum(st.session_state['y_test_small']==1)],
                              marker_colors=['#667eea', '#764ba2'])
                    ])
                    fig.update_layout(title='Actual Class Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("⚠️ Model belum dilatih. Silakan kembali ke tab 'Quantum Model' dan klik tombol 'Train Quantum Model'.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Quantum Machine Learning Dashboard</b></p>
    <p>Powered by Qiskit | Built with Streamlit</p>
    <p>⚛️ Memanfaatkan kekuatan komputasi kuantum untuk prediksi biaya asuransi kesehatan</p>
</div>
""", unsafe_allow_html=True)