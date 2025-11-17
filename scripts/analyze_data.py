"""
Quick data analysis script to understand the dataset
Run this before starting the Streamlit app
"""

import pandas as pd
import numpy as np

def analyze_dataset():
    """Analyze the mitochondrial morphology dataset"""
    
    print("=" * 70)
    print("ANÁLISIS DEL DATASET - MORFOLOGÍA MITOCONDRIAL")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('data/data.csv')
    
    # Basic info
    print(f"\n📊 INFORMACIÓN BÁSICA")
    print(f"{'─' * 70}")
    print(f"Total de observaciones: {len(df)}")
    print(f"Total de participantes: {df['Participant'].nunique()}")
    print(f"Número de columnas: {df.shape[1]}")
    print(f"Columnas: {', '.join(df.columns)}")
    
    # Group distribution
    print(f"\n🔬 DISTRIBUCIÓN POR GRUPO")
    print(f"{'─' * 70}")
    group_counts = df['Group'].value_counts()
    for group, count in group_counts.items():
        pct = (count / len(df)) * 100
        print(f"{group:5s}: {count:3d} observaciones ({pct:.1f}%)")
    
    # Sex distribution
    print(f"\n👥 DISTRIBUCIÓN POR SEXO")
    print(f"{'─' * 70}")
    sex_counts = df['Sex'].value_counts()
    for sex, count in sex_counts.items():
        pct = (count / len(df)) * 100
        print(f"{sex:8s}: {count:3d} observaciones ({pct:.1f}%)")
    
    # Cross-tabulation
    print(f"\n📋 DISTRIBUCIÓN POR GRUPO Y SEXO")
    print(f"{'─' * 70}")
    crosstab = pd.crosstab(df['Group'], df['Sex'])
    print(crosstab)
    
    # Participants per group
    print(f"\n👨‍🔬 PARTICIPANTES POR GRUPO")
    print(f"{'─' * 70}")
    ct_participants = df[df['Group'] == 'CT']['Participant'].nunique()
    ela_participants = df[df['Group'] == 'ELA']['Participant'].nunique()
    print(f"CT:  {ct_participants} participantes")
    print(f"ELA: {ela_participants} participantes")
    
    # Age statistics
    print(f"\n📅 ESTADÍSTICAS DE EDAD")
    print(f"{'─' * 70}")
    age_stats = df.groupby('Group')['Age'].agg(['mean', 'std', 'min', 'max'])
    print(age_stats.round(2))
    
    # Key metrics statistics
    print(f"\n📈 ESTADÍSTICAS DE MÉTRICAS MORFOLÓGICAS")
    print(f"{'─' * 70}")
    
    metrics = ['N mitocondrias', 'PROM IsoVol', 'PROM Surface', 'PROM Length', 'PROM RoughSph']
    
    print("\n🔵 GRUPO CONTROL (CT)")
    ct_stats = df[df['Group'] == 'CT'][metrics].describe().T[['mean', 'std', 'min', 'max']]
    print(ct_stats.round(3))
    
    print("\n🔴 GRUPO ELA")
    ela_stats = df[df['Group'] == 'ELA'][metrics].describe().T[['mean', 'std', 'min', 'max']]
    print(ela_stats.round(3))
    
    # Comparison
    print(f"\n🔬 COMPARACIÓN DE PROMEDIOS (CT vs ELA)")
    print(f"{'─' * 70}")
    comparison = df.groupby('Group')[metrics].mean()
    comparison['Diferencia (ELA - CT)'] = comparison.loc['ELA'] - comparison.loc['CT']
    comparison['Diferencia %'] = ((comparison.loc['ELA'] - comparison.loc['CT']) / comparison.loc['CT'] * 100)
    print(comparison.T.round(3))
    
    # Missing values
    print(f"\n❓ VALORES FALTANTES")
    print(f"{'─' * 70}")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No hay valores faltantes en el dataset")
    else:
        print(missing[missing > 0])
    
    # Observations per participant
    print(f"\n👤 OBSERVACIONES POR PARTICIPANTE")
    print(f"{'─' * 70}")
    obs_per_participant = df.groupby(['Participant', 'Group']).size().reset_index(name='N_obs')
    print(f"Promedio: {obs_per_participant['N_obs'].mean():.1f} observaciones por participante")
    print(f"Mínimo: {obs_per_participant['N_obs'].min()} observaciones")
    print(f"Máximo: {obs_per_participant['N_obs'].max()} observaciones")
    
    print("\n" + "=" * 70)
    print("✓ Análisis completado. Ejecuta 'streamlit run app.py' para la app interactiva")
    print("=" * 70)

if __name__ == "__main__":
    try:
        analyze_dataset()
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo data/data.csv")
        print("Asegúrate de estar en el directorio raíz del proyecto")
    except Exception as e:
        print(f"❌ Error: {e}")
