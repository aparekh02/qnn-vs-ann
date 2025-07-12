import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Quantum Neural Network implementation (simplified for demonstration)
class QuantumNeuralNetwork:
    """
    Simplified Quantum Neural Network implementation for regression
    Uses quantum-inspired optimization for non-linear regression
    """
    def __init__(self, input_dim, hidden_dim=10, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize quantum-inspired parameters
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.random.randn(hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.random.randn(output_dim) * 0.1
        
        # Quantum-inspired parameters
        self.theta = np.random.randn(hidden_dim) * np.pi
        self.phi = np.random.randn(hidden_dim) * np.pi
        
    def quantum_activation(self, x):
        """Quantum-inspired activation function"""
        return np.tanh(x) * np.cos(self.theta) + np.sin(self.phi) * np.exp(-x**2)
    
    def forward(self, X):
        """Forward pass through the quantum neural network"""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.quantum_activation(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the quantum neural network"""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        for epoch in range(epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.quantum_activation(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            
            # Compute loss
            loss = np.mean((z2 - y)**2)
            
            # Backward pass (simplified)
            m = X.shape[0]
            
            # Output layer gradients
            dz2 = (z2 - y) / m
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0)
            
            # Hidden layer gradients
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (1 - np.tanh(z1)**2)  # Simplified derivative
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0)
            
            # Update weights
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            
            # Update quantum parameters
            self.theta -= learning_rate * 0.01 * np.random.randn(self.hidden_dim)
            self.phi -= learning_rate * 0.01 * np.random.randn(self.hidden_dim)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.forward(X)

# Load and prepare data
def load_data():
    """Load the soil nutrient data"""
    data = pd.read_csv('56608ee4e4b071e7ea544e04.csv')
    return data

def prepare_data(data):
    """Prepare data for analysis"""
    # Define nutrient columns
    nutrients = ['TN', 'NO3', 'NH4', 'Ca', 'Mg', 'K', 'P', 'Fe', 'Mn', 'Cu', 'Zn', 'B', 'S', 'Pb', 'Al', 'Cd']
    
    # Create a copy of the data
    df = data.copy()
    
    # Scale the data
    scaler = StandardScaler()
    df[nutrients] = scaler.fit_transform(df[nutrients])
    
    return df, nutrients, scaler

def analyze_dry_days_correlation(df, nutrients):
    """Analyze correlation between dry days and nutrients"""
    correlations = {}
    
    plt.figure(figsize=(20, 15))
    
    for i, nutrient in enumerate(nutrients, 1):
        plt.subplot(4, 4, i)
        
        # Calculate correlation
        corr = df['Dry_Days'].corr(df[nutrient])
        correlations[nutrient] = corr
        
        # Create scatter plot
        plt.scatter(df['Dry_Days'], df[nutrient], alpha=0.6, s=30)
        
        # Fit linear regression
        X = df['Dry_Days'].values.reshape(-1, 1)
        y = df[nutrient].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Plot regression line
        X_plot = np.linspace(df['Dry_Days'].min(), df['Dry_Days'].max(), 100).reshape(-1, 1)
        y_plot = lr.predict(X_plot)
        plt.plot(X_plot, y_plot, 'r--', linewidth=2)
        
        plt.title(f'{nutrient} vs Dry Days\nCorr: {corr:.3f}')
        plt.xlabel('Dry Days')
        plt.ylabel(f'{nutrient} (standardized)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Nutrient Correlations with Dry Days Before Flash Flood', fontsize=16, y=1.02)
    plt.show()
    
    return correlations

def soil_health_analysis(df):
    """Analyze soil health progression with QNN"""
    print("Analyzing Soil Health Progression with Quantum Neural Network...")
    
    # Create soil health index (composite score)
    health_nutrients = ['TN', 'Ca', 'Mg', 'K', 'P', 'Fe']
    toxic_nutrients = ['Pb', 'Al', 'Cd']
    
    # Calculate soil health index
    df['Soil_Health_Index'] = (
        df[health_nutrients].mean(axis=1) - 
        df[toxic_nutrients].mean(axis=1)
    )
    
    # Prepare data for QNN
    X = df[['Dry_Days']].values
    y = df['Soil_Health_Index'].values
    
    # Scale data for QNN
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Train QNN
    qnn = QuantumNeuralNetwork(input_dim=1, hidden_dim=20)
    qnn.fit(X_scaled, y_scaled, epochs=500, learning_rate=0.01)
    
    # Make predictions
    X_plot = np.linspace(df['Dry_Days'].min(), df['Dry_Days'].max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler_X.transform(X_plot)
    y_pred_scaled = qnn.predict(X_plot_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Dry_Days'], df['Soil_Health_Index'], alpha=0.6, s=50, label='Actual Data')
    plt.plot(X_plot, y_pred, 'r-', linewidth=3, label='QNN Prediction')
    plt.xlabel('Dry Days Before Flash Flood')
    plt.ylabel('Soil Health Index')
    plt.title('Soil Health Progression (QNN Non-linear Regression)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return qnn, scaler_X, scaler_y

def nutrient_management_recommendations(df, nutrients, correlations):
    """Generate nutrient management recommendations using QNN"""
    print("Generating Nutrient Management Recommendations...")
    
    # Filter nutrients with significant correlation
    significant_nutrients = [n for n, corr in correlations.items() if abs(corr) > 0.3]
    
    if not significant_nutrients:
        print("No nutrients found with significant correlation to dry days.")
        return
    
    plt.figure(figsize=(20, 15))
    
    for i, nutrient in enumerate(significant_nutrients, 1):
        plt.subplot(3, 3, i)
        
        # Prepare data
        X = df[['Dry_Days']].values
        y = df[nutrient].values
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train QNN for this nutrient
        qnn_nutrient = QuantumNeuralNetwork(input_dim=1, hidden_dim=15)
        qnn_nutrient.fit(X_scaled, y_scaled, epochs=300, learning_rate=0.01)
        
        # Generate recommendations for different addition amounts
        base_dry_days = np.array([30, 60, 90]).reshape(-1, 1)  # Different scenarios
        addition_amounts = np.linspace(0, 2, 50)  # Different nutrient addition amounts
        
        recommendations = []
        
        for days in base_dry_days.flatten():
            day_scaled = scaler_X.transform([[days]])
            base_level = qnn_nutrient.predict(day_scaled)[0]
            
            # Simulate nutrient addition effects
            enhanced_levels = []
            for add_amount in addition_amounts:
                # Simulate the effect of adding nutrients
                enhanced_input = day_scaled + add_amount * 0.1  # Scaling factor
                enhanced_level = qnn_nutrient.predict(enhanced_input)[0]
                enhanced_levels.append(enhanced_level)
            
            plt.plot(addition_amounts, enhanced_levels, 
                    label=f'{days} dry days', linewidth=2)
        
        plt.xlabel('Nutrient Addition Amount (standardized)')
        plt.ylabel(f'{nutrient} Level Response')
        plt.title(f'{nutrient} Management\nCorr: {correlations[nutrient]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Nutrient Management Recommendations (QNN-based)', fontsize=16, y=1.02)
    plt.show()

def temporal_nutrient_changes(df, nutrients):
    """Analyze temporal changes in nutrients over time"""
    print("Analyzing Temporal Nutrient Changes...")
    
    plt.figure(figsize=(20, 12))
    
    for i, nutrient in enumerate(nutrients[:12], 1):  # Show first 12 nutrients
        plt.subplot(3, 4, i)
        
        # Group by time and treatment
        for treatment in df['Treat'].unique():
            treat_data = df[df['Treat'] == treatment]
            time_avg = treat_data.groupby('Time')[nutrient].mean()
            
            # Fit QNN for temporal prediction
            X = np.array(time_avg.index).reshape(-1, 1)
            y = time_avg.values
            
            if len(X) > 2:  # Need at least 3 points for fitting
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                
                qnn_temp = QuantumNeuralNetwork(input_dim=1, hidden_dim=10)
                qnn_temp.fit(X_scaled, y_scaled, epochs=200, learning_rate=0.02)
                
                # Predict future values
                X_future = np.linspace(X.min(), X.max() + 2, 50).reshape(-1, 1)
                X_future_scaled = scaler_X.transform(X_future)
                y_future_scaled = qnn_temp.predict(X_future_scaled)
                y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1)).flatten()
                
                plt.plot(X_future, y_future, '-', linewidth=2, label=f'Treatment {treatment}')
                plt.scatter(X, y, s=50, alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel(f'{nutrient} (standardized)')
        plt.title(f'{nutrient} Temporal Changes')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Temporal Nutrient Changes Over Time (QNN Predictions)', fontsize=16, y=1.02)
    plt.show()

def main():
    """Main analysis function"""
    print("=== Soil Nutrient Analysis with Quantum Neural Networks ===\n")
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = load_data()
    df, nutrients, scaler = prepare_data(data)
    
    print(f"Data shape: {df.shape}")
    print(f"Nutrients analyzed: {len(nutrients)}")
    print(f"Dry days range: {df['Dry_Days'].min()} - {df['Dry_Days'].max()}")
    
    # Analyze correlations
    print("\n1. Analyzing Dry Days Correlations...")
    correlations = analyze_dry_days_correlation(df, nutrients)
    
    # Display significant correlations
    print("\nSignificant correlations (|r| > 0.3):")
    for nutrient, corr in correlations.items():
        if abs(corr) > 0.3:
            print(f"  {nutrient}: {corr:.3f}")
    
    # Soil health analysis
    print("\n2. Soil Health Analysis...")
    qnn, scaler_X, scaler_y = soil_health_analysis(df)
    
    # Nutrient management recommendations
    print("\n3. Nutrient Management Recommendations...")
    nutrient_management_recommendations(df, nutrients, correlations)
    
    # Temporal analysis
    print("\n4. Temporal Nutrient Changes...")
    temporal_nutrient_changes(df, nutrients)
    
    print("\n=== Analysis Complete ===")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Treatments: {df['Treat'].unique()}")
    print(f"Time periods: {df['Time'].unique()}")
    print(f"Sites: {df['Site_ID'].nunique()}")

if __name__ == "__main__":
    main()