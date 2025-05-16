import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from collections import Counter
import json
import os


# Page configuration
st.set_page_config(
    page_title="Transaction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar always open
)

# Title
st.title("📊 Transaction Analysis")
st.markdown("Interactive visualization of transaction data")

# Function to extract addresses
def extract_addresses(df):
    addresses = set()
    
    # Add addresses from From and To fields
    if 'From' in df.columns:
        addresses.update(df['From'].unique())
    if 'To' in df.columns:
        addresses.update(df['To'].unique())
    
    # Try to extract addresses from nativeTransfers, if such column exists
    if 'nativeTransfers' in df.columns:
        for transfers in df['nativeTransfers']:
            try:
                if isinstance(transfers, str):
                    transfers_data = json.loads(transfers)
                else:
                    transfers_data = transfers
                
                if isinstance(transfers_data, list):
                    for transfer in transfers_data:
                        if isinstance(transfer, dict):
                            if 'fromUserAccount' in transfer:
                                addresses.add(transfer['fromUserAccount'])
                            if 'toUserAccount' in transfer:
                                addresses.add(transfer['toUserAccount'])
            except:
                continue
    
    return addresses

# Data loading
@st.cache_data(persist="disk")
def load_data(file_path=None, uploaded_file=None):
    try:
        if uploaded_file is not None:
            # Determine separator based on file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.txt':
                df = pd.read_csv(uploaded_file, sep='\t')
            else:  # Default for CSV
                df = pd.read_csv(uploaded_file)
        elif file_path and os.path.exists(file_path):
            # Determine separator based on file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                df = pd.read_csv(file_path, sep='\t')
            else:  # Default for CSV
                df = pd.read_csv(file_path)
        else:
            return None
        
        # Check and transform columns
        if 'Human Time' in df.columns:
            # Convert Human Time to datetime
            df['Human Time'] = pd.to_datetime(df['Human Time'], errors='coerce')
            
            # Add time characteristics for rows where time was correctly recognized
            date_mask = ~df['Human Time'].isna()
            if date_mask.any():
                df.loc[date_mask, 'date'] = df.loc[date_mask, 'Human Time'].dt.date
                df.loc[date_mask, 'day_of_week'] = df.loc[date_mask, 'Human Time'].dt.day_name()
                df.loc[date_mask, 'hour'] = df.loc[date_mask, 'Human Time'].dt.hour
        
        # Convert columns to numeric format, if they exist
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Upload file through file window
uploaded_file = st.sidebar.file_uploader("Upload transaction file", type=["txt", "csv"], 
                                       key="file_uploader_sidebar", 
                                       help="CSV and TXT formats are supported")

# Initialize data as None
data = None

if uploaded_file is not None:
    # Load data directly from uploaded file without saving
    data = load_data(uploaded_file=uploaded_file)
    if data is not None:
        st.sidebar.success("File successfully loaded! You can proceed with analysis.")

# If data was successfully loaded, display analysis
if data is not None:
    # Sidebar
    st.sidebar.header("Analysis Parameters")
    
    # Analysis selection menu
    analysis_option = st.sidebar.selectbox(
        "Select analysis type:",
        ["General Statistics", "Token Analysis", "Time Analysis", "Amount Analysis", "Wallet Analysis"]
    )
    
    # Display general statistics
    if analysis_option == "General Statistics":
        st.header("General Transaction Statistics")
        
        # Basic information about the data
        st.subheader("About Loaded Data")
        st.info(f"Loaded {len(data):,} rows and {len(data.columns)} columns")
        
        # Display first rows of the dataset
        with st.expander("View first 5 rows of data"):
            st.dataframe(data.head())
            
        # Display list of columns
        with st.expander("Available data columns"):
            st.write(", ".join(data.columns.tolist()))
        
        # Create metrics in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total transactions", f"{len(data):,}")
        
        with col2:
            if 'Value' in data.columns:
                st.metric("Total amount (Value)", f"{data['Value'].sum():.4f}")
            else:
                st.metric("Total amount (Value)", "No data")
        
        with col3:
            if 'Value' in data.columns:
                st.metric("Average transaction amount", f"{data['Value'].mean():.4f}")
            else:
                st.metric("Average transaction amount", "No data")
        
        # Information about data period
        if 'Human Time' in data.columns:
            st.subheader("Data Period")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Start date: {data['Human Time'].min().date()}")
            with col2:
                st.info(f"End date: {data['Human Time'].max().date()}")
        
        # Show raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.dataframe(data)
            
    # Token analysis
    elif analysis_option == "Token Analysis":
        st.header("Transaction Analysis by Tokens")
        
        if 'Token Address' in data.columns:
            # Group by tokens
            token_analysis = data.groupby('Token Address').agg({
                'Signature': 'count' if 'Signature' in data.columns else None,
                'Value': 'sum' if 'Value' in data.columns else None,
                'Amount': 'sum' if 'Amount' in data.columns else None
            }).reset_index()
            
            # Remove None columns
            token_analysis = token_analysis.dropna(axis=1, how='all')
            
            # Rename columns
            rename_dict = {'Token Address': 'Token'}
            if 'Signature' in token_analysis.columns:
                rename_dict['Signature'] = 'Number of Transactions'
            if 'Value' in token_analysis.columns:
                rename_dict['Value'] = 'Total Value'
            if 'Amount' in token_analysis.columns:
                rename_dict['Amount'] = 'Total Amount'
                
            token_analysis = token_analysis.rename(columns=rename_dict)
            
            # Display table with token data
            st.dataframe(token_analysis)
            
            # Visualize distribution by tokens
            st.subheader("Distribution of Transactions by Tokens")
            
            if 'Number of Transactions' in token_analysis.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                token_counts = token_analysis['Number of Transactions']
                token_names = token_analysis['Token']
                
                # Create pie chart
                ax.pie(token_counts, labels=token_names, autopct='%1.1f%%')
                ax.set_title('Distribution of Transactions by Tokens')
                st.pyplot(fig)
            else:
                st.warning("Insufficient data to display token distribution chart")
        else:
            st.warning("The data does not contain a 'Token Address' column. Please select another type of analysis or upload different data.")
        
    # Time analysis
    elif analysis_option == "Time Analysis":
        st.header("Time Analysis of Transactions")
        
        if 'Human Time' in data.columns:
            # Subsections for time analysis
            time_analysis = st.radio(
                "Select time period:",
                ["By Date", "By Day of Week", "By Hour of Day"]
            )
            
            if time_analysis == "By Date":
                # Group by date
                daily_activity = data.groupby('date').size()
                
                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                daily_activity.plot(kind='bar', color='skyblue', ax=ax)
                ax.set_title('Number of Transactions by Date')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Transactions')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data
                if st.checkbox("Show daily data"):
                    daily_df = pd.DataFrame({'Date': daily_activity.index, 'Count': daily_activity.values})
                    st.dataframe(daily_df, hide_index=True)
                
            elif time_analysis == "By Day of Week":
                # Group by day of week
                weekly_activity = data.groupby('day_of_week').size()
                
                # Correct day order
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Reorder days of week
                if set(weekly_activity.index).issubset(set(day_order)):
                    weekly_activity = weekly_activity.reindex(day_order)
                
                # Create DataFrame with days of week
                weekly_df = pd.DataFrame({
                    'Day of Week': weekly_activity.index,
                    'Count': weekly_activity.values
                })
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(weekly_df['Day of Week'], weekly_df['Count'], color='lightgreen')
                ax.set_title('Transaction Activity by Day of Week')
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Number of Transactions')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data
                if st.checkbox("Show data by day of week"):
                    st.dataframe(weekly_df, hide_index=True)
                    
            elif time_analysis == "By Hour of Day":
                # Group by hour
                hourly_activity = data.groupby('hour').size()
                
                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                hourly_activity.plot(kind='bar', color='coral', ax=ax)
                ax.set_title('Transaction Activity by Hour')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Number of Transactions')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data
                if st.checkbox("Show data by hour"):
                    hourly_df = pd.DataFrame({'Hour': hourly_activity.index, 'Count': hourly_activity.values})
                    st.dataframe(hourly_df, hide_index=True)
        else:
            st.warning("The data does not contain time information (column 'Human Time'). Please select another type of analysis or upload different data.")
    
    # Analysis by amounts
    elif analysis_option == "Amount Analysis":
        st.header("Transaction Amount Analysis")
        
        if 'Value' in data.columns:
            # Subsections for amount analysis
            sum_analysis = st.radio(
                "Select analysis type:",
                ["Amount Distribution", "Log-Scale Distribution", "Transaction Categories by Size", "Percentile Analysis"]
            )
            
            if sum_analysis == "Amount Distribution":
                # Allow user to filter outliers for better visualization
                include_outliers = st.checkbox("Include outliers in visualization", value=False)
                
                # Prepare figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if include_outliers:
                    # Full data histogram
                    sns.histplot(data['Value'], bins=50, kde=True, ax=ax)
                else:
                    # Filter outliers for visualization (keeping values below 95th percentile)
                    upper_limit = data['Value'].quantile(0.95)
                    filtered_data = data[data['Value'] <= upper_limit]
                    sns.histplot(filtered_data['Value'], bins=50, kde=True, ax=ax)
                    ax.set_title(f'Distribution of Transaction Amounts (≤ 95th percentile: {upper_limit:.4f})')
                    
                    # Add note about excluded data
                    st.info(f"For better visualization, the chart only shows transactions with amounts up to the 95th percentile ({upper_limit:.4f}). {len(data) - len(filtered_data)} outlier transactions have been excluded from the chart but are included in the statistics below.")
                
                ax.set_title('Distribution of Transaction Amounts')
                ax.set_xlabel('Amount (Value)')
                ax.set_ylabel('Number of Transactions')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics in a more detailed format
                st.subheader("Amount Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum amount", f"{data['Value'].min():.4f}")
                with col2:
                    st.metric("Average amount", f"{data['Value'].mean():.4f}")
                with col3:
                    st.metric("Maximum amount", f"{data['Value'].max():.4f}")
                
                # Additional statistics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median amount", f"{data['Value'].median():.4f}")
                with col2:
                    st.metric("Standard deviation", f"{data['Value'].std():.4f}")
                with col3:
                    st.metric("Total volume", f"{data['Value'].sum():.4f}")
                
            elif sum_analysis == "Log-Scale Distribution":
                # Create a log-scale plot for better visualization of skewed data
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Add a small constant to handle zero values if present
                min_non_zero = data['Value'][data['Value'] > 0].min() if any(data['Value'] > 0) else 0.01
                log_data = data['Value'].copy()
                log_data = log_data.replace(0, min_non_zero/10)  # Replace zeros with a small value
                
                # Plot log-scale histogram
                sns.histplot(log_data, bins=50, ax=ax)
                ax.set_xscale('log')
                ax.set_title('Distribution of Transaction Amounts (Log Scale)')
                ax.set_xlabel('Amount (Value) - Log Scale')
                ax.set_ylabel('Number of Transactions')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("Using a logarithmic scale allows us to better visualize the distribution across multiple orders of magnitude.")
                
            elif sum_analysis == "Transaction Categories by Size":
                # Create dynamic transaction amount categories based on data
                q1 = data['Value'].quantile(0.25)
                median = data['Value'].median()
                q3 = data['Value'].quantile(0.75)
                upper = data['Value'].quantile(0.95)
                
                # Adaptive bin edges based on data distribution
                value_bins = [0, q1, median, q3, upper, float('inf')]
                value_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
                
                # Show the actual ranges
                st.info(f"""
                Transaction categories are defined as:
                - Very Small: ≤ {q1:.4f}
                - Small: {q1:.4f} - {median:.4f}
                - Medium: {median:.4f} - {q3:.4f}
                - Large: {q3:.4f} - {upper:.4f}
                - Very Large: > {upper:.4f}
                """)
                
                data['value_category'] = pd.cut(data['Value'], bins=value_bins, labels=value_labels)
                
                # Distribution by amount categories
                value_distribution = data['value_category'].value_counts().sort_index()
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                value_distribution.plot(kind='bar', color='coral', ax=ax)
                ax.set_title('Distribution of Transactions by Amount Size')
                ax.set_xlabel('Amount Category')
                ax.set_ylabel('Number of Transactions')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data
                if st.checkbox("Show data by amount categories"):
                    value_df = pd.DataFrame({'Category': value_distribution.index, 'Count': value_distribution.values})
                    value_df['Percentage'] = (value_df['Count'] / value_df['Count'].sum()) * 100
                    value_df['Percentage'] = value_df['Percentage'].map('{:.2f}%'.format)
                    st.dataframe(value_df, hide_index=True)
                    
            elif sum_analysis == "Percentile Analysis":
                # Calculate percentiles
                percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
                percentile_values = [data['Value'].quantile(p/100) for p in percentiles]
                
                # Create and display percentile data
                percentile_df = pd.DataFrame({
                    'Percentile': [f"{p}%" for p in percentiles],
                    'Value': [f"{val:.4f}" for val in percentile_values]
                })
                
                st.subheader("Transaction Amount Percentiles")
                st.dataframe(percentile_df, hide_index=True)
                
                # Visualize the percentiles
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(percentiles, percentile_values, marker='o', linestyle='-', color='blue')
                plt.title('Transaction Amount Percentiles')
                plt.xlabel('Percentile')
                plt.ylabel('Amount (Value)')
                plt.grid(True)
                
                # Add non-linear x-axis to highlight the distribution better
                plt.xscale('linear')
                plt.xticks(percentiles)
                
                # Add value annotations for each point
                for i, txt in enumerate(percentile_values):
                    ax.annotate(f'{txt:.4f}', (percentiles[i], percentile_values[i]), 
                                textcoords="offset points", xytext=(0,10), ha='center')
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("The data does not contain a 'Value' column. Please select another type of analysis or upload different data.")
    
    # Wallet analysis
    elif analysis_option == "Wallet Analysis":
        st.header("Wallet Analysis")
        
        # Check for required columns
        has_wallet_data = 'From' in data.columns and 'To' in data.columns
        
        if has_wallet_data:
            # Select wallet analysis type
            wallet_analysis = st.radio(
                "Select analysis type:",
                ["Active Wallets", "Wallet Interactions"]
            )
            
            if wallet_analysis == "Active Wallets":
                # Sender analysis
                st.subheader("Most Active Senders")
                senders = data['From'].value_counts().reset_index()
                senders.columns = ['Wallet Address', 'Number of Sends']
                st.dataframe(senders.head(10))
                
                # Receiver analysis
                st.subheader("Most Active Receivers")
                receivers = data['To'].value_counts().reset_index()
                receivers.columns = ['Wallet Address', 'Number of Receives']
                st.dataframe(receivers.head(10))
                
                # Visualize top-5 senders
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Top-5 senders
                top_senders = senders.head(5)
                ax1.bar(top_senders['Wallet Address'], top_senders['Number of Sends'], color='skyblue')
                ax1.set_title('Top 5 Senders')
                ax1.set_xlabel('Wallet Address')
                ax1.set_ylabel('Number of Sends')
                ax1.tick_params(axis='x', rotation=45)
                
                # Top-5 receivers
                top_receivers = receivers.head(5)
                ax2.bar(top_receivers['Wallet Address'], top_receivers['Number of Receives'], color='lightgreen')
                ax2.set_title('Top 5 Receivers')
                ax2.set_xlabel('Wallet Address')
                ax2.set_ylabel('Number of Receives')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            elif wallet_analysis == "Wallet Interactions":
                # Analysis of interactions between wallets
                st.subheader("Most Frequent Interactions Between Wallets")
                
                # Group by sender-receiver pairs
                wallet_interactions = data.groupby(['From', 'To']).size().reset_index()
                wallet_interactions.columns = ['Sender', 'Receiver', 'Number of Transactions']
                wallet_interactions = wallet_interactions.sort_values('Number of Transactions', ascending=False)
                
                # Display top interactions
                st.dataframe(wallet_interactions.head(10))
                
                # Transaction amounts between wallets if Value column exists
                if 'Value' in data.columns:
                    st.subheader("Transaction Amounts Between Wallets")
                    wallet_value = data.groupby(['From', 'To'])['Value'].sum().reset_index()
                    wallet_value = wallet_value.sort_values('Value', ascending=False)
                    wallet_value.columns = ['Sender', 'Receiver', 'Total Amount']
                    
                    st.dataframe(wallet_value.head(10))
            
            # Get list of unique addresses
            unique_addresses = extract_addresses(data)
            
            # Display number of unique addresses
            st.metric("Unique addresses in transactions", f"{len(unique_addresses):,}")
            
            # Analysis of interactions from nativeTransfers column
            if 'nativeTransfers' in data.columns:
                st.subheader("Most Frequent Interactions from nativeTransfers")
                
                try:
                    # Collect pairs (sender, receiver)
                    transfer_pairs = []
                    
                    for idx, row in data.iterrows():
                        try:
                            transfers = row['nativeTransfers']
                            if isinstance(transfers, str):
                                transfers = json.loads(transfers)
                                
                            if isinstance(transfers, list):
                                for transfer in transfers:
                                    if isinstance(transfer, dict) and 'fromUserAccount' in transfer and 'toUserAccount' in transfer:
                                        transfer_pairs.append((transfer['fromUserAccount'], transfer['toUserAccount']))
                        except:
                            continue
                    
                    # Count frequency of each pair
                    pair_counts = Counter(transfer_pairs)
                    
                    # Display most frequent interactions
                    if pair_counts:
                        pairs_df = pd.DataFrame([
                            {"Sender": pair[0], "Receiver": pair[1], "Number of Transactions": count}
                            for pair, count in pair_counts.most_common(10)
                        ])
                        st.dataframe(pairs_df, hide_index=True)
                    else:
                        st.info("Could not find data about interactions between wallets in the provided dataset.")
                
                except Exception as e:
                    st.error(f"Failed to analyze interactions: {e}")
        else:
            st.warning("The data does not contain 'From' and 'To' columns for wallet analysis. Please select another type of analysis or upload different data.")
else:
    # Information about missing data
    st.warning("Upload a transaction data file using the menu on the left for analysis.")
    
    # Example data for information
    st.info("""
    ### What data can be analyzed?
    
    The application supports analysis of transaction files in CSV and TXT formats with the following columns:
    - Human Time - transaction time
    - From - sender address
    - To - receiver address
    - Value - transaction amount
    - Token Address - token address
    - Amount - number of tokens
    - Signature - transaction signature
    
    Not all columns are required; the analysis will adapt to the available data.
    """)

# Information at the bottom of the page
st.markdown("---")
st.markdown("#### Application Information")
st.info("""
This is an interactive application for transaction analysis. Use the menu on the left to select different types of analysis.
""")