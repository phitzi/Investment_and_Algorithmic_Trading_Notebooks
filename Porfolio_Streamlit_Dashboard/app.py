# streamlit run .\app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="Portfolio Diversification Dashboard",
    page_icon="📊",
    layout="wide"
)

# Function to load portfolio data
@st.cache_data
def load_portfolio_data():
    try:
        with open("portfolio.json", "r") as f:
            portfolio_data = json.load(f)
        return portfolio_data
    except FileNotFoundError:
        st.error("portfolio.json file not found in the current directory.")
        st.info("Current working directory: " + os.getcwd())
        return {}

# Function to load price data
@st.cache_data
def load_price_data():
    try:
        price_data = pd.read_csv("prices.csv")
        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data.set_index('date', inplace=True)
        return price_data
    except FileNotFoundError:
        st.error("prices.csv file not found in the current directory.")
        st.info("Current working directory: " + os.getcwd())
        return pd.DataFrame()

# Function to load currency data
@st.cache_data
def load_currency_data():
    try:
        with open("currencies.json", "r") as f:
            currency_data = json.load(f)
        return currency_data
    except FileNotFoundError:
        st.error("currencies.json file not found in the current directory.")
        st.info("Current working directory: " + os.getcwd())
        return {}

# Load data
try:
    portfolio_data = load_portfolio_data()
    price_data = load_price_data()
    currency_data = load_currency_data()

    # Convert portfolio data to DataFrame
    portfolio_df = pd.DataFrame([
        {
            'Asset': asset,
            'Name': details.get('name', ''),
            'Type': details.get('type', ''),
            'Quantity': details.get('quantity', 0),
            'Currency': details.get('currency', ''),
            'Last Price': details.get('last_price', 0),
            'Change (%)': details.get('change_perc', 0),
            'Total Value in Main Cur': details.get('total_value_in_main_currency', 0),
            'PnL in Main Cur': details.get('pnl_in_main_currency', 0),
            'Weight (%)': details.get('weight', 0)
        }
        for asset, details in portfolio_data.items()
    ])

    # Display header with refresh button
    header_col1, header_col2 = st.columns([0.85, 0.15])
    with header_col1:
        st.title("My Portfolio 360 Dashboard")
    with header_col2:
        if st.button("🔄 Refresh", help="Refresh dashboard data"):
            # Clear the cache to reload fresh data
            st.cache_data.clear()
            st.rerun()

    # Create 2 columns for the pie charts
    col1, col2 = st.columns([0.2,0.8])

    with col1:
        # Calculate total portfolio value and changes
        total_value = sum(details.get('total_value_in_main_currency', 0) for details in portfolio_data.values())
        total_pnl = sum(details.get('pnl_in_main_currency', 0) for details in portfolio_data.values())

        # Calculate daily change value
        total_previous = sum(details.get('previous_price', 0) * details.get('quantity', 0) /
                             currency_data[details.get('currency', 'EUR')]['rate_to_main']
                             for details in portfolio_data.values())
        daily_change = ((total_value - total_previous) / total_previous) * 100 if total_previous else 0

        # Create 3 columns for centering the metrics
        left_spacer, center_col, right_spacer = st.columns([1,2, 1])

        with center_col:
            st.metric(label="Total Portfolio Value", value=f"€{total_value:,.2f}", delta=f"{daily_change:.2f}%")
            st.metric(label="Total P&L", value=f"€{total_pnl:,.2f}",
                      delta=f"{(total_pnl / total_value) * 100:.2f}%" if total_value else "0%")
            st.metric(label="Number of Assets", value=f"{len(portfolio_data)}")

            # Get weights from portfolio data
            weights = np.array([asset_data['weight'] for asset, asset_data in portfolio_data.items()])

            # Calculate daily returns for all assets
            returns = price_data.pct_change().dropna()


            # 1. Herfindahl-based PDI
            def calculate_herfindahl_pdi(weights):
                try:
                    if len(weights) == 0:
                        return 0.0  # No diversification if no weights

                    # Normalize weights to sum to 1
                    normalized_weights = weights / weights.sum()

                    # Calculate Herfindahl index
                    hi = np.sum(normalized_weights ** 2)

                    return 1 - hi
                except Exception as e:
                    st.warning(f"Error calculating Herfindahl-based PDI: {e}")
                    return 0.0


            # 2. Diversification Ratio
            def calculate_diversification_ratio(weights, returns_data):
                try:
                    # Filter returns data to include only assets in the portfolio
                    assets = list(portfolio_data.keys())

                    # Check which assets have price data
                    available_assets = [asset for asset in assets if asset in returns_data.columns]

                    if not available_assets:
                        return 1.0  # Default value if no assets have price data

                    # Filter weights to include only assets with price data
                    asset_indices = [i for i, asset in enumerate(assets) if asset in available_assets]
                    filtered_weights = weights[asset_indices]

                    # Normalize weights to sum to 1
                    filtered_weights = filtered_weights / filtered_weights.sum()

                    # Filter returns data
                    filtered_returns = returns_data[available_assets].copy().dropna()

                    if filtered_returns.empty:
                        return 1.0  # Default value if no data

                    # Calculate individual volatilities (standard deviations)
                    individual_volatilities = filtered_returns.std() * np.sqrt(252)  # Annualized

                    # Calculate covariance matrix
                    cov_matrix = filtered_returns.cov() * 252  # Annualized

                    # Portfolio volatility
                    port_volatility = np.sqrt(filtered_weights.T @ cov_matrix.values @ filtered_weights)

                    # Weighted average individual volatility
                    avg_individual_vol = np.sum(filtered_weights * individual_volatilities.values)

                    return avg_individual_vol / port_volatility if port_volatility > 0 else 1.0
                except Exception as e:
                    st.warning(f"Error calculating Diversification Ratio: {e}")
                    return 1.0


            # 3. Effective Number of Bets
            def calculate_effective_bets(weights, returns_data):
                try:
                    # Filter returns data to include only assets in the portfolio
                    assets = list(portfolio_data.keys())

                    # Check which assets have price data
                    available_assets = [asset for asset in assets if asset in returns_data.columns]

                    if not available_assets:
                        return 1.0  # Default value if no assets have price data

                    # Filter weights to include only assets with price data
                    asset_indices = [i for i, asset in enumerate(assets) if asset in available_assets]
                    filtered_weights = weights[asset_indices]

                    # Normalize weights to sum to 1
                    filtered_weights = filtered_weights / filtered_weights.sum()

                    # Filter returns data
                    filtered_returns = returns_data[available_assets].copy().dropna()

                    if filtered_returns.empty:
                        return 1.0  # Default value if no data

                    # Calculate correlation matrix
                    corr_matrix = filtered_returns.corr()

                    # Calculate ENB denominator
                    denominator = 0
                    n = len(filtered_weights)
                    for i in range(n):
                        for j in range(n):
                            denominator += filtered_weights[i] * filtered_weights[j] * corr_matrix.iloc[i, j]

                    return 1 / denominator if denominator > 0 else 1.0
                except Exception as e:
                    st.warning(f"Error calculating Effective Number of Bets: {e}")
                    return 1.0


            # Calculate the metrics
            herfindahl_pdi = calculate_herfindahl_pdi(weights)
            div_ratio = calculate_diversification_ratio(weights, returns)
            enb = calculate_effective_bets(weights, returns)

            # Create gauge charts for each metric

            # 1. Herfindahl-based PDI gauge
            fig_pdi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=herfindahl_pdi,
                title={'text': "Herfindahl-based PDI"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.7], 'color': 'red'},
                        {'range': [0.7, 0.9], 'color': 'yellow'},
                        {'range': [0.9, 1], 'color': 'green'}
                    ],
                }
            ))
            fig_pdi.update_layout(height=150, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_pdi, use_container_width=True)

            # 2. Diversification Ratio gauge
            fig_dr = go.Figure(go.Indicator(
                mode="gauge+number",
                value=div_ratio,
                title={'text': "Diversification Ratio"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 1.1], 'color': 'red'},
                        {'range': [1.1, 1.4], 'color': 'yellow'},
                        {'range': [1.4, 2], 'color': 'green'}
                    ],
                }
            ))
            fig_dr.update_layout(height=150, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_dr, use_container_width=True)

            # 3. Effective Number of Bets gauge
            fig_enb = go.Figure(go.Indicator(
                mode="gauge+number",
                value=enb,
                title={'text': "Effective Number of Bets"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 1.5], 'color': 'red'},
                        {'range': [1.5, 2.5], 'color': 'yellow'},
                        {'range': [2.5, 5], 'color': 'green'}
                    ],
                }
            ))
            fig_enb.update_layout(height=150, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_enb, use_container_width=True)

    with col2:

        # Display the DataFrame
        st.dataframe(
            portfolio_df,
            hide_index=True,
            column_config={
                'Weight (%)': st.column_config.NumberColumn(format="%.2f%%"),
                'Last Price': st.column_config.NumberColumn(format="%.2f")
            }
        )

        tab1, tab2, tab3 = st.tabs(["Allocations", "Exposures", "Correlations"])

        with tab1:
            col_weights, col_types, col_currencies = st.columns(3)

        with tab2:
            col_countries, col_sectors = st.columns(2)

        with tab3:

            # Add dropdown for selecting window size
            window_options = [10, 30, 90, 180, 352, 720]
            selected_window = st.selectbox(
                "Select rolling window size (days):",
                window_options,
                index=2  # Default to 90 days (index 2)
            )

            # Get all assets from price_data
            assets = price_data.columns.tolist()

            # Filter out assets with too many missing values
            valid_assets = []
            for asset in assets:
                if price_data[asset].count() >= selected_window:  # Ensure enough data for selected window
                    valid_assets.append(asset)

            # Create checkboxes for asset selection
            st.write("Select assets to include in correlation analysis:")

            # Use columns to display checkboxes in a grid
            checkbox_cols = st.columns(4)  # Display in 4 columns

            # Dictionary to store checkbox states
            selected_assets = {}

            # Create checkboxes for each asset
            for i, asset in enumerate(valid_assets):
                col_idx = i % 4  # Determine which column to place the checkbox
                with checkbox_cols[col_idx]:
                    selected_assets[asset] = st.checkbox(asset, value=True)  # Default to selected

            # Get list of selected assets
            selected_asset_list = [asset for asset, selected in selected_assets.items() if selected]

            # Check if we have enough assets selected
            if len(selected_asset_list) < 2:
                st.warning("Please select at least two assets to show correlations.")
            else:
                # Generate pairs from selected assets
                import itertools
                asset_pairs = list(itertools.combinations(selected_asset_list, 2))

                # Create a single plot for selected correlations
                num_pairs = len(asset_pairs)

                # Create a figure for all correlations
                fig = go.Figure()

                # Dictionary to store mean correlations for the legend
                mean_correlations = {}

                # Add each asset pair correlation to the figure
                for pair_idx, (asset1, asset2) in enumerate(asset_pairs):
                    # Filter out NaN values for the selected assets
                    valid_data = price_data[[asset1, asset2]].dropna()

                    if len(valid_data) >= selected_window:  # Ensure we have enough data for the selected window
                        rolling_corr = valid_data[asset1].rolling(window=selected_window).corr(valid_data[asset2])

                        # Create a DataFrame for plotting
                        corr_df = pd.DataFrame({
                            'date': rolling_corr.index,
                            'correlation': rolling_corr.values
                        }).dropna()

                        if not corr_df.empty:
                            # Calculate mean correlation
                            mean_corr = rolling_corr.mean()
                            mean_correlations[f"{asset1} vs {asset2}"] = mean_corr

                            # Add line to the plot
                            fig.add_trace(
                                go.Scatter(
                                    x=corr_df['date'],
                                    y=corr_df['correlation'],
                                    mode='lines',
                                    name=f"{asset1} vs {asset2} (Mean: {mean_corr:.2f})"
                                )
                            )

                # Only display the plot if we have valid pairs
                if len(fig.data) > 0:
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Correlation",
                        height=600,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.5,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=10, r=10, t=50, b=100)
                    )

                    # Add a horizontal line at y=0 for reference
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")

                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)

                    # Display a warning if there are many pairs that might make the plot cluttered
                    if num_pairs > 10:
                        st.warning(f"There are {num_pairs} asset pairs in the plot, which might make it difficult to read. Consider selecting fewer assets to reduce the number of pairs.")
                else:
                    st.warning("No valid correlation data found for the selected asset pairs.")

    with col_weights:

        # Create DataFrame directly from portfolio data
        asset_weights_df = pd.DataFrame([
            {'Asset': asset, 'Weight': asset_data['weight']}
            for asset, asset_data in portfolio_data.items()
        ])

        # Create pie chart using Plotly
        fig_assets = px.pie(
            asset_weights_df,
            values='Weight',
            names='Asset',
            title='Portfolio Asset Allocation'
        )
        st.plotly_chart(fig_assets, use_container_width=True)

    with col_types:

        portfolio_df = pd.DataFrame([
            {'Asset': asset, 'type': asset_data['type'], 'Weight': asset_data.get('weight', 0)}
            for asset, asset_data in portfolio_data.items()
        ])

        # Group by currency and sum the weights
        types_weights_df = portfolio_df.groupby('type')['Weight'].sum().reset_index()

        # Create pie chart using Plotly
        fig_assets = px.pie(
            types_weights_df,
            values='Weight',
            names='type',
            title='Portfolio Asset Allocation'
        )
        st.plotly_chart(fig_assets, use_container_width=True)

    with col_currencies:

        portfolio_df = pd.DataFrame([
            {'Asset': asset, 'Currency': asset_data['currency'], 'Weight': asset_data.get('weight', 0)}
            for asset, asset_data in portfolio_data.items()
        ])

        # Group by currency and sum the weights
        currency_weights_df = portfolio_df.groupby('Currency')['Weight'].sum().reset_index()

        # Create pie chart using Plotly
        fig_assets = px.pie(
            currency_weights_df,
            values='Weight',
            names='Currency',
            title='Currency Exposure'
        )
        st.plotly_chart(fig_assets, use_container_width=True)

    with col_countries:

        country_exposure = defaultdict(float)
        total_portfolio_value = 0

        for symbol, details in portfolio_data.items():
            # Get the position value in main currency
            position_value = details.get('total_value_in_main_currency', 0)
            total_portfolio_value += position_value

            # Handle both simple sector strings and sector dictionaries
            country_data = details.get('country_exposure', '')

            if isinstance(country_data, str):
                # Direct sector assignment
                country_exposure[country_data] += position_value
            elif isinstance(country_data, dict):
                # For ETFs with sector breakdown
                for country, allocation in country_data.items():
                    # Get the equity percentage and convert to decimal
                    equity_pct = float(allocation.get('Relative_to_Category', 0)) / 100
                    # Add weighted position value to sector
                    country_exposure[country] += position_value * equity_pct

        df_country_exposure = pd.DataFrame(
            {'country': list(country_exposure.keys()), 'value': list(country_exposure.values())})

        # Calculate percentages
        total_value = df_country_exposure['value'].sum()
        df_country_exposure['percentage'] = df_country_exposure['value'] / total_value * 100

        # Group countries with less than 3% into "Other"
        mask = df_country_exposure['percentage'] < 3
        other_small = df_country_exposure[mask]['value'].sum()
        df_country_exposure = pd.concat([
            df_country_exposure[~mask],
            pd.DataFrame({'country': ['Other'], 'value': [other_small]}) if other_small > 0 else pd.DataFrame()
        ]).reset_index(drop=True)

        # Sort by value in descending order
        df_country_exposure = df_country_exposure.sort_values('value', ascending=False)

        # If still more than 4 countries, keep top 4 and group rest into "Other"
        if len(df_country_exposure) > 4:
            top_4 = df_country_exposure.head(4)
            other_sum = df_country_exposure.iloc[4:]['value'].sum()
            df_country_exposure = pd.concat([
                top_4,
                pd.DataFrame({'country': ['Other'], 'value': [other_sum]})
            ]).reset_index(drop=True)

        # Create pie chart using Plotly
        fig_assets = px.pie(
            df_country_exposure,
            values='value',
            names='country',
            title='Countries Exposure'
        )
        st.plotly_chart(fig_assets, use_container_width=True)

    with col_sectors:

        sector_exposure = defaultdict(float)
        total_portfolio_value = 0

        for symbol, details in portfolio_data.items():
            # Get the position value in main currency
            position_value = details.get('total_value_in_main_currency', 0)
            total_portfolio_value += position_value

            # Handle both simple sector strings and sector dictionaries
            sector_data = details.get('sector', '')

            if isinstance(sector_data, str):
                # Direct sector assignment
                sector_exposure[sector_data] += position_value
            elif isinstance(sector_data, dict):
                # For ETFs with sector breakdown
                for sector, allocation in sector_data.items():
                    # Get the equity percentage and convert to decimal
                    equity_pct = float(allocation.get('Relative_to_Category', 0)) / 100
                    # Add weighted position value to sector
                    sector_exposure[sector] += position_value * equity_pct

        df_sector_exposure = pd.DataFrame(
            {'sector': list(sector_exposure.keys()), 'value': list(sector_exposure.values())})

        # Calculate percentages
        total_value = df_sector_exposure['value'].sum()
        df_sector_exposure['percentage'] = df_sector_exposure['value'] / total_value * 100

        # Group sectors with less than 3% into "Other"
        mask = df_sector_exposure['percentage'] < 3
        other_small = df_sector_exposure[mask]['value'].sum()
        df_sector_exposure = pd.concat([
            df_sector_exposure[~mask],
            pd.DataFrame({'sector': ['Other'], 'value': [other_small]}) if other_small > 0 else pd.DataFrame()
        ]).reset_index(drop=True)

        # Sort by value in descending order
        df_sector_exposure = df_sector_exposure.sort_values('value', ascending=False)

        # If still more than 4 sectors, keep top5 and group rest into "Other"
        if len(df_sector_exposure) > 5:
            top_5 = df_sector_exposure.head(5)
            other_sum = df_sector_exposure.iloc[5:]['value'].sum()
            df_sector_exposure = pd.concat([
                top_5,
                pd.DataFrame({'sector': ['Other'], 'value': [other_sum]})
            ]).reset_index(drop=True)

        # Create pie chart using Plotly
        fig_assets = px.pie(
            df_sector_exposure,
            values='value',
            names='sector',
            title='Sector Exposure'
        )
        st.plotly_chart(fig_assets, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please make sure all required files (portfolio.json, prices.csv, currencies.json) are in the correct location.")
