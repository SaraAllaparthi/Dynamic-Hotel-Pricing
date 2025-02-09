import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_data():
    """
    Load hotel bookings data from a CSV file located in the same folder.
    Make sure the file 'hotel_bookings.csv' is present in your repository.
    """
    file_path = os.path.join(os.path.dirname(__file__), "hotel_bookings.csv")
    try:
        df = pd.read_csv(file_path)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
        return df
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}")
        return None

def preprocess_occupancy_data(df):
    """
    Preprocess the data for Prophet:
      - Convert the 'reservation_status_date' column to datetime.
      - Filter out canceled bookings (is_canceled == 0).
      - Group by date to compute daily occupancy.
      - Rename columns to 'ds' (date) and 'y' (value) for Prophet.
    """
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df_not_canceled = df[df['is_canceled'] == 0]
    occupancy = df_not_canceled.groupby('reservation_status_date').size().reset_index(name='occupancy')
    prophet_df = occupancy.rename(columns={'reservation_status_date': 'ds', 'occupancy': 'y'})
    return prophet_df

# -------------------------------
# Forecasting with Prophet
# -------------------------------

def forecast_model(prophet_df, forecast_periods):
    """
    Initialize and fit the Prophet model, then generate a forecast.
    """
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    return model, forecast

def visualize_forecast(model, forecast):
    """
    Plot the forecast and its components with enhanced formatting.
    """
    st.markdown("### Forecast Plot")
    fig1 = model.plot(forecast)
    plt.title("Occupancy Forecast")
    plt.xlabel("Date")
    plt.ylabel("Predicted Occupancy")
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)
    
    st.markdown("### Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

def evaluate_forecast(model):
    """
    Evaluate forecast accuracy using Prophet's cross-validation.
    This section displays key performance metrics such as MAE and RMSE.
    """
    st.markdown("### Forecast Accuracy Evaluation")
    df_cv = cross_validation(model, initial='90 days', period='15 days', horizon='30 days')
    df_p = performance_metrics(df_cv)
    st.write("Performance Metrics (e.g., MAE, RMSE):")
    st.dataframe(df_p.head())

# -------------------------------
# Additional Feature Functions
# -------------------------------

def competitor_dynamic_pricing(adj_factor):
    """
    This dashboard simulates competitor data to help adjust your pricing.
    The manager can adjust the competitor adjustment factor to see how it affects
    the recommended base price.
    """
    st.markdown("## Competitor-Based Dynamic Pricing")
    st.markdown(
        """
        **Purpose:** Compare your hotel with competitors and adjust pricing accordingly.
        
        **How it Works:**  
        The app simulates competitor rates and occupancy. An adjustment factor is applied 
        to benchmark your rate relative to the market.
        """
    )
    # Simulated competitor data (in a production system, this would come from an API or scrape)
    competitor_data = pd.DataFrame({
        'hotel': ['Competitor A', 'Competitor B', 'Competitor C'],
        'rate': [120.0, 135.0, 110.0],
        'occupancy_rate': [0.85, 0.90, 0.80]
    })
    st.write("**Competitor Data:**")
    st.dataframe(competitor_data)
    avg_competitor_rate = competitor_data['rate'].mean()
    avg_competitor_occ = competitor_data['occupancy_rate'].mean()
    # The adjustment factor (entered as a percentage, e.g., 5 for 5%) adjusts the base price.
    adjusted_rate = avg_competitor_rate * (1 + (adj_factor / 100))
    st.write(f"**Average Competitor Rate:** ${avg_competitor_rate:.2f}")
    st.write(f"**Adjusted Recommended Base Price:** ${adjusted_rate:.2f}")

def demand_forecasting_with_events(prophet_df):
    """
    This feature incorporates external events (e.g., holidays, festivals) into the forecast.
    The manager can enter event dates to see how they affect demand.
    """
    st.markdown("## Demand Forecasting with Events")
    st.markdown(
        """
        **Purpose:** Account for external events (like holidays or local festivals) that impact occupancy.
        
        **How it Works:**  
        Enter event dates, and the model incorporates these as holidays to adjust the forecast.
        """
    )
    # Allow the manager to input event dates (comma-separated)
    events_text = st.text_input("Enter Event Dates (YYYY-MM-DD, comma separated)", "2015-07-04,2015-12-25")
    event_dates = [d.strip() for d in events_text.split(",") if d.strip()]
    holidays = pd.DataFrame({
        'holiday': 'event',
        'ds': pd.to_datetime(event_dates),
        'lower_window': 0,
        'upper_window': 1,
    })
    st.write("**Events/Holidays Added:**")
    st.dataframe(holidays)
    
    # Re-run forecasting with the additional holiday data
    model = Prophet(holidays=holidays, daily_seasonality=True)
    model.fit(prophet_df)
    forecast_periods = st.slider("Forecast Periods (days)", 30, 180, 90, step=10)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    
    st.write("**Forecast Data (Last 5 Rows):**")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    fig = model.plot(forecast)
    plt.title("Demand Forecast with Events")
    plt.xlabel("Date")
    plt.ylabel("Predicted Occupancy")
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

def personalized_promotions():
    """
    This feature analyzes customer segments and suggests personalized promotions
    to increase occupancy during off-peak seasons.
    """
    st.markdown("## Personalized Promotions")
    st.markdown(
        """
        **Purpose:** Increase occupancy by offering tailored promotions to different customer segments.
        
        **How it Works:**  
        The app examines sample customer data and recommends promotions based on customer type.
        """
    )
    customer_data = pd.DataFrame({
        'customer_type': ['Transient', 'Contract', 'Group', 'Transient', 'Transient'],
        'average_spend': [100, 150, 80, 120, 90]
    })
    st.write("**Customer Data Sample:**")
    st.dataframe(customer_data)
    promotions = {
        'Transient': 'Offer 10% off for early bookings',
        'Contract': 'Provide a complimentary room upgrade',
        'Group': 'Offer special group discounts'
    }
    chosen_segment = st.selectbox("Select Customer Segment", list(promotions.keys()), index=0)
    st.write(f"**Suggested Promotion for {chosen_segment} customers:** {promotions[chosen_segment]}")

def los_optimization():
    """
    This feature helps identify the optimal length-of-stay (LOS) based on historical data,
    so that the hotel can maximize revenue per stay.
    """
    st.markdown("## Length-of-Stay (LOS) Optimization")
    st.markdown(
        """
        **Purpose:** Determine the optimal booking window (length-of-stay) that maximizes revenue.
        
        **How it Works:**  
        Analyzes historical LOS data and suggests the optimal LOS.
        """
    )
    los_data = pd.DataFrame({
        'LOS (nights)': [1, 2, 3, 4, 5, 6],
        'average_rate': [100, 95, 90, 85, 80, 75]
    })
    st.write("**Historical LOS Data:**")
    st.dataframe(los_data)
    optimal_index = los_data['average_rate'].idxmax()
    optimal_los = los_data.loc[optimal_index, 'LOS (nights)']
    st.write(f"**Optimal Length of Stay:** {optimal_los} nights")
    st.write("Consider incentivizing this LOS with special pricing or packages.")

def room_upgrade_suggestions():
    """
    This feature identifies guests who have not yet taken advantage of room upgrades,
    creating an opportunity for targeted cross-selling (e.g., upgrade offers, spa packages).
    """
    st.markdown("## Room Upgrade and Cross-Selling Suggestions")
    st.markdown(
        """
        **Purpose:** Identify potential opportunities to offer room upgrades or add-ons to guests.
        
        **How it Works:**  
        Analyzes sample booking history to recommend upgrade offers.
        """
    )
    booking_data = pd.DataFrame({
        'guest_id': [1, 2, 3, 4, 5],
        'current_room_type': ['Standard', 'Standard', 'Deluxe', 'Standard', 'Suite'],
        'past_upgrade': [False, True, False, False, True]
    })
    st.write("**Booking Data:**")
    st.dataframe(booking_data)
    suggestions = booking_data[booking_data['past_upgrade'] == False]
    st.write("**Room Upgrade Suggestions for Guests:**")
    st.dataframe(suggestions[['guest_id', 'current_room_type']])
    st.write("Consider offering packages (e.g., breakfast or spa deals) to encourage upgrades.")

def price_sensitivity_analysis():
    """
    This dashboard analyzes the relationship between room pricing and conversion rates,
    helping to identify price thresholds that may reduce booking rates.
    """
    st.markdown("## Price Sensitivity Analysis")
    st.markdown(
        """
        **Purpose:** Determine the price point at which conversion rates drop significantly,
        so that pricing can be optimized.
        
        **How it Works:**  
        Compares various room prices against conversion rates and identifies a threshold.
        """
    )
    pricing_data = pd.DataFrame({
        'price': [80, 90, 100, 110, 120, 130],
        'conversion_rate': [0.9, 0.88, 0.85, 0.8, 0.75, 0.7]
    })
    st.write("**Pricing and Conversion Data:**")
    st.dataframe(pricing_data)
    threshold = pricing_data[pricing_data['conversion_rate'] < 0.8]['price'].min()
    if pd.isna(threshold):
        st.write("No significant drop in conversion rate detected.")
    else:
        st.write(f"**Price sensitivity threshold detected at:** ${threshold}")
        st.write("Consider setting room rates below this threshold to maintain bookings.")

def last_minute_pricing():
    """
    This feature automatically suggests last-minute price adjustments for unsold rooms
    to help minimize lost revenue.
    """
    st.markdown("## Last-Minute Pricing Adjustments")
    st.markdown(
        """
        **Purpose:** Quickly adjust room prices for unsold inventory just before check-in,
        to fill empty rooms.
        
        **How it Works:**  
        A simple rule reduces the base price by a fixed percentage when unsold inventory exists.
        """
    )
    unsold_inventory = st.number_input("Enter current unsold inventory (rooms)", min_value=0, value=5, step=1)
    current_price = st.number_input("Enter current base price ($)", min_value=0.0, value=100.0, step=1.0)
    st.write(f"**Current Unsold Inventory:** {unsold_inventory} rooms")
    st.write(f"**Current Base Price:** ${current_price}")
    if unsold_inventory > 0:
        adjusted_price = current_price * 0.9  # 10% discount as an example
        st.write(f"**Suggested Last-Minute Price Adjustment:** ${adjusted_price:.2f}")
    else:
        st.write("No adjustments needed for last-minute pricing.")

def competitor_benchmarking_dashboard():
    """
    This dashboard provides a side-by-side comparison of your hotel's pricing and occupancy
    metrics against key competitors.
    """
    st.markdown("## Competitor Benchmarking Dashboard")
    st.markdown(
        """
        **Purpose:** Quickly assess your hotel’s market position relative to competitors.
        
        **How it Works:**  
        Displays competitor data along with visualizations to compare room prices and occupancy.
        """
    )
    competitor_data = pd.DataFrame({
        'hotel': ['Competitor A', 'Competitor B', 'Competitor C', 'Our Hotel'],
        'price': [120, 135, 110, 125],
        'occupancy_rate': [0.85, 0.90, 0.80, 0.88]
    })
    st.write("**Competitor Benchmarking Data:**")
    st.dataframe(competitor_data)
    
    # Enhanced bar chart with annotations
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(competitor_data['hotel'], competitor_data['price'], color=['blue', 'green', 'red', 'purple'])
    ax.set_xlabel("Hotel")
    ax.set_ylabel("Room Price")
    ax.set_title("Competitor Room Prices")
    ax.grid(True, linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    st.pyplot(fig)

# -------------------------------
# Main Application
# -------------------------------

def main():
    st.title("Hotel Pricing Optimization Dashboard")
    st.markdown(
        """
        Welcome! This application is designed to help hotel managers forecast occupancy and generate pricing recommendations for the next 3–6 months.  
        The app integrates market demand forecasting, competitor analysis, and event-based adjustments to suggest optimal room prices.
        """
    )
    
    # Sidebar: Global interactive controls
    st.sidebar.markdown("## Global Settings")
    # Forecast horizon: select number of months (3-6 months)
    forecast_horizon_months = st.sidebar.slider("Forecast Horizon (months)", 3, 6, 3, step=1)
    # Convert months to days (approximation: 1 month ~ 30 days)
    forecast_periods = forecast_horizon_months * 30
    st.sidebar.write(f"Forecasting for approximately {forecast_periods} days.")
    
    # Competitor adjustment factor (in percentage)
    competitor_adjustment = st.sidebar.number_input("Competitor Adjustment Factor (%)", value=0.0, step=0.1)
    
    # Global event inputs
    events_text = st.sidebar.text_input("Enter Event Dates (YYYY-MM-DD, comma separated)", "2015-07-04,2015-12-25")
    
    # Load and preprocess data
    df = load_data()
    if df is None:
        return
    prophet_df = preprocess_occupancy_data(df)
    st.markdown("## Processed Occupancy Data")
    st.dataframe(prophet_df.head())
    
    # Forecasting section
    st.markdown("## Occupancy Forecasting")
    model, forecast = forecast_model(prophet_df, forecast_periods)
    st.write("**Forecast Data (Last 5 Rows):**")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    visualize_forecast(model, forecast)
    evaluate_forecast(model)
    
    # Additional Features (accessible via sidebar)
    st.sidebar.markdown("## Additional Features")
    feature = st.sidebar.selectbox("Select Feature", 
                                   ["Competitor-Based Dynamic Pricing",
                                    "Demand Forecasting with Events",
                                    "Personalized Promotions",
                                    "LOS Optimization",
                                    "Room Upgrade Suggestions",
                                    "Price Sensitivity Analysis",
                                    "Last-Minute Pricing Adjustments",
                                    "Competitor Benchmarking Dashboard"])
    
    if feature == "Competitor-Based Dynamic Pricing":
        competitor_dynamic_pricing(competitor_adjustment)
    elif feature == "Demand Forecasting with Events":
        demand_forecasting_with_events(prophet_df)
    elif feature == "Personalized Promotions":
        personalized_promotions()
    elif feature == "LOS Optimization":
        los_optimization()
    elif feature == "Room Upgrade Suggestions":
        room_upgrade_suggestions()
    elif feature == "Price Sensitivity Analysis":
        price_sensitivity_analysis()
    elif feature == "Last-Minute Pricing Adjustments":
        last_minute_pricing()
    elif feature == "Competitor Benchmarking Dashboard":
        competitor_benchmarking_dashboard()
    
    # Export forecast data option
    csv_data = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast Data as CSV", data=csv_data, file_name="forecast_data.csv", mime="text/csv")

if __name__ == '__main__':
    main()
